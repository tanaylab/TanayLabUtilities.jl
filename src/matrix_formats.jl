"""
Deal with (some) of the matrix formats. This obviously can't be compherensive but it should cover the matrix types
we have encountered so far and hopefully falls back to reasonable defaults for more exotic matrix types.

In Julia, many array types are wrappers around "parent" arrays. The specific wrappers we deal with in most cases are
`NamedArray` which adds names to the rows and/or columns, `PermutedDimsArray` which flips the order of the axes,
`Transpose` and `Adjoint` which likewise flip the axes (`Adjoint` also transforms complex values), and `ReadOnlyArray`
which prevents mutating the array. And then there are more transformative wrappers such as `SubArray`, `SparseVector`
and `SparseMatrixCSC`, `PyArray`, etc.

This makes life difficult. Specifically, you can't rely (much) on the type system to separate code dealing with
different array types. For example, not all `issparse` arrays derive from `AbstractSparseArray` (because you might have
a sparse array wrapped in something). It would have been great if there were `isdense` and `isstrided` functions to
match and libraries actually used them to trigger optimized code but "that would have been too easy".

The code here tries to put this under some control so we can write robust code which "does the right thing", in most
cases, at least when it comes to converting between formats. This means we are forced to provide alternatives to some
built-in functions (for example, copying arrays). Sigh.
"""
module MatrixFormats

export bestify
export colptr
export copy_array
export dense_mask_vector
export densify
export indtype_for_size
export nzind
export nzval
export rowval
export similar_array
export sparse_matrix_csc
export sparse_vector
export sparse_mask_vector
export sparsify

using ..Brief
using ..Documentation
using ..MatrixLayouts
using ..ReadOnlyArrays
using ..Types

using LinearAlgebra
using NamedArrays
using SparseArrays

function Brief.brief(array::AbstractArray)::String
    return brief_array(array, String[]; transposed = false)
end

function brief_array(array::ReadOnlyArray, prefixes::Vector{String}; transposed::Bool)::String
    push!(prefixes, "ReadOnly")
    return brief_array(parent(array), prefixes; transposed)
end

function brief_array(array::NamedArray, prefixes::Vector{String}; transposed::Bool)::String
    push!(prefixes, "Named")
    return brief_array(parent(array), prefixes; transposed)
end

function brief_array(
    array::PermutedDimsArray{T, 2, P, IP, A},
    prefixes::Vector{String};
    transposed::Bool,
)::String where {T, P, IP, A}
    if P == (1, 2)
        push!(prefixes, "!Permute")
    elseif P == (2, 1)
        push!(prefixes, "Permute")
        transposed = !transposed
    else
        @assert false
    end
    return brief_array(parent(array), prefixes; transposed)
end

function brief_array(array::Transpose, prefixes::Vector{String}; transposed::Bool)::String
    push!(prefixes, "Transpose")
    return brief_array(parent(array), prefixes; transposed = !transposed)
end

function brief_array(array::Adjoint, prefixes::Vector{String}; transposed::Bool)::String
    push!(prefixes, "Adjoint")
    return brief_array(parent(array), prefixes; transposed = !transposed)
end

function brief_array(array::DenseArray, prefixes::Vector{String}; transposed::Bool)::String
    push!(prefixes, "Dense")
    return format_brief_array(array, prefixes; transposed)
end

function brief_array(array::AbstractArray, prefixes::Vector{String}; transposed::Bool)::String
    if issparse(array)
        push!(prefixes, "Sparse $(SparseArrays.indtype(array)) $(percent(length(array.nzval), length(array)))")
        return format_brief_array(array, prefixes; transposed)
    else
        push!(prefixes, string(nameof(typeof(array))))  # UNTESTED
        return format_brief_array(array, prefixes; transposed)  # UNTESTED
    end
end

function format_brief_array(vector::AbstractVector, prefixes::Vector{String}; transposed::Bool)::String  # NOLINT
    suffix = ""

    try
        if strides(vector) != (1,)
            suffix = "; Strided"  # UNTESTED
        elseif !("Dense" in prefixes)
            suffix = "; Dense"  # UNTESTED
        end
    catch
    end

    type = eltype(vector)
    if type <: AbstractString
        type = "Str"
    end

    return "$(length(vector)) x $(type) ($(join(prefixes, ", "))$(suffix)$(mask_suffix(vector)))"
end

function format_brief_array(matrix::AbstractMatrix, prefixes::Vector{String}; transposed::Bool)::String
    suffix = ""

    try
        matrix_strides = strides(matrix)
        matrix_sizes = size(matrix)
        if matrix_strides != (1, matrix_sizes[1]) && matrix_strides != (matrix_sizes[2], 1)
            suffix = "; Strided"  # UNTESTED
        elseif !("Dense" in prefixes)
            suffix = "; Dense"  # FLAKY TESTED
        end
    catch
    end

    layout = major_axis(matrix)
    if transposed
        layout = other_axis(layout)
    end

    if layout === nothing
        layout_suffix = "w/o major axis"  # UNTESTED
    else
        layout_suffix = "in $(axis_name(layout))"
    end

    n_rows, n_columns = size(matrix)
    if transposed
        n_rows, n_columns = n_columns, n_rows
    end
    return "$(n_rows) x $(n_columns) x $(eltype(matrix)) $(layout_suffix) ($(join(prefixes, ", "))$(suffix)$(mask_suffix(matrix)))"
end

function mask_suffix(::AbstractArray)::AbstractString
    return ""
end

function mask_suffix(array::AbstractArray{Bool})::AbstractString
    n_true = sum(array)  # NOJET
    if issparse(array) && n_true == nnz(array)
        return ""
    else
        return "; $(percent(sum(array), length(array))) true"  # NOJET
    end
end

"""
    copy_array(array::AbstractArray; eltype::Maybe{Type} = nothing, indtype::Maybe{Type} = nothing)::AbstractArray

Create a copy of an array. This differs from `Base.copy` in the following:

  - Copying a read-only array returns a mutable array. In contrast, both `Base.copy` and `Base.deepcopy` of a
    [`ReadOnlyArray`](@ref) array will return a [`ReadOnlyArray`](@ref) array, which is technically correct, but is
    rather pointless.

  - Copying a `NamedArray` returns a `NamedArray` that shares the names (but not the data storage).

  - Copying will preserve the layout of the data; for example, copying a `Transpose` array is still a `Transpose` array.
    In contrast, while `Base.deepcopy` will preserve the layout, `Base.copy` will silently [`relayout`](@ref
    TanayLabUtilities.MatrixLayouts.relayout) the matrix, which is both expensive and unexpected.

  - Copying a sparse vector or matrix gives a sparse result. Copying anything else gives a simple dense array regardless
    of the original type. This is done because a `deepcopy` of `PyArray` will still share the underlying buffer, which
    removes the whole point of doing a copy. Sigh.

  - Copying a vector of anything derived from `AbstractString` returns a vector of `AbstractString`.

  - You can override the `eltype` of the array (and/or the `indtype`, if it is sparse).

```jldoctest
using Test

base = [0 1 2; 3 4 0]

# Dense

@test brief(base) == "2 x 3 x Int64 in Columns (Dense)"
@test brief(copy_array(base)) == "2 x 3 x Int64 in Columns (Dense)"
@test copy_array(base) == base
@test copy_array(base) !== base

@test copy_array(base; eltype = Int32) == base
@test brief(copy_array(base; eltype = Int32)) == "2 x 3 x Int32 in Columns (Dense)"

# Sparse

using SparseArrays

sparse = SparseMatrixCSC(base)
@test copy_array(sparse) == sparse
@test copy_array(sparse) !== sparse
@test brief(sparse) == "2 x 3 x Int64 in Columns (Sparse Int64 67%)"
@test brief(copy_array(sparse)) == "2 x 3 x Int64 in Columns (Sparse Int64 67%)"

@test copy_array(sparse; eltype = Int32) == sparse
@test brief(copy_array(sparse; eltype = Int32)) == "2 x 3 x Int32 in Columns (Sparse Int64 67%)"

@test copy_array(sparse; indtype = Int8) == sparse
@test brief(copy_array(sparse; indtype = Int8)) == "2 x 3 x Int64 in Columns (Sparse Int8 67%)"

# ReadOnly

read_only = read_only_array(base)
@test brief(read_only) == "2 x 3 x Int64 in Columns (ReadOnly, Dense)"
@test brief(copy_array(read_only)) == "2 x 3 x Int64 in Columns (Dense)"
@test copy_array(read_only) == read_only
@test copy_array(read_only) !== base

# Named

using NamedArrays

named = NamedArray(base)
@test brief(named) == "2 x 3 x Int64 in Columns (Named, Dense)"
@test brief(copy_array(named)) == "2 x 3 x Int64 in Columns (Named, Dense)"
@test copy_array(named) == named
@test parent(copy_array(named)) !== base

# Permuted

permuted = PermutedDimsArray(base, (2, 1))
@test brief(permuted) == "3 x 2 x Int64 in Rows (Permute, Dense)"
@test brief(copy_array(permuted)) == "3 x 2 x Int64 in Rows (Permute, Dense)"
@test copy_array(permuted) == permuted
@test parent(copy_array(permuted)) !== base

unpermuted = PermutedDimsArray(base, (1, 2))
@test brief(unpermuted) == "2 x 3 x Int64 in Columns (!Permute, Dense)"
@test brief(copy_array(unpermuted)) == "2 x 3 x Int64 in Columns (!Permute, Dense)"
@test copy_array(unpermuted) == unpermuted
@test parent(copy_array(unpermuted)) !== base

# LinearAlgebra

using LinearAlgebra

transposed = transpose(base)
@test brief(transposed) == "3 x 2 x Int64 in Rows (Transpose, Dense)"
@test brief(copy_array(transposed)) == "3 x 2 x Int64 in Rows (Transpose, Dense)"
@test copy_array(transposed) == transposed
@test parent(copy_array(transposed)) !== base

adjointed = adjoint(base)
@test brief(adjointed) == "3 x 2 x Int64 in Rows (Adjoint, Dense)"
@test brief(copy_array(adjointed)) == "3 x 2 x Int64 in Rows (Adjoint, Dense)"
@test copy_array(adjointed) == adjointed
@test parent(copy_array(adjointed)) !== base

println("OK")

# output

OK
```

```jldoctest
using Test

# Dense

base = [0, 1, 2]

@test brief(base) == "3 x Int64 (Dense)"
@test brief(copy_array(base)) == "3 x Int64 (Dense)"
@test copy_array(base) == base
@test copy_array(base) !== base

# Sparse

using SparseArrays

sparse = SparseVector(base)
@test brief(sparse) == "3 x Int64 (Sparse Int64 67%)"
@test brief(copy_array(sparse)) == "3 x Int64 (Sparse Int64 67%)"
@test copy_array(sparse) == sparse
@test copy_array(sparse) !== sparse

# ReadOnly

read_only = read_only_array(base)
@test brief(read_only) == "3 x Int64 (ReadOnly, Dense)"
@test brief(copy_array(read_only)) == "3 x Int64 (Dense)"
@test copy_array(read_only) == read_only
@test copy_array(read_only) !== base

# Named

using NamedArrays

named = NamedArray(base)
@test brief(named) == "3 x Int64 (Named, Dense)"
@test brief(copy_array(named)) == "3 x Int64 (Named, Dense)"
@test copy_array(named) == named
@test parent(copy_array(named)) !== base

# LinearAlgebra

using LinearAlgebra

transposed = transpose(base)
@test brief(transposed) == "3 x Int64 (Transpose, Dense)"
@test brief(copy_array(transposed)) == "3 x Int64 (Transpose, Dense)"
@test copy_array(transposed) == transposed
@test parent(copy_array(transposed)) !== base

adjointed = adjoint(base)
@test brief(adjointed) == "3 x Int64 (Adjoint, Dense)"
@test brief(copy_array(adjointed)) == "3 x Int64 (Adjoint, Dense)"
@test copy_array(adjointed) == adjointed
@test parent(copy_array(adjointed)) !== base

# String

base = split("abc", "")

@test brief(base) == "3 x Str (Dense)"
@test brief(copy_array(base)) == "3 x Str (Dense)"
@test eltype(base) != AbstractString
@test eltype(copy_array(base)) == AbstractString
@test copy_array(base) == base

println("OK")

# output

OK
```
"""
function copy_array(
    matrix::AbstractMatrix;
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractMatrix
    is_sparse, eltype, indtype = parse_sparse_array(matrix; eltype, indtype)

    if is_sparse
        @assert major_axis(matrix) == Columns
        return sparse_matrix_csc(matrix; eltype, indtype)
    elseif major_axis(matrix) == Rows
        return transpose(Matrix{eltype}(transpose(matrix)))  # UNTESTED
    else
        return Matrix{eltype}(matrix)
    end
end

function copy_array(
    vector::AbstractVector;
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractVector
    is_sparse, eltype, indtype = parse_sparse_array(vector; eltype, indtype)

    if is_sparse
        return sparse_vector(vector; eltype, indtype)
    else
        return Vector{eltype}(vector)
    end
end

function copy_array(
    vector::AbstractVector{<:AbstractString};
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,  # NOLINT
)::Vector{AbstractString}
    if eltype === nothing
        eltype = AbstractString
    end

    return Vector{eltype}(vector)  # NOJET
end

function copy_array(array::ReadOnlyArray; eltype::Maybe{Type} = nothing, indtype::Maybe{Type} = nothing)::AbstractArray
    return copy_array(parent(array); eltype, indtype)
end

function copy_array(array::NamedArray; eltype::Maybe{Type} = nothing, indtype::Maybe{Type} = nothing)::NamedArray
    return NamedArray(copy_array(parent(array); eltype, indtype), array.dicts, array.dimnames)
end

function copy_array(
    matrix::PermutedDimsArray{T, 2, P, IP, A};
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::PermutedDimsArray where {T, P, IP, A}
    return PermutedDimsArray(copy_array(parent(matrix); eltype, indtype), P)
end

function copy_array(matrix::Transpose; eltype::Maybe{Type} = nothing, indtype::Maybe{Type} = nothing)::Transpose
    return Transpose(copy_array(parent(matrix); eltype, indtype))
end

function copy_array(matrix::Adjoint; eltype::Maybe{Type} = nothing, indtype::Maybe{Type} = nothing)::Adjoint
    return Adjoint(copy_array(parent(matrix); eltype, indtype))
end

"""
    similar_array(
        array::AbstractArray;
        [value::Any = undef,
        eltype::Maybe{Type} = nothing]
    )::AbstractArray
    end

Return an `array` (vector or a matrix) similar to the given one. By default the data has the same `eltype` as the
original, and is uninitialized unless you specify a `value`. The returned data is always dense (`Vector` or `Matrix`).

This is different from `similar` in that it will preserve the layout of a matrix (for example, `similar_array` of a
`transpose` will also be a `transpose`). Also, `similar_array` of a `NamedArray` will be another `NamedArray` sharing
the axes with the original, and `ReadOnlyArray` wrappers are stripped from the result.

```jldoctest
using Test

base = rand(3, 4)

@test brief(base) == "3 x 4 x Float64 in Columns (Dense)"
@test similar_array(base) !== base
@test brief(similar_array(base)) == "3 x 4 x Float64 in Columns (Dense)"

@test brief(similar_array(base; eltype = Int32)) == "3 x 4 x Int32 in Columns (Dense)"
@test brief(similar_array(base; value = 0.0)) == "3 x 4 x Float64 in Columns (Dense)"
@test all(similar_array(base; value = 0.0) .== 0)

# ReadOnly

read_only = read_only_array(base)
@test brief(read_only) == "3 x 4 x Float64 in Columns (ReadOnly, Dense)"
@test brief(similar_array(read_only)) == "3 x 4 x Float64 in Columns (Dense)"

# Named

using NamedArrays

named = NamedArray(base)
@test brief(named) == "3 x 4 x Float64 in Columns (Named, Dense)"
@test similar_array(named) !== named
@test brief(similar_array(named)) == "3 x 4 x Float64 in Columns (Named, Dense)"

# Permuted

permuted = PermutedDimsArray(base, (2, 1))
@test brief(permuted) == "4 x 3 x Float64 in Rows (Permute, Dense)"
@test similar_array(permuted) !== permuted
@test brief(similar_array(permuted)) == "4 x 3 x Float64 in Rows (Permute, Dense)"

# LinearAlgebra

transposed = transpose(base)
@test brief(transposed) == "4 x 3 x Float64 in Rows (Transpose, Dense)"
@test similar_array(transposed) !== transposed
@test brief(similar_array(transposed)) == "4 x 3 x Float64 in Rows (Transpose, Dense)"

adjointed = adjoint(base)
@test brief(adjointed) == "4 x 3 x Float64 in Rows (Adjoint, Dense)"
@test similar_array(adjointed) !== adjointed
@test brief(similar_array(adjointed)) == "4 x 3 x Float64 in Rows (Adjoint, Dense)"

println("OK")

# output

OK
```
"""
function similar_array(array::AbstractArray; value::Any = undef, eltype::Maybe{Type} = nothing)::AbstractArray
    if eltype === nothing
        eltype = Base.eltype(array)
    end

    similar = Array{eltype}(undef, size(array)...)
    if value != undef
        similar .= value
    end

    return similar
end

function similar_array(array::ReadOnlyArray; value::Any = undef, eltype::Maybe{Type} = nothing)::AbstractArray
    return similar_array(parent(array); value, eltype)
end

function similar_array(array::NamedArray; value::Any = undef, eltype::Maybe{Type} = nothing)::AbstractArray
    return NamedArray(similar_array(parent(array); value, eltype), array.dicts, array.dimnames)
end

function similar_array(
    matrix::PermutedDimsArray{T, 2, P, IP, A};
    value::Any = undef,
    eltype::Maybe{Type} = nothing,
)::AbstractArray where {T, P, IP, A}
    return PermutedDimsArray(similar_array(parent(matrix); value, eltype), P)
end

function similar_array(array::Transpose; value::Any = undef, eltype::Maybe{Type} = nothing)::AbstractArray
    return Transpose(similar_array(parent(array); value, eltype))
end

function similar_array(array::Adjoint; value::Any = undef, eltype::Maybe{Type} = nothing)::AbstractArray
    return Adjoint(similar_array(parent(array); value, eltype))
end

"""
    sparsify(
        matrix::AbstractMatrix;
        copy::Bool = false,
        eltype::Maybe{Type} = nothing,
        indtype::Maybe{Type} = nothing
    )::AbstractMatrix

    sparsify(
        vector::AbstractVector;
        copy::Bool = false,
        eltype::Maybe{Type} = nothing,
        indtype::Maybe{Type} = nothing
    )::AbstractVector

Return a sparse version of an array, possibly forcing a different `eltype` and/or `indtype`. If given a dense matrix,
the default `indtype` will be [`indtype_for_size`](@ref) for the matrix. This will preserve the matrix layout (for
example, `sparsify` of a transposed matrix will be a transposed matrix). If `copy`, this will create a copy even if it
is already sparse and has the correct `eltype` and `indtype`.


```jldoctest
using Test
using SparseArrays

# Dense

dense = rand(3, 4)
@test sparsify(dense) == dense
@test brief(dense) == "3 x 4 x Float64 in Columns (Dense)"
@test brief(sparsify(dense)) == "3 x 4 x Float64 in Columns (Sparse UInt32 100%)"

# Sparse

sparse = SparseMatrixCSC([0 1 2; 3 4 0])
@test sparsify(sparse) === sparse
@test brief(sparse) == "2 x 3 x Int64 in Columns (Sparse Int64 67%)"

@test sparsify(sparse; copy = true) == sparse
@test sparsify(sparse; copy = true) !== sparse
@test brief(sparsify(sparse)) == "2 x 3 x Int64 in Columns (Sparse Int64 67%)"

@test sparsify(sparse; eltype = Int8) == sparse
@test brief(sparsify(sparse; eltype = Int8)) == "2 x 3 x Int8 in Columns (Sparse Int64 67%)"

@test sparsify(sparse; indtype = Int8) == sparse
@test brief(sparsify(sparse; indtype = Int8)) == "2 x 3 x Int64 in Columns (Sparse Int8 67%)"

# ReadOnly

read_only = read_only_array(sparse)
@test sparsify(read_only) === read_only
@test brief(read_only) == "2 x 3 x Int64 in Columns (ReadOnly, Sparse Int64 67%)"

read_only = read_only_array(dense)
@test sparsify(read_only) == read_only
@test brief(sparsify(read_only)) == "3 x 4 x Float64 in Columns (ReadOnly, Sparse UInt32 100%)"

# Named

using NamedArrays

named = NamedArray(sparse)
@test sparsify(named) === named
@test brief(named) == "2 x 3 x Int64 in Columns (Named, Sparse Int64 67%)"

named = NamedArray(dense)
@test sparsify(named) == named
@test brief(sparsify(named)) == "3 x 4 x Float64 in Columns (Named, Sparse UInt32 100%)"

# Permuted

permuted = PermutedDimsArray(sparse, (2, 1))
@test sparsify(permuted) === permuted
@test brief(permuted) == "3 x 2 x Int64 in Rows (Permute, Sparse Int64 67%)"

unpermuted = PermutedDimsArray(sparse, (1, 2))
@test sparsify(unpermuted) === unpermuted
@test brief(unpermuted) == "2 x 3 x Int64 in Columns (!Permute, Sparse Int64 67%)"

permuted = PermutedDimsArray(dense, (2, 1))
@test sparsify(permuted) == permuted
@test brief(permuted) == "4 x 3 x Float64 in Rows (Permute, Dense)"
@test brief(sparsify(permuted)) == "4 x 3 x Float64 in Rows (Permute, Sparse UInt32 100%)"

unpermuted = PermutedDimsArray(dense, (1, 2))
@test sparsify(unpermuted) == unpermuted
@test brief(unpermuted) == "3 x 4 x Float64 in Columns (!Permute, Dense)"
@test brief(sparsify(unpermuted)) == "3 x 4 x Float64 in Columns (!Permute, Sparse UInt32 100%)"

# LinearAlgebra

transposed = transpose(sparse)
@test sparsify(transposed) === transposed
@test brief(transposed) == "3 x 2 x Int64 in Rows (Transpose, Sparse Int64 67%)"

adjointed = adjoint(sparse)
@test sparsify(adjointed) === adjointed
@test brief(adjointed) == "3 x 2 x Int64 in Rows (Adjoint, Sparse Int64 67%)"

transposed = transpose(dense)
@test sparsify(transposed) == transposed
@test brief(transposed) == "4 x 3 x Float64 in Rows (Transpose, Dense)"
@test brief(sparsify(transposed)) == "4 x 3 x Float64 in Rows (Transpose, Sparse UInt32 100%)"

adjointed = adjoint(dense)
@test sparsify(adjointed) == adjointed
@test brief(adjointed) == "4 x 3 x Float64 in Rows (Adjoint, Dense)"
@test brief(sparsify(adjointed)) == "4 x 3 x Float64 in Rows (Adjoint, Sparse UInt32 100%)"

println("OK")

# output

OK
```

```jldoctest
using Test
using SparseArrays

# Dense

dense = rand(4)
@test sparsify(dense) == dense
@test brief(dense) == "4 x Float64 (Dense)"
@test brief(sparsify(dense)) == "4 x Float64 (Sparse UInt32 100%)"

# Sparse

sparse = SparseVector([0, 1, 2, 0])
@test sparsify(sparse) === sparse
@test brief(sparse) == "4 x Int64 (Sparse Int64 50%)"

@test sparsify(sparse; copy = true) == sparse
@test sparsify(sparse; copy = true) !== sparse
@test brief(sparsify(sparse)) == "4 x Int64 (Sparse Int64 50%)"

@test sparsify(sparse; eltype = Int8) == sparse
@test brief(sparsify(sparse; eltype = Int8)) == "4 x Int8 (Sparse Int64 50%)"

@test sparsify(sparse; indtype = Int8) == sparse
@test brief(sparsify(sparse; indtype = Int8)) == "4 x Int64 (Sparse Int8 50%)"

println("OK")

# output

OK
```
"""
function sparsify(
    matrix::AbstractMatrix;
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractMatrix
    is_sparse, eltype, indtype = parse_sparse_array(matrix; eltype, indtype)

    if !is_sparse || copy || eltype != Base.eltype(matrix) || indtype != SparseArrays.indtype(matrix)
        matrix = sparse_matrix_csc(matrix; eltype, indtype)
    end

    return matrix
end

function sparsify(
    vector::AbstractVector;
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractVector
    is_sparse, eltype, indtype = parse_sparse_array(vector; eltype, indtype)

    if !is_sparse || copy || eltype != Base.eltype(vector) || indtype != SparseArrays.indtype(vector)
        vector = sparse_vector(vector; eltype, indtype)
    end

    return vector
end

function sparsify(
    array::ReadOnlyArray;
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractArray
    parent_array = sparsify(parent(array); copy, eltype, indtype)
    if parent_array === parent(array)
        return array
    else
        return ReadOnlyArray(parent_array)
    end
end

function sparsify(
    array::PermutedDimsArray{T, 2, P, IP, A};
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractMatrix where {T, P, IP, A}
    parent_array = sparsify(parent(array); copy, eltype, indtype)
    if parent_array === parent(array)
        return array
    else
        return PermutedDimsArray(parent_array, P)
    end
end

function sparsify(
    array::Transpose;
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractArray
    parent_array = sparsify(parent(array); copy, eltype, indtype)
    if parent_array === parent(array)
        return array
    else
        return Transpose(parent_array)
    end
end

function sparsify(
    array::Adjoint;
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractArray
    parent_array = sparsify(parent(array); copy, eltype, indtype)
    if parent_array === parent(array)
        return array
    else
        return Adjoint(parent_array)
    end
end

function sparsify(
    array::NamedArray;
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::NamedArray
    parent_array = sparsify(parent(array); copy, eltype, indtype)
    if parent_array === parent(array)
        return array
    else
        return NamedArray(parent_array, array.dicts, array.dimnames)
    end
end

"""
    densify(matrix::AbstractMatrix; copy::Bool = false, eltype::Maybe{Type} = nothing)::AbstractMatrix
    densify(vector::AbstractVector; copy::Bool = false, eltype::Maybe{Type} = nothing)::AbstractVector

Return a dense version of an array, possibly forcing a different `eltype`. This will preserve the matrix layout (for
example, `densify` of a transposed matrix will be a transposed matrix). If `copy`, this will create a copy even if it is
already dense and has the correct `eltype`.

```jldoctest
using Test
using SparseArrays

# Dense

dense = rand(3, 4)
@test densify(dense) === dense
@test brief(dense) == "3 x 4 x Float64 in Columns (Dense)"

@test densify(dense; copy = true) !== dense
@test densify(dense; copy = true) == dense
@test brief(densify(dense; copy = true)) == "3 x 4 x Float64 in Columns (Dense)"

@test isapprox(densify(dense; eltype = Float32), dense)
@test brief(densify(dense; eltype = Float32)) == "3 x 4 x Float32 in Columns (Dense)"

# Sparse

sparse = SparseMatrixCSC([0 1 2; 3 4 0])

@test densify(sparse) == sparse
@test brief(densify(sparse)) == "2 x 3 x Int64 in Columns (Dense)"
@test brief(densify(sparse; eltype = Int8)) == "2 x 3 x Int8 in Columns (Dense)"

# ReadOnly

read_only = read_only_array(sparse)
@test densify(read_only) == read_only
@test brief(read_only) == "2 x 3 x Int64 in Columns (ReadOnly, Sparse Int64 67%)"
@test brief(densify(read_only)) == "2 x 3 x Int64 in Columns (ReadOnly, Dense)"

read_only = read_only_array(dense)
@test densify(read_only) == dense

# Named

using NamedArrays

named = NamedArray(sparse)
@test densify(named) == named
@test brief(named) == "2 x 3 x Int64 in Columns (Named, Sparse Int64 67%)"
@test brief(densify(named)) == "2 x 3 x Int64 in Columns (Named, Dense)"

named = NamedArray(dense)
@test densify(named) == dense

# Permuted

permuted = PermutedDimsArray(dense, (2, 1))
@test densify(permuted) === permuted
@test brief(permuted) == "4 x 3 x Float64 in Rows (Permute, Dense)"

unpermuted = PermutedDimsArray(dense, (1, 2))
@test densify(unpermuted) === unpermuted
@test brief(unpermuted) == "3 x 4 x Float64 in Columns (!Permute, Dense)"

permuted = PermutedDimsArray(sparse, (2, 1))
@test densify(permuted) == permuted
@test brief(permuted) == "3 x 2 x Int64 in Rows (Permute, Sparse Int64 67%)"
@test brief(densify(permuted)) == "3 x 2 x Int64 in Rows (Permute, Dense)"

unpermuted = PermutedDimsArray(sparse, (1, 2))
@test densify(unpermuted) == unpermuted
@test brief(unpermuted) == "2 x 3 x Int64 in Columns (!Permute, Sparse Int64 67%)"
@test brief(densify(unpermuted)) == "2 x 3 x Int64 in Columns (!Permute, Dense)"

# LinearAlgebra

transposed = transpose(dense)
@test densify(transposed) === transposed
@test brief(transposed) == "4 x 3 x Float64 in Rows (Transpose, Dense)"

adjointed = adjoint(dense)
@test densify(adjointed) === adjointed
@test brief(adjointed) == "4 x 3 x Float64 in Rows (Adjoint, Dense)"

transposed = transpose(sparse)
@test densify(transposed) == transposed
@test brief(transposed) == "3 x 2 x Int64 in Rows (Transpose, Sparse Int64 67%)"
@test brief(densify(transposed)) == "3 x 2 x Int64 in Rows (Transpose, Dense)"

adjointed = adjoint(sparse)
@test densify(adjointed) == adjointed
@test brief(adjointed) == "3 x 2 x Int64 in Rows (Adjoint, Sparse Int64 67%)"
@test brief(densify(adjointed)) == "3 x 2 x Int64 in Rows (Adjoint, Dense)"

println("OK")

# output

OK
```

```jldoctest
using Test
using SparseArrays

# Sparse

sparse = SparseVector([0, 1, 2, 0])

@test densify(sparse) == sparse
@test brief(densify(sparse)) == "4 x Int64 (Dense)"

# Dense

dense = rand(4)
@test densify(dense) === dense
@test brief(dense) == "4 x Float64 (Dense)"

@test densify(dense; copy = true) !== dense
@test densify(dense; copy = true) == dense
@test brief(densify(dense; copy = true)) == "4 x Float64 (Dense)"

@test isapprox(densify(dense; eltype = Float32), dense)
@test brief(densify(dense; eltype = Float32)) == "4 x Float32 (Dense)"

println("OK")

# output

OK
```
"""
function densify(matrix::AbstractMatrix; copy::Bool = false, eltype::Maybe{Type} = nothing)::AbstractMatrix
    is_sparse, eltype = parse_dense_array(matrix; eltype)

    if is_sparse || copy || major_axis(matrix) == Nothing || eltype != Base.eltype(matrix)
        matrix = Matrix{eltype}(matrix)
    end

    return matrix
end

function densify(vector::AbstractVector; copy::Bool = false, eltype::Maybe{Type} = nothing)::AbstractVector
    is_sparse, eltype = parse_dense_array(vector; eltype)

    if is_sparse || copy || eltype != Base.eltype(vector)
        vector = Vector{eltype}(vector)
    end

    return vector
end

function densify(array::ReadOnlyArray; copy::Bool = false, eltype::Maybe{Type} = nothing)::AbstractArray
    parent_array = densify(parent(array); copy, eltype)
    if parent_array === parent(array)
        return array
    else
        return ReadOnlyArray(parent_array)
    end
end

function densify(
    array::PermutedDimsArray{T, 2, P, IP, A};
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
)::AbstractMatrix where {T, P, IP, A}
    parent_array = densify(parent(array); copy, eltype)
    if parent_array === parent(array)
        return array
    else
        return PermutedDimsArray(parent_array, P)
    end
end

function densify(array::Transpose; copy::Bool = false, eltype::Maybe = nothing)::AbstractArray
    parent_array = densify(parent(array); copy, eltype)
    if parent_array === parent(array)
        return array
    else
        return Transpose(parent_array)
    end
end

function densify(array::Adjoint; copy::Bool = false, eltype::Maybe{Type} = nothing)::AbstractArray
    parent_array = densify(parent(array); copy, eltype)
    if parent_array === parent(array)
        return array
    else
        return Adjoint(parent_array)
    end
end

function densify(array::NamedArray; copy::Bool = false, eltype::Maybe{Type} = nothing)::AbstractArray
    parent_array = densify(parent(array); copy, eltype)
    if parent_array === parent(array)
        return array
    else
        return NamedArray(parent_array, array.dicts, array.dimnames)
    end
end

function parse_sparse_array(
    array::AbstractArray;
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::Tuple{Bool, Type, Type}
    is_sparse, eltype = parse_dense_array(array; eltype)

    if indtype === nothing
        if is_sparse
            indtype = SparseArrays.indtype(array)
        else
            indtype = indtype_for_size(length(array))
        end
    end

    return (is_sparse, eltype, indtype)
end

function parse_dense_array(array::AbstractArray; eltype::Maybe{Type} = nothing)::Tuple{Bool, Type}
    if eltype === nothing
        eltype = Base.eltype(array)
    end

    is_sparse = issparse(array)

    return (is_sparse, eltype)
end

"""
    sparse_matrix_csc(
        matrix::AbstractMatrix;
        eltype::Maybe{Type} = nothing,
        indtype::Maybe{Type} = nothing
    )::SparseMatrixCSC

    sparse_matrix_csc(
        colptr::AbstractVector,
        rowval::AbstractVector,
        nzval::AbstractVector
    )::Union{ReadOnlyArray, SparseMatrixCSC}

Create a sparse column-major matrix. This differs from the simple `SparseMatrixCSC` in the following ways:

  - The integer index type is `UInt32` if possible. Only very large matrix sizes use `UInt64`. This greatly reduces
    the size of large matrices.

  - If constructing the matrix from three vectors, then if any of them are `ReadOnlyArray`, this will return a
    `ReadOnlyArray` wrapper for the result (which will internally refer to the mutable arrays).

  - If `eltype` is specified, this will be the element type of the result.

```jldoctest
using Test

# Matrix

@test brief(sparse_matrix_csc([0 1 2; 3 4 0])) == "2 x 3 x Int64 in Columns (Sparse UInt32 67%)"
@test brief(sparse_matrix_csc([0 1 2; 3 4 0]; eltype = Float32)) == "2 x 3 x Float32 in Columns (Sparse UInt32 67%)"
@test brief(sparse_matrix_csc([0 1 2; 3 4 0]; indtype = UInt8)) == "2 x 3 x Int64 in Columns (Sparse UInt8 67%)"

# Vectors

sparse = sparse_matrix_csc([0 1 2; 3 4 0])

@test brief(sparse_matrix_csc(2, 3, sparse.colptr, sparse.rowval, sparse.nzval)) == "2 x 3 x Int64 in Columns (Sparse UInt32 67%)"
@test brief(sparse_matrix_csc(2, 3, read_only_array(sparse.colptr), read_only_array(sparse.rowval), read_only_array(sparse.nzval))) ==
      "2 x 3 x Int64 in Columns (ReadOnly, Sparse UInt32 67%)"

println("OK")

# output

OK
```
"""
function sparse_matrix_csc(
    matrix::AbstractMatrix;
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::SparseMatrixCSC
    _, eltype, indtype = parse_sparse_array(matrix; eltype, indtype)

    if eltype != Base.eltype(matrix)
        matrix = SparseMatrixCSC{Base.eltype(matrix), indtype}(matrix)
        return SparseMatrixCSC{eltype, indtype}(size(matrix)..., matrix.colptr, matrix.rowval, eltype.(matrix.nzval))
    else
        return SparseMatrixCSC{eltype, indtype}(matrix)
    end
end

function sparse_matrix_csc(
    n_columns::Integer,
    n_rows::Integer,
    colptr::AbstractVector,
    rowval::AbstractVector,
    nzval::AbstractVector,
)::Union{ReadOnlyArray, SparseMatrixCSC}
    if is_read_only_array(colptr) || is_read_only_array(rowval) || is_read_only_array(nzval)
        return ReadOnlyArray(
            SparseMatrixCSC(n_columns, n_rows, mutable_array(colptr), mutable_array(rowval), mutable_array(nzval)),
        )
    else
        return SparseMatrixCSC(n_columns, n_rows, colptr, rowval, nzval)
    end
end

"""
    sparse_vector(
        vector::AbstractMatrix;
        eltype::Maybe{Type} = nothing,
        indtype::Maybe{Type} = nothing,
    )::SparseVector

    sparse_vector(
        size::Integer,
        inzind::AbstractVector,
        nzval::AbstractVector
    )::Union{ReadOnlyArray, SparseVector}

Create a sparse vector. This differs from the simple `SparseVector` in the following ways:

  - The integer index type is `UInt32` if possible. Only very large matrix sizes use `UInt64`. This greatly reduces
    the size of large matrices.

  - If constructing the vector from two vectors, then if any of them are `ReadOnlyArray`, this will return a
    `ReadOnlyArray` wrapper for the result (which will internally refer to the mutable arrays).

  - If `eltype` is specified, this will be the element type of the result.

```jldoctest
using Test

# Vector

@test brief(sparse_vector([0, 1, 2])) == "3 x Int64 (Sparse UInt32 67%)"
@test brief(sparse_vector([0, 1, 2]; eltype = Float32)) == "3 x Float32 (Sparse UInt32 67%)"

# Vectors

@test brief(sparse_vector(3, [1, 3], [1.0, 2.0])) == "3 x Float64 (Sparse Int64 67%)"
@test brief(sparse_vector(3, read_only_array([1, 3]), read_only_array([1.0, 2.0]))) == "3 x Float64 (ReadOnly, Sparse Int64 67%)"

println("OK")

# output

OK
```
"""
function sparse_vector(
    vector::AbstractVector;
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::SparseVector
    _, eltype, indtype = parse_sparse_array(vector; eltype, indtype)

    if eltype != Base.eltype(vector)
        vector = SparseVector{Base.eltype(vector), indtype}(vector)
        return SparseVector{eltype, indtype}(length(vector), vector.nzind, eltype.(vector.nzval))  # NOJET
    else
        return SparseVector{eltype, indtype}(vector)
    end
end

function sparse_vector(size::Integer, nzind::AbstractVector, nzval::AbstractVector)::Union{ReadOnlyArray, SparseVector}
    if is_read_only_array(nzind) || is_read_only_array(nzval)
        return read_only_array(SparseVector(size, mutable_array(nzind), mutable_array(nzval)))  # NOJET
    else
        return SparseVector(size, nzind, nzval)
    end
end

"""
    sparse_mask_vector(
        size::Integer,
        inzind::AbstractVector
    )::Union{ReadOnlyArray, SparseVector{Bool}}

Create a sparse mask vector using only the indices of the `true` entries. Alas, this still needs to allocate
a vector of `Bool` for the data.

```jldoctest
using Test

@test brief(sparse_mask_vector(3, [1, 3])) == "3 x Bool (Sparse Int64 67%)"
@test brief(sparse_mask_vector(3, read_only_array([1, 3]))) == "3 x Bool (ReadOnly, Sparse Int64 67%)"

println("OK")

# output

OK
```
"""
function sparse_mask_vector(size::Integer, indptr::AbstractVector)::Union{ReadOnlyArray, SparseVector{Bool}}
    if is_read_only_array(indptr)
        return read_only_array(SparseVector(size, mutable_array(indptr), fill(true, length(indptr))))  # NOJET
    else
        return SparseVector(size, indptr, fill(true, length(indptr)))
    end
end

"""
    dense_mask_vector(
        size::Integer,
        inzind::AbstractVector
    )::Vector{Bool}

Create a dense mask vector using only the indices of the `true` entries.

```jldoctest
println(brief(dense_mask_vector(4, [1, 3])))

# output

4 x Bool (Dense; 50% true)
```
"""
function dense_mask_vector(size::Integer, indptr::AbstractVector)::Vector{Bool}
    vector = zeros(Bool, size)
    vector[indptr] .= true
    return vector
end

"""
    indtype_for_size(size::Integer)::Type

Return the an integer data type which is large enough to hold indices and offsets for a `SparseMatrixCSC` matrix of some
`size` (total number of elements). We try to use `UInt32` whenever possible because for large matrices (especially with
32-bit value types) this will drastically reduce the amount of space used.

```jldoctest
println(10000000 => indtype_for_size(10000000))
println(10000000000 => indtype_for_size(10000000000))

# output

10000000 => UInt32
10000000000 => UInt64
```
"""
function indtype_for_size(size::Integer)::Type
    if size <= typemax(UInt32)
        return UInt32
    else
        return UInt64
    end
end

"""
    bestify(
        matrix::AbstractMatrix;
        min_sparse_saving_fraction::AbstractFloat = $(DEFAULT.min_sparse_saving_fraction),
        copy::Bool = false,
        eltype::Maybe{Type} = nothing,
    )::AbstractMatrix

    bestify(
        matrix::AbstractVector;
        min_sparse_saving_fraction::AbstractFloat = $(DEFAULT.min_sparse_saving_fraction),
        copy::Bool = false,
        eltype::Maybe{Type} = nothing,
    )::AbstractVector

Return a "best" (dense or sparse) version of an array. The sparse format is chosen if it saves at least
`min_sparse_saving_fraction` of the storage of the dense format. If `copy`, this will create a copy even if it is
already in the best format.

If `eltype` is specified, computes the savings (and create the "best" version) using this element type. In addition, if
given a sparse matrix, we consider the [`indtype_for_size`](@ref) for it, and if that saves
`min_sparse_saving_fraction` relative to the current sparse representation, we'll create a new one using the
better (smaller) `indtype`.

```jldoctest
using Test
using LinearAlgebra

# Dense

dense = zeros(Int32, 5, 5)
view(dense, diagind(dense)) .= 1

@test bestify(dense) == dense
@test brief(bestify(dense)) == "5 x 5 x Int32 in Columns (Sparse UInt32 20%)"

@test bestify(dense; min_sparse_saving_fraction = 0.5) === dense

# Sparse

sparse = sparse_matrix_csc(dense)
@test bestify(sparse) === sparse
@test brief(sparse) == "5 x 5 x Int32 in Columns (Sparse UInt32 20%)"

# ReadOnly

read_only = read_only_array(dense)
@test bestify(read_only; min_sparse_saving_fraction = 0.5) === read_only
@test brief(read_only) == "5 x 5 x Int32 in Columns (ReadOnly, Dense)"

@test bestify(read_only) == read_only
@test brief(bestify(read_only)) == "5 x 5 x Int32 in Columns (ReadOnly, Sparse UInt32 20%)"

read_only = read_only_array(sparse)
@test bestify(read_only) === read_only
@test brief(read_only) == "5 x 5 x Int32 in Columns (ReadOnly, Sparse UInt32 20%)"

@test bestify(read_only; min_sparse_saving_fraction = 0.5) == read_only
@test brief(bestify(read_only; min_sparse_saving_fraction = 0.5)) == "5 x 5 x Int32 in Columns (ReadOnly, Dense)"

# Named

using NamedArrays

named = NamedArray(dense)
@test bestify(named; min_sparse_saving_fraction = 0.5) === named
@test brief(named) == "5 x 5 x Int32 in Columns (Named, Dense)"

@test bestify(named) == named
@test brief(bestify(named)) == "5 x 5 x Int32 in Columns (Named, Sparse UInt32 20%)"

named = NamedArray(sparse)
@test bestify(named) === named
@test brief(named) == "5 x 5 x Int32 in Columns (Named, Sparse UInt32 20%)"

@test bestify(named; min_sparse_saving_fraction = 0.5) == named
@test brief(bestify(named; min_sparse_saving_fraction = 0.5)) == "5 x 5 x Int32 in Columns (Named, Dense)"

# Permuted

permuted = PermutedDimsArray(dense, (2, 1))
@test bestify(permuted; min_sparse_saving_fraction = 0.5) === permuted
@test brief(permuted) == "5 x 5 x Int32 in Rows (Permute, Dense)"

@test bestify(permuted) == permuted
@test brief(bestify(permuted)) == "5 x 5 x Int32 in Rows (Permute, Sparse UInt32 20%)"

permuted = PermutedDimsArray(sparse, (1, 2))
@test bestify(permuted) === permuted
@test brief(permuted) == "5 x 5 x Int32 in Columns (!Permute, Sparse UInt32 20%)"

@test bestify(permuted; min_sparse_saving_fraction = 0.5) == permuted
@test brief(bestify(permuted; min_sparse_saving_fraction = 0.5)) == "5 x 5 x Int32 in Columns (!Permute, Dense)"

# LinearAlgebra

transposed = transpose(dense)
@test bestify(transposed; min_sparse_saving_fraction = 0.5) === transposed
@test brief(transposed) == "5 x 5 x Int32 in Rows (Transpose, Dense)"

@test bestify(transposed) == transposed
@test brief(bestify(transposed)) == "5 x 5 x Int32 in Rows (Transpose, Sparse UInt32 20%)"

adjointed = adjoint(sparse)
@test bestify(adjointed) === adjointed
@test brief(adjointed) == "5 x 5 x Int32 in Rows (Adjoint, Sparse UInt32 20%)"

@test bestify(adjointed; min_sparse_saving_fraction = 0.5) == adjointed
@test brief(bestify(adjointed; min_sparse_saving_fraction = 0.5)) == "5 x 5 x Int32 in Rows (Adjoint, Dense)"

println("OK")

# output

OK
```

```jldoctest
using Test
using LinearAlgebra

# Dense

dense = zeros(Int32, 3)
dense[1] = 1

@test bestify(dense) == dense
@test brief(bestify(dense)) == "3 x Int32 (Sparse UInt32 33%)"

@test bestify(dense; min_sparse_saving_fraction = 0.5) === dense

# Sparse

sparse = sparse_vector(dense)
@test bestify(sparse) === sparse
@test brief(sparse) == "3 x Int32 (Sparse UInt32 33%)"

println("OK")

# output

OK
```
"""
@documented function bestify(
    array::AbstractArray;
    min_sparse_saving_fraction::AbstractFloat = 0.25,
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractArray
    is_sparse, eltype, indtype = parse_sparse_array(array; eltype, indtype)

    dense_storage = dense_array_storage(array; eltype)
    sparse_storage = sparse_array_storage(array; eltype, indtype)

    if sparse_storage > dense_storage * (1.0 - min_sparse_saving_fraction)
        return densify(array; eltype, copy)
    end

    if !copy && is_sparse
        current_storage = sparse_array_storage(array; eltype, indtype = SparseArrays.indtype(array))  # NOJET
        copy = sparse_storage <= current_storage * (1.0 - min_sparse_saving_fraction)
    end

    return sparsify(array; eltype, indtype, copy)
end

function bestify(
    array::ReadOnlyArray;
    min_sparse_saving_fraction::AbstractFloat = 0.25,
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractArray
    parent_array = bestify(parent(array); min_sparse_saving_fraction, copy, eltype, indtype)
    if parent_array === parent(array)
        return array
    else
        return read_only_array(parent_array)
    end
end

function bestify(
    matrix::PermutedDimsArray{T, 2, P, IP, A};
    min_sparse_saving_fraction::AbstractFloat = 0.25,
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractMatrix where {T, P, IP, A}
    parent_matrix = bestify(parent(matrix); min_sparse_saving_fraction, copy, eltype, indtype)
    if parent_matrix === parent(matrix)
        return matrix
    else
        return PermutedDimsArray(parent_matrix, P)
    end
end

function bestify(
    array::Transpose;
    min_sparse_saving_fraction::AbstractFloat = 0.25,
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractArray
    parent_array = bestify(parent(array); min_sparse_saving_fraction, copy, eltype, indtype)
    if parent_array === parent(array)
        return array
    else
        return Transpose(parent_array)
    end
end

function bestify(
    array::Adjoint;
    min_sparse_saving_fraction::AbstractFloat = 0.25,
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractArray
    parent_array = bestify(parent(array); min_sparse_saving_fraction, copy, eltype, indtype)
    if parent_array === parent(array)
        return array
    else
        return Adjoint(parent_array)
    end
end

function bestify(
    array::NamedArray;
    min_sparse_saving_fraction::AbstractFloat = 0.25,
    copy::Bool = false,
    eltype::Maybe{Type} = nothing,
    indtype::Maybe{Type} = nothing,
)::AbstractArray
    parent_array = bestify(parent(array); min_sparse_saving_fraction, copy, eltype, indtype)
    if parent_array === parent(array)
        return array
    else
        return NamedArray(parent_array, array.dicts, array.dimnames)
    end
end

function sparse_array_storage(matrix::AbstractMatrix; eltype::Type, indtype::Type)::Int64
    if issparse(matrix)
        n_nz = nnz(matrix)
    else
        n_nz = sum(matrix .!= 0)
    end

    return n_nz * (sizeof(eltype) + sizeof(indtype)) + (size(matrix, Columns) + 1) * sizeof(indtype)
end

function sparse_array_storage(vector::AbstractVector; eltype::Type, indtype::Type)::Int64
    if issparse(vector)
        n_nz = nnz(vector)
    else
        n_nz = sum(vector .!= 0)
    end

    return n_nz * (sizeof(eltype) + sizeof(indtype))
end

function dense_array_storage(array::AbstractArray; eltype::Type)::Int64
    return length(array) * sizeof(eltype)
end

# Why do we have to define these ourselves... Sigh.

function SparseArrays.indtype(read_only::ReadOnlyArray)::Type  # UNTESTED
    return SparseArrays.indtype(parent(read_only))
end

function SparseArrays.nnz(read_only::ReadOnlyArray)::Integer  # UNTESTED
    return SparseArrays.nnz(parent(read_only))
end

# These we can excuse...

function SparseArrays.indtype(named::NamedArray)::Type  # UNTESTED
    return SparseArrays.indtype(named.array)
end

function SparseArrays.nnz(named::NamedArray)::Integer  # UNTESTED
    return SparseArrays.nnz(named.array)
end

"""
    colptr(sparse::AbstractMatrix)::AbstractVector{<:Integer}

Return the `colptr` of a `sparse` matrix.

```jldoctest
using Test
using NamedArrays
using SparseArrays

sparse_matrix = SparseMatrixCSC([0 1 2; 3 4 0])
@assert colptr(sparse_matrix) === sparse_matrix.colptr
@assert colptr(read_only_array(sparse_matrix)) === sparse_matrix.colptr
@assert colptr(NamedArray(sparse_matrix)) === sparse_matrix.colptr

println("OK")

# output

OK
```
"""
function colptr(read_only::SparseArrays.ReadOnly)::AbstractVector
    return colptr(parent(read_only))
end

function colptr(matrix::SparseMatrixCSC)::AbstractVector
    return matrix.colptr
end

function colptr(named::NamedArray)::AbstractVector
    @assert issparse(parent(named))
    return colptr(parent(named))
end

"""
    nzind(sparse::AbstractVector)::AbstractVector{<:Integer}

Return the `nzind` of a `sparse` vector.

```jldoctest
using Test
using NamedArrays
using SparseArrays

sparse_vector = SparseVector([0, 1, 2])
@assert nzind(sparse_vector) === sparse_vector.nzind
@assert nzind(read_only_array(sparse_vector)) === sparse_vector.nzind
@assert nzind(NamedArray(sparse_vector)) === sparse_vector.nzind

println("OK")

# output

OK
```
"""
function nzind(read_only::SparseArrays.ReadOnly)::AbstractVector
    return nzind(parent(read_only))
end

function nzind(vector::SparseVector)::AbstractVector
    return vector.nzind
end

function nzind(named::NamedArray)::AbstractVector
    @assert issparse(parent(named))
    return nzind(parent(named))
end

"""
    nzval(sparse::AbstractArray)::AbstractVector

Return the `nzval` of a `sparse` array.

```jldoctest
using Test
using NamedArrays
using SparseArrays

sparse_matrix = SparseMatrixCSC([0 1 2; 3 4 0])
@assert nzval(sparse_matrix) === sparse_matrix.nzval
@assert nzval(read_only_array(sparse_matrix)) === sparse_matrix.nzval
@assert nzval(NamedArray(sparse_matrix)) === sparse_matrix.nzval

sparse_vector = SparseVector([0, 1, 2])
@assert nzval(sparse_vector) === sparse_vector.nzval
@assert nzval(read_only_array(sparse_vector)) === sparse_vector.nzval
@assert nzval(NamedArray(sparse_vector)) === sparse_vector.nzval

println("OK")

# output

OK
```
"""
function nzval(vector::SparseVector)::AbstractVector
    return vector.nzval
end

function nzval(matrix::SparseMatrixCSC)::AbstractVector
    return matrix.nzval
end

function nzval(read_only::SparseArrays.ReadOnly)::AbstractVector
    return nzval(parent(read_only))
end

function nzval(named::NamedArray)::AbstractVector
    @assert issparse(parent(named))
    return nzval(parent(named))
end

"""
    rowval(sparse::AbstractArray)::AbstractVector{<Integer}

Return the `rowval` of a `sparse` array.

```jldoctest
using Test
using NamedArrays
using SparseArrays

sparse_matrix = SparseMatrixCSC([0 1 2; 3 4 0])
@assert rowval(sparse_matrix) === sparse_matrix.rowval
@assert rowval(read_only_array(sparse_matrix)) === sparse_matrix.rowval
@assert rowval(NamedArray(sparse_matrix)) === sparse_matrix.rowval

println("OK")

# output

OK
```
"""
function rowval(matrix::SparseMatrixCSC)::AbstractVector
    return matrix.rowval
end

function rowval(read_only::SparseArrays.ReadOnly)::AbstractVector
    return rowval(parent(read_only))
end

function rowval(named::NamedArray)::AbstractVector
    @assert issparse(parent(named))
    return rowval(parent(named))
end

end  # module
