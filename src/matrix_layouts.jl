"""
Matrix data that has a clear layout, that is, a [`major_axis`](@ref), regardless of whether it is dense or sparse.

That is, for [`Columns`](@ref)-major data, the values of each column are laid out consecutively in memory (each column
is a single contiguous vector), so any operation that works on whole columns will be fast (e.g., summing the value of
each column). In contrast, the values of each row are stored far apart from each other, so any operation that works on
whole rows will be very slow in comparison (e.g., summing the value of each row).

For [`Rows`](@ref)-major data, the values of each row are laid out consecutively in memory (each row is a single
contiguous vector). In contrast, the values of each column are stored far apart from each other. In this case, summing
columns would be slow, and summing rows would be fast.

This is much simpler than the [ArrayLayouts](https://github.com/JuliaLinearAlgebra/ArrayLayouts.jl) module which
attempts to fully describe the layout of N-dimensional arrays, a much more ambitious goal which is an overkill for our
needs.

!!! note

    The "default" layout in Julia is column-major, which inherits this from matlab, which inherits this from FORTRAN,
    allegedly because this is more efficient for some linear algebra operations. In contrast, most system languages and
    Python `numpy` use row-major layout by default, because that's the sane layout (and generalizes better for tensors).
    At any rate, all systems work just fine with data of either memory layout; the key consideration is to keep track of
    the layout, and to apply operations "with the grain" rather than "against the grain" of the data.
"""
module MatrixLayouts

export @assert_matrix
export @assert_vector
export @check_turbo_matrix
export @check_turbo_vector
export Columns
export Rows
export axis_name
export canonical_array
export is_contiguous_array
export is_costly_to_read
export major_axis
export minor_axis
export other_axis
export plain_array
export relayout
export relayout!
export require_major_axis
export require_minor_axis
export flip
export flipped
export GLOBAL_INEFFICIENT_ACTION_HANDLER
export RELAYOUT_BUDGET_KB
export RELAYOUT_MIN_WRITE_KB

using ..Documentation
using ..Types
using ..Brief
using ..FlameTime
using ..Handlers
using ..ReadOnlyArrays

using Base.Threads
using DiskArrays
using Distributed
using LinearAlgebra
using LoopVectorization
using NamedArrays
using SparseArrays

"""
A symbolic name for the rows axis. It is more readable to write, say, `size(matrix, Rows)`, instead of
`size(matrix, 1)`.
"""
const Rows = 1

"""
A symbolic name for the columns axis. It is more readable to write, say, `size(matrix, Columns)`, instead of
`size(matrix, 2)`.
"""
const Columns = 2

"""
    axis_name(axis::Maybe{Integer})::String

Return the name of the axis (for messages).

```jldoctest
println(axis_name(nothing))
println(axis_name(Rows))
println(axis_name(Columns))
println(axis_name(3))

# output

nothing
Rows
Columns
ERROR: invalid matrix axis: 3
```
"""
function axis_name(axis::Maybe{Integer})::String
    if axis === nothing
        return "nothing"
    elseif axis == Rows
        return "Rows"
    elseif axis == Columns
        return "Columns"
    else
        error("invalid matrix axis: $(axis)")
    end
end

"""
    major_axis(matrix::AbstractMatrix)::Maybe{Int8}

Return the index of the major axis of a matrix, that is, the axis one should keep **fixed** for an efficient inner loop
accessing the matrix elements. If the matrix doesn't support any efficient access axis, returns `nothing`.

```jldoctest
base = [0 1 2; 3 4 0]

@assert major_axis(base) == Columns

# Slice

@assert major_axis(@view base[:, [1, 3, 2]]) == Columns

# Named

using NamedArrays

@assert major_axis(NamedArray(base)) == Columns

# Permuted

permuted = PermutedDimsArray(base, (2, 1))
@assert major_axis(permuted) == Rows

@assert flip(permuted) === base

unpermuted = PermutedDimsArray(base, (1, 2))
@assert major_axis(unpermuted) == Columns

# LinearAlgebra

transposed = transpose(base)
@assert major_axis(transposed) == Rows

@assert flip(transposed) === base

adjointed = adjoint(base)
@assert major_axis(adjointed) == Rows

# Sparse

using SparseArrays

sparse = SparseMatrixCSC(base)
@assert major_axis(sparse) == Columns

# output

```
"""
function major_axis(matrix::Union{NamedMatrix, ReadOnlyArray})::Maybe{Int8}
    return major_axis(parent(matrix))
end

# DiskArrays wraps its arrays in its own `PermutedDiskArray` instead of a `PermutedDimsArray`.
function major_axis(
    matrix::Union{PermutedDimsArray{T, 2, P, IP, A}, DiskArrays.AbstractPermutedDiskArray{T, 2, P, IP, A}},
)::Maybe{Int8} where {T, P, IP, A}
    if P == (Rows, Columns)
        return major_axis(parent(matrix))
    elseif P == (Columns, Rows)
        return other_axis(major_axis(parent(matrix)))
    else
        @assert false "can't handle matrix type: $(typeof(matrix))"  # UNTESTED
    end
end

function major_axis(matrix::Union{Transpose, Adjoint})::Maybe{Int8}
    return other_axis(major_axis(matrix.parent))
end

# A flipped vector is a single row of adjacent elements.
function major_axis(::Union{Transpose{<:Any, <:AbstractVector}, Adjoint{<:Any, <:AbstractVector}})::Maybe{Int8}
    return Rows
end

function major_axis(::AbstractSparseMatrix)::Maybe{Int8}
    return Columns
end

function major_axis(::BitMatrix)::Maybe{Int8}
    return Columns
end

function major_axis(matrix::AbstractMatrix)::Maybe{Int8}
    try
        matrix_strides = strides(matrix)
        if matrix_strides[Rows] == 1
            return Columns
        elseif matrix_strides[Columns] == 1  # UNTESTED
            return Rows  # UNTESTED
        else
            return nothing  # UNTESTED
        end

    catch MethodError  # NOLINT
        return nothing
    end
end

function major_axis(matrix::SubArray)::Maybe{Int8}
    if ndims(matrix.parent) == 2
        return major_axis(matrix.parent)
    else
        # A view of an array of more dimensions has the layout its strides give it, if any.
        return invoke(major_axis, Tuple{AbstractMatrix}, matrix)
    end
end

function major_axis(matrix::DiskArrays.CachedDiskArray)::Maybe{Int8}  # UNTESTED
    return major_axis(matrix.parent)
end

"""
    require_major_axis(matrix::AbstractMatrix)::Int8

Similar to [`major_axis`](@ref) but will `error` if the matrix isn't in either row-major or column-major layout.

```jldoctest
base = [0 1 2; 3 4 0]

@assert require_major_axis(base) == Columns
@assert require_major_axis(@view base[:, [1, 3, 2]]) == Columns

# output

```
"""
function require_major_axis(matrix::AbstractMatrix)::Int8
    axis = major_axis(matrix)
    if axis === nothing
        error("type: $(typeof(matrix)) is not in any-major layout")  # UNTESTED
    end
    return axis
end

"""
    minor_axis(matrix::AbstractMatrix)::Maybe{Int8}

Return the index of the minor axis of a matrix, that is, the axis one should **vary** for an efficient inner loop
accessing the matrix elements. If the matrix doesn't support any efficient access axis, returns `nothing`.

```jldoctest
base = [0 1 2; 3 4 0]

@assert minor_axis(base) == Rows

# Slice

@assert minor_axis(@view base[:, [1, 3, 2]]) == Rows

# Named

using NamedArrays

@assert minor_axis(NamedArray(base)) == Rows

# Permuted

permuted = PermutedDimsArray(base, (2, 1))
@assert minor_axis(permuted) == Columns

@assert flip(permuted) === base

unpermuted = PermutedDimsArray(base, (1, 2))
@assert minor_axis(unpermuted) == Rows

# LinearAlgebra

transposed = transpose(base)
@assert minor_axis(transposed) == Columns

@assert flip(transposed) === base

adjointed = adjoint(base)
@assert minor_axis(adjointed) == Columns

@assert flip(adjointed) === base

# Sparse

using SparseArrays

sparse = SparseMatrixCSC(base)
@assert minor_axis(sparse) == Rows

# output

```
"""
function minor_axis(matrix::AbstractMatrix)::Maybe{Int8}
    return other_axis(major_axis(matrix))
end

"""
    require_minor_axis(matrix::AbstractMatrix)::Int8

Similar to [`minor_axis`](@ref) but will `error` if the matrix isn't in either row-major or column-major layout.

```jldoctest
base = [0 1 2; 3 4 0]

@assert require_minor_axis(base) == Rows

@assert require_minor_axis(@view base[:, [1, 3, 2]]) == Rows

# output

```
"""
function require_minor_axis(matrix::AbstractMatrix)::Int8
    return other_axis(require_major_axis(matrix))
end

"""
    other_axis(axis::Maybe{Integer})::Maybe{Int8}

Return the other `matrix` `axis` (that is, convert between [`Rows`](@ref) and [`Columns`](@ref)). If given `nothing`
returns `nothing`.

```jldoctest
@assert other_axis(nothing) === nothing
@assert other_axis(Rows) == Columns
@assert other_axis(Columns) == Rows

other_axis(3)

# output

ERROR: invalid matrix axis: 3
```
"""
function other_axis(::Nothing)::Nothing
    return nothing
end

function other_axis(axis::Integer)::Int8
    if axis == Rows || axis == Columns
        return Int8(3 - axis)
    end

    return error("invalid matrix axis: $(axis)")
end

# Assert that `value` is an `AbstractVector`, with a friendly error message (naming the value and briefly describing
# it) if not.
function assert_is_vector_check(
    source_file::AbstractString,
    source_line::Integer,
    value_name::AbstractString,
    value::Any,
)::Nothing
    if !(value isa AbstractVector)
        error("non-vector " * value_name * ": " * brief(value) * "\nin: " * source_file * ":" * string(source_line))
    end
    return nothing
end

# Assert that `vector` has `n_elements`, with a friendly error message if not.
function assert_vector_size_check(
    source_file::AbstractString,
    source_line::Integer,
    vector_name::AbstractString,
    vector::AbstractVector,
    n_elements_name::AbstractString,
    n_elements::Integer,
)::Nothing
    if length(vector) != n_elements
        error(
            "wrong size: " *
            string(length(vector)) *
            "\nof the vector: " *
            vector_name *
            "\nis different from " *
            n_elements_name *
            ": " *
            string(n_elements) *
            "\nin: " *
            source_file *
            ":" *
            string(source_line),
        )
    end
    return nothing
end

"""
    @assert_vector(vector::Any, [n_elements::Integer])

Assert that the `vector` is an `AbstractVector` and optionally that it has `n_elements`, with a friendly error message
if it fails.

```jldoctest
vector = [0, 1, 2]

@assert_vector(vector)
n_elements = 3
@assert_vector(vector, n_elements)
m_elements = 2
@assert_vector(vector, m_elements)

# output

ERROR: wrong size: 3
of the vector: vector
is different from m_elements: 2
```

```jldoctest; filter = r"@.*"
scalar = 1
@assert_vector(scalar)

# output

ERROR: non-vector scalar: 1
```
"""
macro assert_vector(vector)
    return esc(
        :($(@__MODULE__).assert_is_vector_check(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(vector)),
            $vector,
        )),
    )
end

macro assert_vector(vector, n_elements)
    return esc(
        :($(@__MODULE__).assert_is_vector_check(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(vector)),
            $vector,
        );
        $(@__MODULE__).assert_vector_size_check(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(vector)),
            $vector,
            $(string(n_elements)),
            $n_elements,
        )),
    )
end

# Assert that `value` is an `AbstractMatrix`, with a friendly error message if not.
function assert_is_matrix_check(
    source_file::AbstractString,
    source_line::Integer,
    value_name::AbstractString,
    value::Any,
)::Nothing
    if !(value isa AbstractMatrix)
        error("non-matrix " * value_name * ": " * brief(value) * "\nin: " * source_file * ":" * string(source_line))
    end
    return nothing
end

# Assert that `matrix` has `(n_rows, n_columns)`, with a friendly error message if not.
function assert_matrix_size_check(
    source_file::AbstractString,
    source_line::Integer,
    matrix_name::AbstractString,
    matrix::AbstractMatrix,
    n_rows_name::AbstractString,
    n_rows::Integer,
    n_columns_name::AbstractString,
    n_columns::Integer,
)::Nothing
    if size(matrix) != (n_rows, n_columns)
        error(
            "wrong size: " *
            string(size(matrix)) *
            "\nof the matrix: " *
            matrix_name *
            "\nis different from (" *
            n_rows_name *
            ", " *
            n_columns_name *
            "): (" *
            string(n_rows) *
            ", " *
            string(n_columns) *
            ")\nin: " *
            source_file *
            ":" *
            string(source_line),
        )
    end
    return nothing
end

"""
The [`AbnormalHandler`](@ref) to use when accessing a matrix in an inefficient way ("against the grain"). Returns the
previous handler. The default handler is `WarnHandler`.
"""
GLOBAL_INEFFICIENT_ACTION_HANDLER::AbnormalHandler = WarnHandler

"""
    @assert_matrix(matrix::Any, [n_rows::Integer, n_columns::Integer], [major_axis::Int8])

Assert that the `matrix` is an `AbstractMatrix` and optionally that it has `n_rows` and `n_columns`. If the `major_axis`
is given, and does not match the matrix, invokes the [`GLOBAL_INEFFICIENT_ACTION_HANDLER`](@ref).

```jldoctest
matrix = [0 1 2; 3 4 0]

@assert_matrix(matrix)
n_rows, n_columns = (2, 3)
@assert_matrix(matrix, Columns)
@assert_matrix(matrix, n_rows, n_columns)
@assert_matrix(matrix, n_rows, n_columns, Columns)

m_rows, m_columns = (3, 2)
@assert_matrix(matrix, m_rows, m_columns)

# output

ERROR: wrong size: (2, 3)
of the matrix: matrix
is different from (m_rows, m_columns): (3, 2)
```

```jldoctest
matrix = [0 1 2; 3 4 0]

TanayLabUtilities.MatrixLayouts.GLOBAL_INEFFICIENT_ACTION_HANDLER = IgnoreHandler

@assert_matrix(matrix, Rows)

TanayLabUtilities.MatrixLayouts.GLOBAL_INEFFICIENT_ACTION_HANDLER = ErrorHandler

@assert_matrix(matrix, Rows)

# output

ERROR: inefficient major axis: Columns
for matrix: 2 x 3 x Int64 in Columns (Dense)
```
"""
macro assert_matrix(matrix)
    return esc(
        :($(@__MODULE__).assert_is_matrix_check(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(matrix)),
            $matrix,
        )),
    )
end

macro assert_matrix(matrix, axis)
    return esc(
        :($(@__MODULE__).assert_is_matrix_check(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(matrix)),
            $matrix,
        );
        $(@__MODULE__).check_efficient_action(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(matrix)),
            $matrix,
            $axis,
        )),
    )
end

macro assert_matrix(matrix, n_rows, n_columns)
    return esc(
        :($(@__MODULE__).assert_is_matrix_check(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(matrix)),
            $matrix,
        );
        $(@__MODULE__).assert_matrix_size_check(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(matrix)),
            $matrix,
            $(string(n_rows)),
            $n_rows,
            $(string(n_columns)),
            $n_columns,
        )),
    )
end

macro assert_matrix(matrix, n_rows, n_columns, axis)
    return esc(
        :($(@__MODULE__).assert_is_matrix_check(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(matrix)),
            $matrix,
        );
        $(@__MODULE__).assert_matrix_size_check(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(matrix)),
            $matrix,
            $(string(n_rows)),
            $n_rows,
            $(string(n_columns)),
            $n_columns,
        );
        $(@__MODULE__).check_efficient_action(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(matrix)),
            $matrix,
            $axis,
        )),
    )
end

# Assert that `value` is an `AbstractVector` and can be used in a `LoopVectorization` `@turbo` loop, with a friendly
# error message (naming the value and briefly describing it) if either check fails.
function check_turbo_vector_compatible(
    source_file::AbstractString,
    source_line::Integer,
    value_name::AbstractString,
    value::Any,
)::Nothing
    if !(value isa AbstractVector)
        error("non-vector " * value_name * ": " * brief(value) * "\nin: " * source_file * ":" * string(source_line))
    end
    if !LoopVectorization.check_args(value)
        error(
            "not @turbo compatible: " *
            value_name *
            ": " *
            brief(value) *
            "\nin: " *
            source_file *
            ":" *
            string(source_line),
        )
    end
    return nothing
end

# As above for `AbstractMatrix`.
function check_turbo_matrix_compatible(
    source_file::AbstractString,
    source_line::Integer,
    value_name::AbstractString,
    value::Any,
)::Nothing
    if !(value isa AbstractMatrix)
        error("non-matrix " * value_name * ": " * brief(value) * "\nin: " * source_file * ":" * string(source_line))
    end
    if !LoopVectorization.check_args(value)
        error(
            "not @turbo compatible: " *
            value_name *
            ": " *
            brief(value) *
            "\nin: " *
            source_file *
            ":" *
            string(source_line),
        )
    end
    return nothing
end

"""
    @check_turbo_vector(vector::Any)

Assert that `vector` is an `AbstractVector` that can be used in a `LoopVectorization` `@turbo` loop (a strided array of
a supported element type), with a friendly error message if not. Use this just before a `@turbo` loop instead of a
verbose manual `LoopVectorization.check_args` assertion.

```jldoctest
vector = [1.0, 2.0, 3.0]
@check_turbo_vector(vector)

# output

```
"""
macro check_turbo_vector(vector)
    return esc(
        :($(@__MODULE__).check_turbo_vector_compatible(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(vector)),
            $vector,
        )),
    )
end

"""
    @check_turbo_matrix(matrix::Any)

Assert that `matrix` is an `AbstractMatrix` that can be used in a `LoopVectorization` `@turbo` loop, with a friendly
error message if not. Use this just before a `@turbo` loop instead of a verbose manual `LoopVectorization.check_args`
assertion.

```jldoctest
matrix = [1.0 2.0; 3.0 4.0]
@check_turbo_matrix(matrix)

# output

```
"""
macro check_turbo_matrix(matrix)
    return esc(
        :($(@__MODULE__).check_turbo_matrix_compatible(
            $(string(__source__.file)),
            $(__source__.line),
            $(string(matrix)),
            $matrix,
        )),
    )
end

function check_efficient_action(
    source_file::AbstractString,
    source_line::Integer,
    operand::AbstractString,
    matrix::AbstractMatrix,
    axis::Integer,
)::Nothing
    if major_axis(matrix) != axis
        global GLOBAL_INEFFICIENT_ACTION_HANDLER
        handle_abnormal(GLOBAL_INEFFICIENT_ACTION_HANDLER) do
            return """
                inefficient major axis: $(axis_name(major_axis(matrix)))
                for $(operand): $(brief(matrix))
                in: $(source_file):$(source_line)
                """
        end
    end
end

"""
The amount of scratch memory, in KB, [`relayout!`](@ref) uses for a large matrix. The data of a larger matrix is
relayouted in two sequential passes, which is much faster when the destination is memory-mapped to a file. The default
is 256 MB.
"""
RELAYOUT_BUDGET_KB::Int = 262144

"""
The minimal size, in KB, of each write [`relayout!`](@ref) does to the destination of a large matrix. Smaller writes are
much slower when the destination is memory-mapped to a file. If [`RELAYOUT_BUDGET_KB`](@ref) is too small to allow this,
then more memory is used. The default is 64 KB.
"""
RELAYOUT_MIN_WRITE_KB::Int = 64

"""
    relayout!(destination::AbstractMatrix, source::AbstractMatrix)::AbstractMatrix
    relayout!(destination::AbstractMatrix, source::NamedMatrix)::NamedMatrix

Return the same `matrix` data, but in the other memory layout.

Suppose you have a column-major UMIs matrix, whose rows are cells, and columns are genes. Therefore, looping on the UMIs
of a gene will be fast, but looping on the UMIs of a cell will be slow. A `flip` (or `transpose`, no `!`, or
`PermutedDimsArray`) of a matrix is fast; they create a zero-copy wrapper of the matrix with flipped axes, so its rows
will be genes and columns will be cells, but in row-major layout. Therefore, **still**, looping on the UMIs of a gene is
fast, and looping on the UMIs of a cell is slow.

In contrast, `transpose!` (with a `!`) (or [`flipped`](@ref)) is slow; it creates a rearranged copy of the data, also
returning a matrix whose rows are genes and columns are cells, but this time, in column-major layout. Therefore, in this
case looping on the UMIs of a gene will be slow, and looping on the UMIs of a cell will be fast.

The `relayout!` is essentially a zero-copy `flip` of the slow `transpose!`. You end up with a matrix that
**appears** to be the same as the original (rows are cells and columns are genes), but behaves **differently** - looping
on the UMIs of a gene will be slow, and looping on the UMIs of a cell will be fast. In addition, `relayout!` will work
for both sparse and dense matrices. If the `source` is a `NamedMatrix`, then the result will be a `NamedMatrix` with the
same axes (zero-copy shared from the `source`). If `destination` is already a `NamedMatrix`, then its axes must match
`source`.

The caller is responsible for providing a sensible `destination` matrix (sparse for a sparse `source`, dense for a
non-sparse `source`, with compatible storage sizes).

If the data is larger than [`RELAYOUT_BUDGET_KB`](@ref), it is relayouted in bands of the `destination` columns. The
first pass reads the `source` in order, and appends each value to its band. The second pass rearranges each band in
memory. Both passes write the `destination` sequentially, in chunks of at least [`RELAYOUT_MIN_WRITE_KB`](@ref). This is
much faster than `transpose!` when the `destination` is memory-mapped to a file, especially on a network file system.
The passes use up to `nthreads()` tasks, as many as the budget allows.

!!! note

    It is almost always worthwhile to `relayout!` a matrix and then looping "with the grain" of the data, instead of
    skipping it and looping "against the grain" of the data. This is because (in Julia at least) the implementation of
    `transpose!` is optimized for the task, while the other operations typically don't provide any specific
    optimizations for working "against the grain" of the data. The benefits of a `relayout!` become more significant the
    more operations are done on the data in the loop.

```jldoctest
using LinearAlgebra

source = rand(3, 4)
destination = flip(rand(4, 3))

result = relayout!(destination, source)
@assert result === destination
@assert brief(source) == "3 x 4 x Float64 in Columns (Dense)"
@assert brief(result) == "3 x 4 x Float64 in Rows (Transpose, Dense)"
@assert result == source

# Named

using NamedArrays

named_source = NamedArray(rand(3, 4))
destination = flip(rand(4, 3))
result = relayout!(destination, named_source)
@assert parent(result) === destination
@assert brief(named_source) == "3 x 4 x Float64 in Columns (Named, Dense)"
@assert brief(result) == "3 x 4 x Float64 in Rows (Named, Transpose, Dense)"
@assert result == named_source

source = rand(3, 4)
named_destination = NamedArray(flip(rand(4, 3)))
result = relayout!(named_destination, source)
@assert result === named_destination
@assert brief(source) == "3 x 4 x Float64 in Columns (Dense)"
@assert brief(result) == "3 x 4 x Float64 in Rows (Named, Transpose, Dense)"
@assert result == source

source = rand(3, 4)
named_destination = Transpose(NamedArray(rand(4, 3)))
result = relayout!(named_destination, source)
@assert result === named_destination
@assert brief(source) == "3 x 4 x Float64 in Columns (Dense)"
@assert brief(result) == "3 x 4 x Float64 in Rows (Transpose, Named, Dense)"
@assert result == source

named_source = NamedArray(rand(3, 4))
named_destination = NamedArray(flip(rand(4, 3)))
result = relayout!(named_destination, named_source)
@assert result === named_destination
@assert brief(named_source) == "3 x 4 x Float64 in Columns (Named, Dense)"
@assert brief(result) == "3 x 4 x Float64 in Rows (Named, Transpose, Dense)"
@assert result == named_source

# Permuted

source = rand(3, 4)
destination = PermutedDimsArray(rand(4, 3), (2, 1))
result = relayout!(destination, source)
@assert result === destination
@assert brief(source) == "3 x 4 x Float64 in Columns (Dense)"
@assert brief(result) == "3 x 4 x Float64 in Rows (Permute, Dense)"
@assert result == source

source = rand(3, 4)
destination = PermutedDimsArray(adjoint(rand(4, 3)), (1, 2))
result = relayout!(destination, source)
@assert result === destination
@assert brief(source) == "3 x 4 x Float64 in Columns (Dense)"
@assert brief(result) == "3 x 4 x Float64 in Rows (!Permute, Adjoint, Dense)"
@assert result == source

# Sparse

using SparseArrays

source = SparseMatrixCSC([0.0 1.0 2.0; 3.0 4.0 0.0])
destination = flip(SparseMatrixCSC([30.0 0.0; 0.0 40.0; 20.0 10.0]))
result = relayout!(destination, source)
@assert result === destination
@assert brief(source) == "2 x 3 x Float64 in Columns (Sparse 4 (67%) [Int64])"
@assert brief(result) == "2 x 3 x Float64 in Rows (Transpose, Sparse 4 (67%) [Int64])"
@assert result == source

# output

```
"""
function relayout!(destination::AbstractMatrix, source::AbstractMatrix)::AbstractMatrix
    @assert size(destination) == size(source)
    @assert major_axis(destination) == minor_axis(source)
    @assert issparse(destination) == issparse(source)
    return flame_timed("relayout") do
        return named_relayout(destination, source)
    end
end

function named_relayout(destination::AbstractMatrix, source::NamedMatrix)::NamedArray
    @assert named_relayout(destination, parent(source)) === destination
    result = NamedArray(destination, source.dicts, source.dimnames)
    return result
end

function named_relayout(destination::NamedArray, source::NamedMatrix)::NamedArray
    @assert destination.dimnames == source.dimnames
    @assert destination.dicts == source.dicts
    @assert named_relayout(parent(destination), parent(source)) === parent(destination)
    return destination
end

function named_relayout(destination::NamedArray, source::AbstractMatrix)::AbstractMatrix
    @assert named_relayout(parent(destination), source) === parent(destination)
    return destination
end

function named_relayout(destination::AbstractMatrix, source::AbstractMatrix)::AbstractMatrix
    @assert unnamed_relayout(destination, source) === destination
    return destination
end

function unnamed_relayout(
    destination::PermutedDimsArray{T, 2, P, IP, A},
    source::AbstractMatrix,
)::AbstractMatrix where {T, P, IP, A}
    if P == (Rows, Columns)
        @assert named_relayout(parent(destination), source) === parent(destination)
    elseif P == (Columns, Rows)
        @assert named_relayout(parent(destination), flip(source)) === parent(destination)
    else
        @assert false
    end
    return destination
end

function unnamed_relayout(destination::Transpose, source::AbstractMatrix)::AbstractMatrix
    @assert named_relayout(parent(destination), flip(source)) === parent(destination)
    return destination
end

function unnamed_relayout(destination::Adjoint, source::AbstractMatrix)::AbstractMatrix
    @assert named_relayout(parent(destination), adjoint(source)) === parent(destination)
    return destination
end

# The methods for plain storage are defined with the parallel loops they use.
function unnamed_relayout(destination::AbstractMatrix, source::AbstractMatrix)::AbstractMatrix  # UNTESTED
    return error("""
               unsupported relayout destination: $(typeof(destination))
               and source: $(typeof(source))
               """)
end

"""
    relayout(matrix::AbstractMatrix)::AbstractMatrix
    relayout(matrix::NamedMatrix)::NamedMatrix

Same as [`relayout!`](@ref) but allocates the destination matrix for you. Is equivalent to
`flip(flipped(matrix))`.

```jldoctest
base = rand(3, 4)
@assert relayout(base) == base;
@assert major_axis(relayout(base)) == minor_axis(base);

# output

```
"""
function relayout(matrix::AbstractMatrix)::AbstractMatrix
    return flip(flipped(matrix))
end

"""
    flipped(matrix::AbstractMatrix)::AbstractMatrix

Return a transpose of a matrix, but instead of simply using a zero-copy wrapper, it actually rearranges the data. See
[`relayout!`](@ref).

```jldoctest
# Dense

base = rand(3, 4)
@assert flipped(base) == flip(base)
@assert brief(base) == "3 x 4 x Float64 in Columns (Dense)"
@assert brief(flip(base)) == "4 x 3 x Float64 in Rows (Transpose, Dense)"
@assert brief(flipped(base)) == "4 x 3 x Float64 in Columns (Dense)"

# Named

using NamedArrays

base = NamedArray(rand(3, 4))
@assert flipped(base) == flip(base)
@assert brief(base) == "3 x 4 x Float64 in Columns (Named, Dense)"
@assert brief(flip(base)) == "4 x 3 x Float64 in Rows (Named, Transpose, Dense)"
@assert brief(flipped(base)) == "4 x 3 x Float64 in Columns (Named, Dense)"

# Permuted

base = PermutedDimsArray(rand(3, 4), (2, 1))
@assert flipped(base) == flip(base)
@assert brief(base) == "4 x 3 x Float64 in Rows (Permute, Dense)"
@assert brief(flip(base)) == "3 x 4 x Float64 in Columns (Dense)"
@assert brief(flipped(base)) == "3 x 4 x Float64 in Rows (Transpose, Dense)"

base = PermutedDimsArray(rand(3, 4), (1, 2))
@assert flipped(base) == flip(base)
@assert brief(base) == "3 x 4 x Float64 in Columns (!Permute, Dense)"
@assert brief(flip(base)) == "4 x 3 x Float64 in Rows (Transpose, Dense)"
@assert brief(flipped(base)) == "4 x 3 x Float64 in Columns (Dense)"

# LinearAlgebra

using LinearAlgebra

base = transpose(rand(3, 4))
@assert flipped(base) == transpose(base)
@assert brief(base) == "4 x 3 x Float64 in Rows (Transpose, Dense)"
@assert brief(transpose(base)) == "3 x 4 x Float64 in Columns (Dense)"
@assert brief(flipped(base)) == "3 x 4 x Float64 in Rows (Transpose, Dense)"

base = adjoint(rand(3, 4))
@assert flipped(base) == transpose(base)
@assert brief(base) == "4 x 3 x Float64 in Rows (Adjoint, Dense)"
@assert brief(flip(base)) == "3 x 4 x Float64 in Columns (Dense)"
@assert brief(flipped(base)) == "3 x 4 x Float64 in Rows (Transpose, Dense)"

# ReadOnly

base = read_only_array(rand(3, 4))
@assert flipped(base) == transpose(base)
@assert brief(base) == "3 x 4 x Float64 in Columns (ReadOnly, Dense)"
@assert brief(flip(base)) == "4 x 3 x Float64 in Rows (ReadOnly, Transpose, Dense)"
@assert brief(flipped(base)) == "4 x 3 x Float64 in Columns (Dense)"

# Sparse

using SparseArrays

base = SparseMatrixCSC([0.0 1.0 2.0; 3.0 4.0 0.0])
@assert flipped(base) == flip(base)
@assert brief(base) == "2 x 3 x Float64 in Columns (Sparse 4 (67%) [Int64])"
@assert brief(flip(base)) == "3 x 2 x Float64 in Rows (Transpose, Sparse 4 (67%) [Int64])"
@assert brief(flipped(base)) == "3 x 2 x Float64 in Columns (Sparse 4 (67%) [Int64])"

# output

```
"""
function flipped(matrix::NamedMatrix)::NamedArray
    return NamedArray(flipped(parent(matrix)), flip_tuple(matrix.dicts), flip_tuple(matrix.dimnames))
end

function flip_tuple(tuple::Tuple{T1, T2})::Tuple{T2, T1} where {T1, T2}
    value1, value2 = tuple
    return (value2, value1)
end

# The method for any other matrix is defined with the parallel loops it uses.

function flipped(matrix::ReadOnlyArray)::AbstractMatrix
    return flipped(parent(matrix))
end

function flipped(matrix::PermutedDimsArray{T, 2, P, IP, A})::AbstractMatrix where {T, P, IP, A}
    if P == (Rows, Columns)
        return flipped(parent(matrix))
    elseif P == (Columns, Rows)
        return flip(flipped(parent(matrix)))
    else
        @assert false
    end
end

function flipped(matrix::Transpose{T})::AbstractMatrix{T} where {T}
    if T <: Complex
        return Transpose(flipped(parent(matrix)))  # UNTESTED
    else
        return flip(flipped(parent(matrix)))
    end
end

function flipped(matrix::Adjoint{T})::AbstractMatrix{T} where {T}
    if T <: Complex
        return Adjoint(flipped(parent(matrix)))  # UNTESTED
    else
        return flip(flipped(parent(matrix)))
    end
end

"""
    flip(AbstractMatrix)::AbstractMatrix

Flip the axes of a matrix. This applies `PermutedDimsArray` to the matrix. However, unlike the standard Julia functions,
which in their infinite wisdom blindly add a wrapper on top of the matrix, this function looks at the input first. As
long as it is not a matrix of `Complex` values, it will strip away an existing `Transpose` and/or `Adjoint` and/or
`PermutedDimsArray` wrapper, instead of attaching an additional one. In addition, `flip` will also look inside a
`ReadOnlyArray` and/or `NamedArray` to cancel out an internal flip wrapper. A view of a flipped matrix flips to the view
of the matrix.

!!! note

    Something along these lines really should have been in `Base`. Since there isn't, everyone is expected to create
    their own version of this - this is ours. We try to use it "universally" in our code instead of `PermutedDimsArray`,
    `transpose`, `adjoint`, `m'`, etc.
"""
function flip(matrix::Transpose{T}) where {T}
    return parent(matrix)
end

function flip(matrix::Adjoint{T}) where {T}
    if T <: Complex
        return Transpose(matrix)  # UNTESTED
    else
        return parent(matrix)
    end
end

function flip(
    matrix::Union{PermutedDimsArray{T, 2, P, IP, A}, DiskArrays.AbstractPermutedDiskArray{T, 2, P, IP, A}},
)::AbstractMatrix where {T, P, IP, A}
    if P == (Columns, Rows)
        return parent(matrix)
    elseif P == (Rows, Columns)
        return flip(parent(matrix))
    else
        @assert false
    end
end

function flip(matrix::AbstractMatrix)::AbstractMatrix
    return flipped_array(matrix, nothing)
end

function flip(matrix::SubArray{<:Any, 2})::AbstractMatrix
    indices = parentindices(matrix)
    if length(indices) == 2 && is_flipped_matrix(parent(matrix))
        # A view of a flip flips to the view of its parent, with the indices swapped.
        return view(parent(parent(matrix)), reverse(indices)...)
    else
        return flipped_array(matrix, nothing)
    end
end

function flip(matrix::ReadOnlyArray)::AbstractMatrix
    return ReadOnlyArray(flip(parent(matrix)))
end

function flip(matrix::NamedArray)::AbstractMatrix
    @assert ndims(matrix) == 2
    return NamedArray(flip(matrix.array), reverse(matrix.dicts), reverse(matrix.dimnames))
end

"""
    canonical_array(array::AbstractArray; flip_as::Maybe{Type} = nothing)::AbstractArray

Return the `array` in a canonical form, which has the same values at the same indices. It never copies and never reads
the data. See [`MatrixFormats`](@ref TanayLabUtilities.MatrixFormats) for when to use which function.

  - Read-only and named wrappers are removed. A DiskArrays view becomes the `SubArray` it wraps.
  - A flip is a `Transpose`, an `Adjoint`, or a `PermutedDimsArray` that swaps the dimensions. There is at most one
    flip, and it is the outermost wrapper. A view of a flip becomes a flip of a view.
    Two flips cancel out. A `PermutedDimsArray` that keeps the dimensions is removed.
  - By default, the flip is the one [`flip`](@ref) uses: a `Transpose` for numbers, and a `PermutedDimsArray`
    otherwise. Set `flip_as` to `Transpose`, `PermutedDimsArray` or `Adjoint` (for real numbers) to get a specific one.

```jldoctest
using LinearAlgebra

matrix = [1 2 3; 4 5 6]
@assert canonical_array(read_only_array(matrix)) === matrix
@assert canonical_array(PermutedDimsArray(transpose(matrix), (2, 1))) === matrix

canonical = canonical_array(view(transpose(matrix), 2:3, 1:2))
@assert canonical isa Transpose
@assert parent(parent(canonical)) === matrix
@assert canonical == view(transpose(matrix), 2:3, 1:2)

@assert canonical_array(transpose(matrix); flip_as = PermutedDimsArray) isa PermutedDimsArray

# output

```
"""
function canonical_array(array::AbstractArray; flip_as::Maybe{Type} = nothing)::AbstractArray  # NOLINT
    return array
end

function canonical_array(array::SparseArrays.ReadOnly; flip_as::Maybe{Type} = nothing)::AbstractArray
    return canonical_array(parent(array); flip_as)
end

function canonical_array(array::NamedArray; flip_as::Maybe{Type} = nothing)::AbstractArray
    return canonical_array(array.array; flip_as)
end

function canonical_array(array::DiskArrays.AbstractSubDiskArray; flip_as::Maybe{Type} = nothing)::AbstractArray
    return canonical_array(DiskArrays.subarray(array); flip_as)
end

function canonical_array(
    matrix::Union{
        Transpose{<:Any, <:AbstractMatrix},
        Adjoint{<:Any, <:AbstractMatrix},
        PermutedDimsArray{<:Any, 2, (Columns, Rows)},
        DiskArrays.AbstractPermutedDiskArray{<:Any, 2, (Columns, Rows)},
    };
    flip_as::Maybe{Type} = nothing,
)::AbstractMatrix
    return flipped_canonical_array(parent(matrix), flip_as)
end

function canonical_array(
    matrix::Union{
        PermutedDimsArray{<:Any, 2, (Rows, Columns)},
        DiskArrays.AbstractPermutedDiskArray{<:Any, 2, (Rows, Columns)},
    };
    flip_as::Maybe{Type} = nothing,
)::AbstractMatrix
    return canonical_array(parent(matrix); flip_as)
end

function canonical_array(array::SubArray; flip_as::Maybe{Type} = nothing)::AbstractArray
    parent_array = canonical_array(parent(array))
    indices = parentindices(array)
    if length(indices) == 2 && is_flipped_matrix(parent_array)
        # A view of a flip is a flip of the view of its parent, with the indices swapped.
        flipped_view = view(parent(parent_array), reverse(indices)...)
        return ndims(flipped_view) == 2 ? flipped_array(flipped_view, flip_as) : flipped_view
    elseif parent_array === parent(array)
        return array
    else
        return view(parent_array, indices...)
    end
end

# The canonical flip of the `matrix`. A flip of a flip cancels out.
function flipped_canonical_array(matrix::AbstractMatrix, flip_as::Maybe{Type})::AbstractMatrix
    canonical_matrix = canonical_array(matrix)
    if is_flipped_matrix(canonical_matrix)
        return parent(canonical_matrix)
    else
        return flipped_array(canonical_matrix, flip_as)
    end
end

# A flip of the `matrix`, of the class `flip_as`. By default, a `Transpose` for numbers, and a `PermutedDimsArray`
# otherwise.
function flipped_array(matrix::AbstractMatrix, flip_as::Maybe{Type})::AbstractMatrix
    if flip_as === nothing
        flip_as = eltype(matrix) <: Number ? Transpose : PermutedDimsArray
    end
    if flip_as === PermutedDimsArray
        return PermutedDimsArray(matrix, (Columns, Rows))
    elseif flip_as === Adjoint
        @assert eltype(matrix) <: Real
        return Adjoint(matrix)
    else
        @assert flip_as === Transpose
        return Transpose(matrix)
    end
end

# Whether the `array` is a flip of a matrix: a `Transpose`, an `Adjoint`, or a `PermutedDimsArray` (or the
# `PermutedDiskArray` of DiskArrays) that swaps the dimensions.
function is_flipped_matrix(::AbstractArray)::Bool
    return false
end

function is_flipped_matrix(
    ::Union{
        Transpose{<:Any, <:AbstractMatrix},
        Adjoint{<:Any, <:AbstractMatrix},
        PermutedDimsArray{<:Any, 2, (Columns, Rows)},
        DiskArrays.AbstractPermutedDiskArray{<:Any, 2, (Columns, Rows)},
    },
)::Bool
    return true
end

"""
    plain_array(
        array::AbstractArray;
        flip_as::Maybe{Type} = nothing,
        contiguous::Bool = false,
    )::AbstractArray

Return the values of the `array`, for reading all of them. The result is plain storage (an `Array`, `SparseMatrixCSC` or
`SparseVector`), a strided view of an `Array`, or a flip of either. It may share the storage of the `array`, so it must
not be modified. See [`MatrixFormats`](@ref TanayLabUtilities.MatrixFormats) for when to use which function.

The `array` is first put in its [`canonical_array`](@ref) form. This removes read-only and named wrappers without
copying. A strided view is kept, since code reads it efficiently. Set `contiguous` to copy a view with gaps, for code
that needs contiguous memory (e.g. HDF5 writers).

A flip is never copied. Instead its plain parent is flipped again. By default this is the same wrapper [`flip`](@ref)
uses: a `Transpose` for numbers, and a `PermutedDimsArray` otherwise. Set `flip_as` to `Transpose`,
`PermutedDimsArray` or `Adjoint` (for real numbers) to get a specific one, for code that only handles that one.

Any other array is copied. A lazy array type (which reads its data on demand) may add a method that returns its
materialization.

```jldoctest
matrix = [1 2; 3 4]
@assert plain_array(read_only_array(matrix)) === matrix
@assert parent(plain_array(transpose(matrix); flip_as = PermutedDimsArray)) === matrix

vector = [1, 2, 3, 4, 5]
@assert plain_array(view(vector, 1:2:5)) isa SubArray
@assert plain_array(view(vector, 1:2:5); contiguous = true) isa Vector

# output

```
"""
function plain_array(
    array::AbstractArray;
    flip_as::Maybe{Type} = nothing,  # NOLINT
    contiguous::Bool = false,  # NOLINT
)::AbstractArray
    return plain_copy(array)
end

function plain_array(
    array::Union{Array, SparseMatrixCSC, SparseVector};
    flip_as::Maybe{Type} = nothing,  # NOLINT
    contiguous::Bool = false,  # NOLINT
)::AbstractArray
    return array
end

function plain_array(array::SubArray; flip_as::Maybe{Type} = nothing, contiguous::Bool = false)::AbstractArray
    canonical_view = canonical_array(array; flip_as)
    if canonical_view !== array
        return plain_array(canonical_view; flip_as, contiguous)
    elseif is_plain_array(array) && (!contiguous || is_contiguous_array(array))
        return array
    else
        return plain_copy(array)
    end
end

# Whether the `array` is plain storage (an `Array`, `SparseMatrixCSC` or `SparseVector`) or a strided view of an
# `Array`, which [`plain_array`](@ref) returns as is (unless it must be contiguous).
function is_plain_array(array::AbstractArray)::Bool
    return array isa Union{Array, SparseMatrixCSC, SparseVector} ||
           (array isa SubArray && parent(array) isa Array && array isa StridedArray)
end

function plain_array(array::ReadOnlyArray; flip_as::Maybe{Type} = nothing, contiguous::Bool = false)::AbstractArray
    return plain_array(parent(array); flip_as, contiguous)
end

function plain_array(array::NamedArray; flip_as::Maybe{Type} = nothing, contiguous::Bool = false)::AbstractArray
    return plain_array(array.array; flip_as, contiguous)
end

function plain_array(
    array::Union{Transpose, Adjoint, PermutedDimsArray, DiskArrays.AbstractPermutedDiskArray};
    flip_as::Maybe{Type} = nothing,
    contiguous::Bool = false,
)::AbstractArray
    canonical = canonical_array(array; flip_as)
    if is_flipped_matrix(canonical)
        return flipped_plain_array(parent(canonical), flip_as, contiguous)
    elseif canonical !== array
        return plain_array(canonical; flip_as, contiguous)
    else
        return plain_copy(array)  # NOJET
    end
end

# A copy of the values of the `array` in plain storage.
function plain_copy(array::AbstractArray)::AbstractArray
    if issparse(array)
        return ndims(array) == 1 ? SparseVector(array) : SparseMatrixCSC(array)  # NOJET
    else
        return Array(array)
    end
end

"""
    is_contiguous_array(array::AbstractArray)::Bool

Whether the elements of the `array` are adjacent in memory, in column-major order. This is `false` for any array that
isn't strided (e.g. a transpose, or a lazy array).

```jldoctest
matrix = [1 2 3; 4 5 6]
@assert is_contiguous_array(matrix)
@assert is_contiguous_array(view(matrix, :, 2:3))
@assert !is_contiguous_array(view(matrix, 1, :))
@assert !is_contiguous_array(transpose(matrix))

# output

```
"""
function is_contiguous_array(::AbstractArray)::Bool
    return false
end

function is_contiguous_array(array::StridedArray)::Bool
    expected_stride = 1
    for dimension in 1:ndims(array)
        if size(array, dimension) > 1 && stride(array, dimension) != expected_stride
            return false
        end
        expected_stride *= size(array, dimension)
    end
    return true
end

"""
    is_costly_to_read(array::AbstractArray)::Bool

Whether reading all the values of the `array` is costly, so code that doesn't need them (e.g. [`brief`](@ref)) should
avoid reading them. This is `false` by default. It is `true` for a DiskArrays array, which reads its values from storage
on demand, and for a view or a wrapper of a costly array. A lazy array type (which reads its data on demand) may add a
method, which returns `true` until its data is in memory.

```jldoctest
DiskArrays = TanayLabUtilities.MatrixLayouts.DiskArrays

@assert !is_costly_to_read(rand(3, 4))
@assert is_costly_to_read(view(transpose(DiskArrays.TestTypes.AccessCountDiskArray(rand(3, 4))), 1:2, :))

# output

```
"""
function is_costly_to_read(::AbstractArray)::Bool
    return false
end

function is_costly_to_read(
    array::Union{SubArray, Transpose, Adjoint, PermutedDimsArray, SparseArrays.ReadOnly, NamedArray},
)::Bool
    return is_costly_to_read(parent(array))
end

function is_costly_to_read(::DiskArrays.AbstractDiskArray)::Bool
    return true
end

# The flip of the `matrix`, wrapping its plain storage. A flipped plain storage is flipped back to the storage.
function flipped_plain_array(matrix::AbstractMatrix, flip_as::Maybe{Type}, contiguous::Bool)::AbstractMatrix
    plain_matrix = plain_array(matrix; contiguous)
    if is_flipped_matrix(plain_matrix)
        return parent(plain_matrix)
    else
        return flipped_array(plain_matrix, flip_as)
    end
end

end # module
