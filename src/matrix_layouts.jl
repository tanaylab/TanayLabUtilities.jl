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
export major_axis
export minor_axis
export other_axis
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
Rows = 1

"""
A symbolic name for the rows axis. It is more readable to write, say, `size(matrix, Columns)`, instead of
`size(matrix, 2)`.
"""
Columns = 2

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

function major_axis(matrix::PermutedDimsArray{T, 2, P, IP, A})::Maybe{Int8} where {T, P, IP, A}
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
    return major_axis(matrix.parent)
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
function other_axis(axis::Maybe{Integer})::Maybe{Int8}
    if axis === nothing
        return nothing
    end

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

function unnamed_relayout(destination::SparseMatrixCSC, source::AbstractMatrix)::SparseMatrixCSC
    @assert size(destination) == size(source)
    @assert issparse(source)
    flipped_source = flip(mutable_array(source))
    if !(flipped_source isa SparseMatrixCSC && sparse_relayout_by_bands!(destination, flipped_source))
        @assert LinearAlgebra.transpose!(destination, flipped_source) === destination
    end
    return destination
end

function unnamed_relayout(destination::DenseMatrix, source::AbstractMatrix)::DenseMatrix
    @assert size(destination) == size(source)
    @assert !issparse(source)
    flipped_source = flip(mutable_array(source))
    if !dense_relayout_by_bands!(destination, flipped_source)
        @assert LinearAlgebra.transpose!(destination, flipped_source) === destination
    end
    return destination
end

function unnamed_relayout(destination::AbstractMatrix, source::AbstractMatrix)::AbstractMatrix  # UNTESTED
    return error("""
               unsupported relayout destination: $(typeof(destination))
               and source: $(typeof(source))
               """)
end

# Whether to relayout `data_bytes` of data in bands, rather than using `transpose!`.
function is_relayout_by_bands(data_bytes::Integer)::Bool
    global RELAYOUT_BUDGET_KB
    return data_bytes > RELAYOUT_BUDGET_KB * 1024
end

# The number of tasks, and the total scratch memory (in bytes), to relayout `data_bytes` of data in bands. The writes of
# each task are at least `RELAYOUT_MIN_WRITE_KB` if its memory is at least `sqrt(2 * data_bytes * min_write_bytes)`. If
# the budget does not allow even a single such task, then it is exceeded.
function relayout_tasks_and_budget(data_bytes::Integer)::Tuple{Int, Int}
    global RELAYOUT_BUDGET_KB
    global RELAYOUT_MIN_WRITE_KB
    budget_bytes = RELAYOUT_BUDGET_KB * 1024
    task_bytes = ceil(Int, sqrt(2.0 * data_bytes * RELAYOUT_MIN_WRITE_KB * 1024))
    n_tasks = clamp(budget_bytes ÷ task_bytes, 1, nthreads())
    return (n_tasks, max(budget_bytes, n_tasks * task_bytes))
end

# Invoke `task_function(task_index)` for each of the `n_tasks`, in parallel.
function run_relayout_tasks(task_function::Function, n_tasks::Integer)::Nothing
    if n_tasks == 1
        task_function(1)
    else
        @sync for task_index in 1:n_tasks  # NOJET
            Threads.@spawn task_function(task_index)
        end
    end
    return nothing
end

# Split the entries `1:(length(cumulative_weights) - 1)` into `n_chunks` contiguous ranges with about the same total
# weight. The `cumulative_weights` are like a sparse matrix `colptr`.
function relayout_chunks(cumulative_weights::AbstractVector{<:Integer}, n_chunks::Integer)::Vector{UnitRange{Int}}
    n_entries = length(cumulative_weights) - 1
    total_weight = cumulative_weights[end] - cumulative_weights[1]
    chunk_firsts = [1]
    for chunk_index in 1:(n_chunks - 1)
        target_weight = cumulative_weights[1] + total_weight * chunk_index / n_chunks
        chunk_first = searchsortedfirst(cumulative_weights, target_weight)
        push!(chunk_firsts, clamp(chunk_first, chunk_firsts[end], n_entries + 1))
    end
    push!(chunk_firsts, n_entries + 1)
    return [chunk_firsts[chunk_index]:(chunk_firsts[chunk_index + 1] - 1) for chunk_index in 1:n_chunks]
end

# The number of bits needed to hold the non-negative `value`.
function n_bits_for(value::Integer)::Int
    return 64 - leading_zeros(UInt64(value))
end

# Relayout the large sparse `source` into the `destination` in bands of the `destination` columns, which are the
# `source` rows. Pass 1 stores each entry with a packed index, which holds both its column in the band and its row. A
# band is therefore at most as wide as the index type allows. Returns whether this was done; if not, use `transpose!`
# instead.
function sparse_relayout_by_bands!(destination::SparseMatrixCSC{Td, Id}, source::SparseMatrixCSC)::Bool where {Td, Id}
    n_destination_rows, n_destination_columns = size(destination)
    @assert size(source) == (n_destination_columns, n_destination_rows)
    n_entries = nnz(source)
    entry_bytes = sizeof(Id) + sizeof(Td)
    if !is_relayout_by_bands(n_entries * entry_bytes)
        return false
    end
    row_bits = n_bits_for(n_destination_rows - 1)
    band_column_bits = 8 * sizeof(Id) - (Id <: Signed ? 1 : 0) - row_bits
    @assert band_column_bits >= 0
    @assert length(destination.rowval) >= n_entries
    @assert length(destination.nzval) >= n_entries

    n_tasks, budget_bytes = relayout_tasks_and_budget(n_entries * entry_bytes)
    colptr = flame_timed("relayout.count") do
        return count_sparse_relayout_colptr(source, n_destination_columns)
    end
    copyto!(destination.colptr, colptr)

    max_band_entries = max(1, budget_bytes ÷ (2 * n_tasks * entry_bytes))
    band_first_columns, band_of_column = sparse_relayout_bands(colptr, max_band_entries, 1 << min(band_column_bits, 62))
    n_bands = length(band_first_columns) - 1

    chunks = relayout_chunks(source.colptr, n_tasks)
    positions = Matrix{Int}(undef, n_bands, n_tasks)
    run_relayout_tasks(n_tasks) do task_index
        return count_sparse_relayout_chunk_bands!(
            view(positions, :, task_index),
            source,
            chunks[task_index],
            band_of_column,
        )
    end
    @inbounds for band_index in 1:n_bands
        position = colptr[band_first_columns[band_index]]
        for task_index in 1:n_tasks
            n_chunk_band_entries = positions[band_index, task_index]
            positions[band_index, task_index] = position
            position += n_chunk_band_entries
        end
    end

    buffer_entries = max(1, budget_bytes ÷ (n_tasks * n_bands * entry_bytes))
    flame_timed("relayout.fill_bands") do
        run_relayout_tasks(n_tasks) do task_index
            return fill_sparse_relayout_bands!(
                destination,
                source,
                chunks[task_index],
                band_first_columns,
                band_of_column,
                view(positions, :, task_index),
                row_bits,
                buffer_entries,
            )
        end
    end

    max_band_width =
        maximum(band_first_columns[band_index + 1] - band_first_columns[band_index] for band_index in 1:n_bands)
    max_entries_in_band = maximum(
        colptr[band_first_columns[band_index + 1]] - colptr[band_first_columns[band_index]] for band_index in 1:n_bands
    )
    next_band_index = Atomic{Int}(1)
    flame_timed("relayout.sort_bands") do
        run_relayout_tasks(n_tasks) do _
            return sort_sparse_relayout_bands!(
                destination,
                colptr,
                band_first_columns,
                next_band_index,
                row_bits,
                min(max_entries_in_band, max_band_entries),
                max_band_width,
            )
        end
    end

    return true
end

# The `colptr` of the relayout of the sparse `source`, which counts the entries of each of its rows.
function count_sparse_relayout_colptr(source::SparseMatrixCSC, n_destination_columns::Integer)::Vector{Int}
    colptr = zeros(Int, n_destination_columns + 1)
    colptr[1] = 1
    source_rowval = source.rowval
    @inbounds for source_position in 1:nnz(source)
        colptr[source_rowval[source_position] + 1] += 1
    end
    return cumsum!(colptr, colptr)
end

# Split the relayout destination columns into bands. A band has at most `max_band_entries` entries (unless it is a
# single column), and at most `max_band_columns` columns. Returns the first column of each band (followed by one past
# the last column), and the band of each column.
function sparse_relayout_bands(
    colptr::AbstractVector{Int},
    max_band_entries::Integer,
    max_band_columns::Integer,
)::Tuple{Vector{Int}, Vector{Int32}}
    n_columns = length(colptr) - 1
    band_first_columns = Int[]
    band_of_column = Vector{Int32}(undef, n_columns)
    first_column = 1
    while first_column <= n_columns
        last_column = first_column
        while last_column < n_columns &&
                  last_column - first_column + 1 < max_band_columns &&
                  colptr[last_column + 2] - colptr[first_column] <= max_band_entries
            last_column += 1
        end
        push!(band_first_columns, first_column)
        band_of_column[first_column:last_column] .= length(band_first_columns)
        first_column = last_column + 1
    end
    push!(band_first_columns, n_columns + 1)
    return (band_first_columns, band_of_column)
end

# Count the entries of the `source_columns` of the sparse `source` that go to each band.
function count_sparse_relayout_chunk_bands!(
    n_band_entries::AbstractVector{Int},
    source::SparseMatrixCSC,
    source_columns::UnitRange{Int},
    band_of_column::Vector{Int32},
)::Nothing
    n_band_entries .= 0
    if !isempty(source_columns)
        source_rowval = source.rowval
        @inbounds for source_position in
                      source.colptr[first(source_columns)]:(source.colptr[last(source_columns) + 1] - 1)

            n_band_entries[band_of_column[source_rowval[source_position]]] += 1
        end
    end
    return nothing
end

# Pass 1 of a sparse relayout. Append the entries of the `source_columns` of the sparse `source` to their bands in the
# `destination`, starting at the `positions` of this chunk in each band. Each entry is buffered with a packed index of
# its column in the band and its row, and the buffers are written when full.
function fill_sparse_relayout_bands!(
    destination::SparseMatrixCSC{Td, Id},
    source::SparseMatrixCSC,
    source_columns::UnitRange{Int},
    band_first_columns::Vector{Int},
    band_of_column::Vector{Int32},
    positions::AbstractVector{Int},
    row_bits::Int,
    buffer_entries::Int,
)::Nothing where {Td, Id}
    n_bands = length(band_first_columns) - 1
    indices_buffers = Matrix{Id}(undef, buffer_entries, n_bands)
    values_buffers = Matrix{Td}(undef, buffer_entries, n_bands)
    n_buffered_entries = zeros(Int, n_bands)
    source_colptr = source.colptr
    source_rowval = source.rowval
    source_nzval = source.nzval

    @inbounds for source_column in source_columns
        packed_row = Id(source_column - 1)
        for source_position in source_colptr[source_column]:(source_colptr[source_column + 1] - 1)
            destination_column = source_rowval[source_position]
            band_index = band_of_column[destination_column]
            n_buffered = n_buffered_entries[band_index] + 1
            packed_column = Id(destination_column - band_first_columns[band_index]) << row_bits
            indices_buffers[n_buffered, band_index] = packed_column | packed_row
            values_buffers[n_buffered, band_index] = source_nzval[source_position]
            if n_buffered == buffer_entries
                write_sparse_relayout_buffer!(
                    destination,
                    indices_buffers,
                    values_buffers,
                    band_index,
                    n_buffered,
                    positions,
                )
                n_buffered = 0
            end
            n_buffered_entries[band_index] = n_buffered
        end
    end

    for band_index in 1:n_bands
        write_sparse_relayout_buffer!(
            destination,
            indices_buffers,
            values_buffers,
            band_index,
            n_buffered_entries[band_index],
            positions,
        )
    end
    return nothing
end

# Write the first `n_buffered` entries of a band's buffers to the `destination` at the band's position, and advance it.
@inline function write_sparse_relayout_buffer!(
    destination::SparseMatrixCSC,
    indices_buffers::Matrix,
    values_buffers::Matrix,
    band_index::Integer,
    n_buffered::Integer,
    positions::AbstractVector{Int},
)::Nothing
    if n_buffered > 0
        first_buffered = (band_index - 1) * size(indices_buffers, 1) + 1
        copyto!(destination.rowval, positions[band_index], indices_buffers, first_buffered, n_buffered)
        copyto!(destination.nzval, positions[band_index], values_buffers, first_buffered, n_buffered)
        positions[band_index] += n_buffered
    end
    return nothing
end

# Pass 2 of a sparse relayout. Take the next band until there are none left. Sort the entries of each band by their
# column in the band, keeping the order of the rows in each column, and unpack the rows from the packed indices. A band
# of a single column needs no sorting, so it is unpacked in pieces, and may be larger than `max_band_entries`.
function sort_sparse_relayout_bands!(
    destination::SparseMatrixCSC{Td, Id},
    colptr::Vector{Int},
    band_first_columns::Vector{Int},
    next_band_index::Atomic{Int},
    row_bits::Int,
    max_band_entries::Int,
    max_band_width::Int,
)::Nothing where {Td, Id}
    n_bands = length(band_first_columns) - 1
    destination_rowval = destination.rowval
    destination_nzval = destination.nzval
    row_mask = (one(Id) << row_bits) - one(Id)
    packed_indices = Vector{Id}(undef, max_band_entries)
    band_values = Vector{Td}(undef, max_band_entries)
    sorted_rows = Vector{Id}(undef, max_band_entries)
    sorted_values = Vector{Td}(undef, max_band_entries)
    next_positions = Vector{Int}(undef, max_band_width)

    while true
        band_index = atomic_add!(next_band_index, 1)
        if band_index > n_bands
            return nothing
        end
        first_column = band_first_columns[band_index]
        band_width = band_first_columns[band_index + 1] - first_column
        first_position = colptr[first_column]
        n_band_entries = colptr[first_column + band_width] - first_position

        if band_width == 1
            for first_piece_offset in 0:max_band_entries:(n_band_entries - 1)
                n_piece_entries = min(max_band_entries, n_band_entries - first_piece_offset)
                copyto!(packed_indices, 1, destination_rowval, first_position + first_piece_offset, n_piece_entries)
                @inbounds for entry_index in 1:n_piece_entries
                    packed_indices[entry_index] = (packed_indices[entry_index] & row_mask) + one(Id)
                end
                copyto!(destination_rowval, first_position + first_piece_offset, packed_indices, 1, n_piece_entries)
            end

        elseif n_band_entries > 0
            copyto!(packed_indices, 1, destination_rowval, first_position, n_band_entries)
            copyto!(band_values, 1, destination_nzval, first_position, n_band_entries)
            @inbounds for band_column in 1:band_width
                next_positions[band_column] = colptr[first_column + band_column - 1] - first_position + 1
            end
            @inbounds for entry_index in 1:n_band_entries
                packed_index = packed_indices[entry_index]
                band_column = Int(packed_index >> row_bits) + 1
                sorted_index = next_positions[band_column]
                next_positions[band_column] = sorted_index + 1
                sorted_rows[sorted_index] = (packed_index & row_mask) + one(Id)
                sorted_values[sorted_index] = band_values[entry_index]
            end
            copyto!(destination_rowval, first_position, sorted_rows, 1, n_band_entries)
            copyto!(destination_nzval, first_position, sorted_values, 1, n_band_entries)
        end
    end
end

# Relayout the large dense `source` into the `destination` in bands of the `destination` columns, which are the `source`
# rows. Pass 1 leaves each band's region holding the band's `source` rows, and pass 2 transposes each band in memory.
# Returns whether this was done; if not, use `transpose!` instead.
function dense_relayout_by_bands!(destination::DenseMatrix{Td}, source::AbstractMatrix)::Bool where {Td}
    n_destination_rows, n_destination_columns = size(destination)
    @assert size(source) == (n_destination_columns, n_destination_rows)
    data_bytes = length(destination) * sizeof(Td)
    if !is_relayout_by_bands(data_bytes)
        return false
    end

    n_tasks, budget_bytes = relayout_tasks_and_budget(data_bytes)
    column_bytes = n_destination_rows * sizeof(Td)
    band_width = max(1, budget_bytes ÷ (2 * n_tasks * column_bytes))
    n_bands = cld(n_destination_columns, band_width)
    buffer_rows = max(1, budget_bytes ÷ (n_tasks * n_bands * band_width * sizeof(Td)))
    destination_values = vec(destination)

    chunks = relayout_chunks(0:n_destination_rows, n_tasks)
    flame_timed("relayout.fill_bands") do
        run_relayout_tasks(n_tasks) do task_index
            return fill_dense_relayout_bands!(destination_values, source, chunks[task_index], band_width, buffer_rows)
        end
    end

    next_band_index = Atomic{Int}(1)
    flame_timed("relayout.transpose_bands") do
        run_relayout_tasks(n_tasks) do _
            return transpose_dense_relayout_bands!(
                destination_values,
                n_destination_rows,
                n_destination_columns,
                band_width,
                next_band_index,
            )
        end
    end

    return true
end

# Pass 1 of a dense relayout. Append the band slices of the `destination_rows` (which are `source` columns) to their
# bands in the `destination_values`. Each band's region ends up holding the band's `source` rows, in column-major order.
# The slices are buffered, and the buffers are written when they hold `buffer_rows` rows.
function fill_dense_relayout_bands!(
    destination_values::AbstractVector{Td},
    source::AbstractMatrix,
    destination_rows::UnitRange{Int},
    band_width::Int,
    buffer_rows::Int,
)::Nothing where {Td}
    n_destination_columns, n_destination_rows = size(source)
    n_bands = cld(n_destination_columns, band_width)
    buffers = [Vector{Td}(undef, band_width * buffer_rows) for _ in 1:n_bands]
    n_buffered_rows = 0

    for destination_row in destination_rows
        n_buffered_rows += 1
        for band_index in 1:n_bands
            first_column = (band_index - 1) * band_width + 1
            last_column = min(first_column + band_width - 1, n_destination_columns)
            n_band_columns = last_column - first_column + 1
            copyto!(
                buffers[band_index],
                (n_buffered_rows - 1) * n_band_columns + 1,
                view(source, first_column:last_column, destination_row),
                1,
                n_band_columns,
            )
        end

        if n_buffered_rows == buffer_rows || destination_row == last(destination_rows)
            first_buffered_row = destination_row - n_buffered_rows + 1
            for band_index in 1:n_bands
                first_column = (band_index - 1) * band_width + 1
                n_band_columns = min(band_width, n_destination_columns - first_column + 1)
                position = (first_column - 1) * n_destination_rows + (first_buffered_row - 1) * n_band_columns + 1
                copyto!(destination_values, position, buffers[band_index], 1, n_band_columns * n_buffered_rows)
            end
            n_buffered_rows = 0
        end
    end
    return nothing
end

# Pass 2 of a dense relayout. Take the next band until there are none left, and transpose it in memory. A band of a
# single column is already in place.
function transpose_dense_relayout_bands!(
    destination_values::AbstractVector{Td},
    n_destination_rows::Int,
    n_destination_columns::Int,
    band_width::Int,
    next_band_index::Atomic{Int},
)::Nothing where {Td}
    n_bands = cld(n_destination_columns, band_width)
    band_values = Vector{Td}(undef, band_width * n_destination_rows)
    transposed_values = Vector{Td}(undef, band_width * n_destination_rows)

    while true
        band_index = atomic_add!(next_band_index, 1)
        if band_index > n_bands
            return nothing
        end
        first_column = (band_index - 1) * band_width + 1
        n_band_columns = min(band_width, n_destination_columns - first_column + 1)
        if n_band_columns > 1
            n_band_values = n_band_columns * n_destination_rows
            position = (first_column - 1) * n_destination_rows + 1
            copyto!(band_values, 1, destination_values, position, n_band_values)
            transpose!(
                reshape(view(transposed_values, 1:n_band_values), n_destination_rows, n_band_columns),
                reshape(view(band_values, 1:n_band_values), n_band_columns, n_destination_rows),
            )
            copyto!(destination_values, position, transposed_values, 1, n_band_values)
        end
    end
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

function flipped(matrix::AbstractMatrix)::AbstractMatrix
    return flame_timed("flipped") do
        axis = require_major_axis(matrix)
        local result

        if issparse(matrix)
            flame_timed("flipped.sparse") do
                @assert axis == Columns
                result = SparseMatrixCSC(flip(mutable_array(matrix)))  # NOLINT
                return nothing
            end

        else
            flame_timed("flipped.dense") do
                if axis == Columns
                    result = Matrix{eltype(matrix)}(undef, size(matrix, Columns), size(matrix, Rows))
                    result = permutedims!(result, mutable_array(matrix), (2, 1))

                elseif axis == Rows  # UNTESTED
                    result = Matrix{eltype(matrix)}(undef, size(matrix, Rows), size(matrix, Columns))  # UNTESTED
                    result = permutedims!(result, flip(mutable_array(matrix)), (2, 1))  # UNTESTED
                    result = flip(result)  # UNTESTED

                else
                    @assert false
                end
            end
        end

        @assert major_axis(result) == axis
        @assert size(result, Rows) == size(matrix, Columns)
        @assert size(result, Columns) == size(matrix, Rows)

        return result
    end
end

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
`ReadOnlyArray` and/or `NamedArray` to cancel out an internal flip wrapper.

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

function flip(matrix::PermutedDimsArray{T, 2, P, IP, A})::AbstractMatrix where {T, P, IP, A}
    if P == (Columns, Rows)
        return parent(matrix)
    elseif P == (Rows, Columns)
        return flip(parent(matrix))
    else
        @assert false
    end
end

function flip(matrix::AbstractMatrix)::AbstractMatrix
    if eltype(matrix) <: Number
        return Transpose(matrix)
    else
        return PermutedDimsArray(matrix, (Columns, Rows))  # UNTESTED
    end
end

function flip(matrix::ReadOnlyArray)::AbstractMatrix
    return ReadOnlyArray(flip(parent(matrix)))
end

function flip(matrix::NamedArray)::AbstractMatrix
    @assert ndims(matrix) == 2
    return NamedArray(flip(matrix.array), reverse(matrix.dicts), reverse(matrix.dimnames))
end

end # module
