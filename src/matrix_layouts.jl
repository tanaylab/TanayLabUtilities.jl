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
export transposer
export GLOBAL_INEFFICIENT_ACTION_HANDLER

using ..Documentation
using ..Types
using ..Brief
using ..Handlers
using ..ReadOnlyArrays
using Distributed
using LinearAlgebra
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
using Test

base = [0 1 2; 3 4 0]

@test major_axis(base) == Columns

# Slice

@test major_axis(@view base[:, [1, 3, 2]]) == nothing

# Named

using NamedArrays

@test major_axis(NamedArray(base)) == Columns

# Permuted

permuted = PermutedDimsArray(base, (2, 1))
@test major_axis(permuted) == Rows

unpermuted = PermutedDimsArray(base, (1, 2))
@test major_axis(unpermuted) == Columns

# LinearAlgebra

transposed = transpose(base)
@test major_axis(transposed) == Rows

adjointed = adjoint(base)
@test major_axis(adjointed) == Rows

# Sparse

using SparseArrays

sparse = SparseMatrixCSC(base)
@test major_axis(sparse) == Columns

println("OK")

# output

OK
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

function major_axis(::BitMatrix)::Maybe{Int8}  # UNTESTED
    return Columns
end

function major_axis(matrix::AbstractMatrix)::Maybe{Int8}
    try
        matrix_strides = strides(matrix)
        if matrix_strides[Rows] == 1  # NOJET
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

"""
    require_major_axis(matrix::AbstractMatrix)::Int8

Similar to [`major_axis`](@ref) but will `error` if the matrix isn't in either row-major or column-major layout.

```jldoctest
using Test

base = [0 1 2; 3 4 0]

@test require_major_axis(base) == Columns

require_major_axis(@view base[:, [1, 3, 2]])

# output

ERROR: type: SubArray{Int64, 2, Matrix{Int64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Vector{Int64}}, false} is not in any-major layout
```
"""
function require_major_axis(matrix::AbstractMatrix)::Int8
    axis = major_axis(matrix)
    if axis === nothing
        error("type: $(typeof(matrix)) is not in any-major layout")
    end
    return axis
end

"""
    minor_axis(matrix::AbstractMatrix)::Maybe{Int8}

Return the index of the minor axis of a matrix, that is, the axis one should **vary** for an efficient inner loop
accessing the matrix elements. If the matrix doesn't support any efficient access axis, returns `nothing`.

```jldoctest
using Test

base = [0 1 2; 3 4 0]

@test minor_axis(base) == Rows

# Slice

@test minor_axis(@view base[:, [1, 3, 2]]) == nothing

# Named

using NamedArrays

@test minor_axis(NamedArray(base)) == Rows

# Permuted

permuted = PermutedDimsArray(base, (2, 1))
@test minor_axis(permuted) == Columns

unpermuted = PermutedDimsArray(base, (1, 2))
@test minor_axis(unpermuted) == Rows

# LinearAlgebra

transposed = transpose(base)
@test minor_axis(transposed) == Columns

adjointed = adjoint(base)
@test minor_axis(adjointed) == Columns

# Sparse

using SparseArrays

sparse = SparseMatrixCSC(base)
@test minor_axis(sparse) == Rows

println("OK")

# output

OK
```
"""
function minor_axis(matrix::AbstractMatrix)::Maybe{Int8}
    return other_axis(major_axis(matrix))
end

"""
    require_minor_axis(matrix::AbstractMatrix)::Int8

Similar to [`minor_axis`](@ref) but will `error` if the matrix isn't in either row-major or column-major layout.

```jldoctest
using Test

base = [0 1 2; 3 4 0]

@test require_minor_axis(base) == Rows

require_minor_axis(@view base[:, [1, 3, 2]])

# output

ERROR: type: SubArray{Int64, 2, Matrix{Int64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Vector{Int64}}, false} is not in any-major layout
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
using Test

@test other_axis(nothing) == nothing
@test other_axis(Rows) == Columns
@test other_axis(Columns) == Rows

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

macro assert_is_vector(source_file, source_line, vector)
    vector_name = string(vector)
    return esc(
        :(
            if !($vector isa AbstractVector)
                error(
                    "non-vector " *
                    $vector_name *
                    ": " *
                    brief($vector) *
                    "\nin: " *
                    $(string(source_file)) *
                    ":" *
                    $(string(source_line)),
                )
            end
        ),
    )
end

macro assert_vector_size(source_file, source_line, vector, n_elements)
    vector_name = string(vector)
    n_elements_name = string(n_elements)
    return esc(
        :(
            if length($vector) != $n_elements
                error(
                    "wrong size: " *
                    string(length($vector)) *
                    "\nof the vector: " *
                    $vector_name *
                    "\nis different from " *
                    $n_elements_name *
                    ": " *
                    string($n_elements) *
                    "\nin: " *
                    $(string(source_file)) *
                    ":" *
                    $(string(source_line)),
                )
            end
        ),
    )
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
    return esc(:(TanayLabUtilities.MatrixLayouts.@assert_is_vector($(__source__.file), $(__source__.line), $vector)))
end

macro assert_vector(vector, n_elements)
    return esc(
        :(  #
            TanayLabUtilities.MatrixLayouts.@assert_is_vector($(__source__.file), $(__source__.line), $vector);   #
            TanayLabUtilities.MatrixLayouts.@assert_vector_size(
                $(__source__.file),
                $(__source__.line),
                $vector,
                $n_elements
            )  #
        ),
    )
end

macro assert_is_matrix(source_file, source_line, matrix)
    matrix_name = string(matrix)
    return esc(
        :(
            if !($matrix isa AbstractMatrix)
                error(
                    "non-matrix " *
                    $matrix_name *
                    ": " *
                    brief($matrix) *
                    "\nin: " *
                    $(string(source_file)) *
                    ":" *
                    $(string(source_line)),
                )
            end
        ),
    )
end

macro assert_matrix_size(source_file, source_line, matrix, n_rows, n_columns)
    matrix_name = string(matrix)
    n_rows_name = string(n_rows)
    n_columns_name = string(n_columns)
    return esc(
        :(
            if size($matrix) != ($n_rows, $n_columns)
                error(
                    "wrong size: " *
                    string(size($matrix)) *
                    "\nof the matrix: " *
                    $matrix_name *
                    "\nis different from (" *
                    $n_rows_name *
                    ", " *
                    $n_columns_name *
                    "): (" *
                    string($n_rows) *
                    ", " *
                    string($n_columns) *
                    ")\nin: " *
                    $(string(source_file)) *
                    ":" *
                    $(string(source_line)),
                )
            end
        ),
    )
end

macro check_matrix_layout(source_file, source_line, matrix, major_axis)
    matrix_name = string(matrix)
    return esc(
        :(
            TanayLabUtilities.MatrixLayouts.check_efficient_action(
                $source_file,
                $source_line,
                $matrix_name,
                $matrix,
                $major_axis,
            ),
        ),
    )
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
    return esc(:(TanayLabUtilities.MatrixLayouts.@assert_is_matrix($(__source__.file), $(__source__.line), $matrix)))
end

macro assert_matrix(matrix, axis)
    return esc(
        :( #
            TanayLabUtilities.MatrixLayouts.@assert_is_matrix($(__source__.file), $(__source__.line), $matrix); #
            TanayLabUtilities.MatrixLayouts.@check_matrix_layout(
                $(string(__source__.file)),
                $(__source__.line),
                $matrix,
                $axis
            ) #
        ),
    )
end

macro assert_matrix(matrix, n_rows, n_columns)
    return esc(
        :(  #
            TanayLabUtilities.MatrixLayouts.@assert_is_matrix($(__source__.file), $(__source__.line), $matrix);  #
            TanayLabUtilities.MatrixLayouts.@assert_matrix_size(
                $(__source__.file),
                $(__source__.line),
                $matrix,
                $n_rows,
                $n_columns
            )  #
        ),
    )
end

macro assert_matrix(matrix, n_rows, n_columns, axis)
    return esc(
        :( #
            TanayLabUtilities.MatrixLayouts.@assert_is_matrix($(__source__.file), $(__source__.line), $matrix); #
            TanayLabUtilities.MatrixLayouts.@assert_matrix_size(
                $(__source__.file),
                $(__source__.line),
                $matrix,
                $n_rows,
                $n_columns
            ); #
            TanayLabUtilities.MatrixLayouts.@check_matrix_layout(
                $(string(__source__.file)),
                $(__source__.line),
                $matrix,
                $axis
            ) #
        ),
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
            text = brief(matrix)
            return """
                inefficient major axis: $(axis_name(major_axis(matrix)))
                for $(operand): $(text)
                in: $(source_file):$(source_line)
                """
        end
    end
end

"""
    relayout!(destination::AbstractMatrix, source::AbstractMatrix)::AbstractMatrix
    relayout!(destination::AbstractMatrix, source::NamedMatrix)::NamedMatrix

Return the same `matrix` data, but in the other memory layout.

Suppose you have a column-major UMIs matrix, whose rows are cells, and columns are genes. Therefore, looping on the UMIs
of a gene will be fast, but looping on the UMIs of a cell will be slow. A `transpose` (no `!`) of a matrix is fast; it
creates a zero-copy wrapper of the matrix with flipped axes, so its rows will be genes and columns will be cells, but in
row-major layout. Therefore, **still**, looping on the UMIs of a gene is fast, and looping on the UMIs of a cell is
slow.

In contrast, `transpose!` (with a `!`) (or [`transposer`](@ref)) is slow; it creates a rearranged copy of the data, also
returning a matrix whose rows are genes and columns are cells, but this time, in column-major layout. Therefore, in this
case looping on the UMIs of a gene will be slow, and looping on the UMIs of a cell will be fast.

!!! note

    It is almost always worthwhile to `relayout!` a matrix and then looping "with the grain" of the data, instead of
    skipping it and looping "against the grain" of the data. This is because (in Julia at least) the implementation of
    `transpose!` is optimized for the task, while the other operations typically don't provide any specific
    optimizations for working "against the grain" of the data. The benefits of a `relayout!` become more significant the
    more operations are done on the data in the loop.

The `relayout!` is essentially a zero-copy `transpose` of the slow `transpose!`. You end up with a matrix that
**appears** to be the same as the original (rows are cells and columns are genes), but behaves **differently** - looping
on the UMIs of a gene will be slow, and looping on the UMIs of a cell will be fast. In addition, `relayout!` will work
for both sparse and dense matrices. If the `source` is a `NamedMatrix`, then the result will be a `NamedMatrix` with the
same axes (zero-copy shared from the `source`). If `destination` is already a `NamedMatrix`, then its axes must match
`source`.

!!! note

    The caller is responsible for providing a sensible `destination` matrix (sparse for a sparse `source`, dense for a
    non-sparse `source`).

```jldoctest
using Test

using LinearAlgebra

source = rand(3, 4)
destination = transpose(rand(4, 3))

result = relayout!(destination, source)
@test result === destination
@test brief(source) == "3 x 4 x Float64 in Columns (Dense)"
@test brief(result) == "3 x 4 x Float64 in Rows (Transpose, Dense)"
@test result == source

# Named

using NamedArrays

named_source = NamedArray(rand(3, 4))
destination = transpose(rand(4, 3))
result = relayout!(destination, named_source)
@test parent(result) === destination
@test brief(named_source) == "3 x 4 x Float64 in Columns (Named, Dense)"
@test brief(result) == "3 x 4 x Float64 in Rows (Named, Transpose, Dense)"
@test result == named_source

source = rand(3, 4)
named_destination = NamedArray(transpose(rand(4, 3)))
result = relayout!(named_destination, source)
@test result === named_destination
@test brief(source) == "3 x 4 x Float64 in Columns (Dense)"
@test brief(result) == "3 x 4 x Float64 in Rows (Named, Transpose, Dense)"
@test result == source

source = rand(3, 4)
named_destination = Transpose(NamedArray(rand(4, 3)))
result = relayout!(named_destination, source)
@test result === named_destination
@test brief(source) == "3 x 4 x Float64 in Columns (Dense)"
@test brief(result) == "3 x 4 x Float64 in Rows (Transpose, Named, Dense)"
@test result == source

named_source = NamedArray(rand(3, 4))
named_destination = NamedArray(transpose(rand(4, 3)))
result = relayout!(named_destination, named_source)
@test result === named_destination
@test brief(named_source) == "3 x 4 x Float64 in Columns (Named, Dense)"
@test brief(result) == "3 x 4 x Float64 in Rows (Named, Transpose, Dense)"
@test result == named_source

# Permuted

source = rand(3, 4)
destination = PermutedDimsArray(rand(4, 3), (2,1))
result = relayout!(destination, source)
@test result === destination
@test brief(source) == "3 x 4 x Float64 in Columns (Dense)"
@test brief(result) == "3 x 4 x Float64 in Rows (Permute, Dense)"
@test result == source

source = rand(3, 4)
destination = PermutedDimsArray(adjoint(rand(4, 3)), (1,2))
result = relayout!(destination, source)
@test result === destination
@test brief(source) == "3 x 4 x Float64 in Columns (Dense)"
@test brief(result) == "3 x 4 x Float64 in Rows (!Permute, Adjoint, Dense)"
@test result == source

# Sparse

using SparseArrays

source = SparseMatrixCSC([0.0 1.0 2.0; 3.0 4.0 0.0])
destination = transpose(SparseMatrixCSC([30.0 0.0; 0.0 40.0; 20.0 10.0]))
result = relayout!(destination, source)
@test result === destination
@test brief(source) == "2 x 3 x Float64 in Columns (Sparse Int64 67%)"
@test brief(result) == "2 x 3 x Float64 in Rows (Transpose, Sparse Int64 67%)"
@test result == source

println("OK")

# output

OK
```
"""
function relayout!(destination::AbstractMatrix, source::AbstractMatrix)::AbstractMatrix
    @assert size(destination) == size(source)
    @assert major_axis(destination) == minor_axis(source)
    @assert issparse(destination) == issparse(source)
    return named_relayout(destination, source)
end

function named_relayout(destination::AbstractMatrix, source::NamedMatrix)::NamedArray
    @debug "relayout! destination: $(brief(destination)) source: $(brief(source)) {"
    @assert named_relayout(destination, parent(source)) === destination
    result = NamedArray(destination, source.dicts, source.dimnames)
    @debug "relayout! result: $(brief(result)) }"
    return result
end

function named_relayout(destination::NamedArray, source::NamedMatrix)::NamedArray
    @debug "relayout! destination: $(brief(destination)) source: $(brief(source)) {"
    @assert destination.dimnames == source.dimnames  # NOJET
    @assert destination.dicts == source.dicts
    @assert named_relayout(parent(destination), parent(source)) === parent(destination)
    @debug "relayout! result: $(brief(destination)) }"
    return destination
end

function named_relayout(destination::NamedArray, source::AbstractMatrix)::AbstractMatrix
    @debug "relayout! destination: $(brief(destination)) source: $(brief(source)) {"
    @assert named_relayout(parent(destination), source) === parent(destination)
    @debug "relayout! result: $(brief(destination)) }"
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
    @debug "relayout! destination: $(brief(destination)) source: $(brief(source)) {"
    if P == (Rows, Columns)
        @assert named_relayout(parent(destination), source) === parent(destination)
    elseif P == (Columns, Rows)
        @assert named_relayout(parent(destination), transpose(source)) === parent(destination)
    else
        @assert false
    end
    @debug "relayout! result: $(brief(destination)) }"
    return destination
end

function unnamed_relayout(destination::Transpose, source::AbstractMatrix)::AbstractMatrix
    @debug "relayout! destination: $(brief(destination)) source: $(brief(source)) {"
    @assert named_relayout(parent(destination), transpose(source)) === parent(destination)
    @debug "relayout! result: $(brief(destination)) }"
    return destination
end

function unnamed_relayout(destination::Adjoint, source::AbstractMatrix)::AbstractMatrix
    @debug "relayout! destination: $(brief(destination)) source: $(brief(source)) {"
    @assert named_relayout(parent(destination), adjoint(source)) === parent(destination)
    @debug "relayout! result: $(brief(destination)) }"
    return destination
end

function unnamed_relayout(destination::SparseMatrixCSC, source::AbstractMatrix)::SparseMatrixCSC
    @debug "relayout! destination: $(brief(destination)) source: $(brief(source)) {"
    @assert size(destination) == size(source)
    @assert issparse(source)
    @assert LinearAlgebra.transpose!(destination, transpose(mutable_array(source))) === destination  # NOJET
    @debug "relayout! result: $(brief(destination)) }"
    return destination
end

function unnamed_relayout(destination::DenseMatrix, source::AbstractMatrix)::DenseMatrix
    @debug "relayout! destination: $(brief(destination)) source: $(brief(source)) {"
    @assert size(destination) == size(source)
    @assert !issparse(source)
    @assert LinearAlgebra.transpose!(destination, transpose(mutable_array(source))) === destination
    @debug "relayout! result: $(brief(destination)) }"
    return destination
end

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
`transpose(transposer(matrix))`.

```jldoctest
base = rand(3, 4)
@assert relayout(base) == base
@assert major_axis(relayout(base)) == minor_axis(base)

# output
```
"""
function relayout(matrix::AbstractMatrix)::AbstractMatrix
    return transpose(transposer(matrix))
end

"""
    transposer(matrix::AbstractMatrix)::AbstractMatrix

Return a transpose of a matrix, but instead of simply using a zero-copy wrapper, it actually rearranges the data. See
[`relayout!`](@ref).

```jldoctest
using Test

# Dense

base = rand(3, 4)
@test transposer(base) == transpose(base)
@test brief(base) == "3 x 4 x Float64 in Columns (Dense)"
@test brief(transpose(base)) == "4 x 3 x Float64 in Rows (Transpose, Dense)"
@test brief(transposer(base)) == "4 x 3 x Float64 in Columns (Dense)"

# Named

using NamedArrays

base = NamedArray(rand(3, 4))
@test transposer(base) == transpose(base)
@test brief(base) == "3 x 4 x Float64 in Columns (Named, Dense)"
@test brief(transpose(base)) == "4 x 3 x Float64 in Rows (Named, Transpose, Dense)"
@test brief(transposer(base)) == "4 x 3 x Float64 in Columns (Named, Dense)"

# Permuted

base = PermutedDimsArray(rand(3, 4), (2,1))
@test transposer(base) == transpose(base)
@test brief(base) == "4 x 3 x Float64 in Rows (Permute, Dense)"
@test brief(transpose(base)) == "3 x 4 x Float64 in Columns (Transpose, Permute, Dense)"
@test brief(transposer(base)) == "3 x 4 x Float64 in Rows (Transpose, Dense)"

base = PermutedDimsArray(rand(3, 4), (1,2))
@test transposer(base) == transpose(base)
@test brief(base) == "3 x 4 x Float64 in Columns (!Permute, Dense)"
@test brief(transpose(base)) == "4 x 3 x Float64 in Rows (Transpose, !Permute, Dense)"
@test brief(transposer(base)) == "4 x 3 x Float64 in Columns (Dense)"

# LinearAlgebra

using LinearAlgebra

base = transpose(rand(3, 4))
@test transposer(base) == transpose(base)
@test brief(base) == "4 x 3 x Float64 in Rows (Transpose, Dense)"
@test brief(transpose(base)) == "3 x 4 x Float64 in Columns (Dense)"
@test brief(transposer(base)) == "3 x 4 x Float64 in Rows (Transpose, Dense)"

base = adjoint(rand(3, 4))
@test transposer(base) == transpose(base)
@test brief(base) == "4 x 3 x Float64 in Rows (Adjoint, Dense)"
@test brief(transpose(base)) == "3 x 4 x Float64 in Columns (Dense)"
@test brief(transposer(base)) == "3 x 4 x Float64 in Rows (Adjoint, Dense)"

# ReadOnly

base = read_only_array(rand(3, 4))
@test transposer(base) == transpose(base)
@test brief(base) == "3 x 4 x Float64 in Columns (ReadOnly, Dense)"
@test brief(transpose(base)) == "4 x 3 x Float64 in Rows (Transpose, ReadOnly, Dense)"
@test brief(transposer(base)) == "4 x 3 x Float64 in Columns (Dense)"

# Sparse

using SparseArrays

base = SparseMatrixCSC([0.0 1.0 2.0; 3.0 4.0 0.0])
@test transposer(base) == transpose(base)
@test brief(base) == "2 x 3 x Float64 in Columns (Sparse Int64 67%)"
@test brief(transpose(base)) == "3 x 2 x Float64 in Rows (Transpose, Sparse Int64 67%)"
@test brief(transposer(base)) == "3 x 2 x Float64 in Columns (Sparse Int64 67%)"

println("OK")

# output

OK
```
"""
function transposer(matrix::NamedMatrix)::NamedArray
    return NamedArray(transposer(parent(matrix)), flip_tuple(matrix.dicts), flip_tuple(matrix.dimnames))
end

function flip_tuple(tuple::Tuple{T1, T2})::Tuple{T2, T1} where {T1, T2}
    value1, value2 = tuple
    return (value2, value1)
end

function transposer(matrix::AbstractMatrix)::AbstractMatrix
    @debug "transposer $(brief(matrix)) {"

    axis = require_major_axis(matrix)

    if issparse(matrix)
        @assert axis == Columns
        result = SparseMatrixCSC(transpose(mutable_array(matrix)))

    elseif axis == Columns
        result = Matrix{eltype(matrix)}(undef, size(matrix, Columns), size(matrix, Rows))
        result = LinearAlgebra.transpose!(result, mutable_array(matrix))

    elseif axis == Rows  # UNTESTED
        result = Matrix{eltype(matrix)}(undef, size(matrix, Rows), size(matrix, Columns))  # UNTESTED
        result = LinearAlgebra.transpose!(result, transpose(mutable_array(matrix)))  # UNTESTED
        result = transpose(result)  # UNTESTED

    else
        @assert false
    end

    @assert major_axis(result) == axis
    @assert size(result, Rows) == size(matrix, Columns)
    @assert size(result, Columns) == size(matrix, Rows)

    @debug "transposer $(brief(result)) }"
    return result
end

function transposer(matrix::ReadOnlyArray)::AbstractMatrix
    return transposer(parent(matrix))
end

function transposer(matrix::PermutedDimsArray{T, 2, P, IP, A})::AbstractMatrix where {T, P, IP, A}
    if P == (Rows, Columns)
        return transposer(parent(matrix))
    elseif P == (Columns, Rows)
        return transpose(transposer(parent(matrix)))
    else
        @assert false
    end
end

function transposer(matrix::Transpose)::AbstractMatrix
    return transpose(transposer(parent(matrix)))
end

function transposer(matrix::Adjoint)::AbstractMatrix
    return adjoint(transposer(parent(matrix)))
end

end # module
