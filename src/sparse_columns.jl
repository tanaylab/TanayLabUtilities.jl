"""
Read the values of some of the rows of the columns of a `SparseMatrixCSC`.

Reading `matrix[row, column]` for many rows costs a binary search per row, while walking a whole column visits every
stored value of the column, wanted or not. The functions here take the wanted rows as a sorted vector of row indices,
and merge it with the sorted rows of each column, galloping forward for each wanted row. This costs about
`s * log(nnz / s)` for `s` wanted rows and `nnz` stored values in the column. So it behaves like `s` binary searches
when few rows are wanted, and like a walk of the column when most rows are wanted.

The position of a wanted row is its index in the vector of wanted rows. Callers use it to index their own buffers.

These functions take a plain `SparseMatrixCSC` and a column index. A caller with a lazy matrix converts it once, outside
its loop, using `plain_array` for all of it or `matrix[:, columns]` for a few of its columns. Most of them also take a
single sparse column (a `SparseVector`, or a view of a column of a `SparseMatrixCSC`), such as the columns
[`parallel_foreach_column`](@ref) gives, which reads a lazy matrix one column at a time.
"""
module SparseColumns

export gather_sparse_column!
export stored_values_at_rows
export sum_sparse_columns_per_group!

using SparseArrays

import SparseArrays.SparseVectorUnion

"""
    stored_values_at_rows(
        matrix::SparseMatrixCSC,
        column_index::Integer,
        row_indices::AbstractVector{<:Integer},
    )

    stored_values_at_rows(
        column::SparseArrays.SparseVectorUnion,
        row_indices::AbstractVector{<:Integer},
    )

Iterate on the stored values of one column of a sparse `matrix`, or of a sparse `column` (a `SparseVector`, or a view
of a column of a `SparseMatrixCSC`), in some of its rows. The wanted `row_indices` must be sorted in increasing order,
with no repetitions. This yields `(position, value)` for each wanted row that has a stored value, where `position` is
the index of the row in `row_indices`. Wanted rows that have no stored value (that is, are zero) are skipped.

```jldoctest
using SparseArrays
using TanayLabUtilities.SparseColumns

matrix = sparse([0 1; 2 0; 3 4; 0 5])

@assert collect(stored_values_at_rows(matrix, 1, [1, 3, 4])) == [(2, 3)]
@assert collect(stored_values_at_rows(matrix, 2, [1, 3, 4])) == [(1, 1), (2, 4), (3, 5)]
@assert collect(stored_values_at_rows(view(matrix, :, 2), [1, 3, 4])) == [(1, 1), (2, 4), (3, 5)]

# output

```
"""
function stored_values_at_rows(
    matrix::SparseMatrixCSC,
    column_index::Integer,
    row_indices::AbstractVector{<:Integer},
)::StoredValuesAtRows
    @assert issorted(row_indices; lt = <=) "row indices are not strictly increasing"
    return StoredValuesAtRows(rowvals(matrix), nonzeros(matrix), nzrange(matrix, column_index), row_indices)
end

function stored_values_at_rows(column::SparseVectorUnion, row_indices::AbstractVector{<:Integer})::StoredValuesAtRows
    @assert issorted(row_indices; lt = <=) "row indices are not strictly increasing"
    return StoredValuesAtRows(SparseArrays.nonzeroinds(column), nonzeros(column), 1:nnz(column), row_indices)
end

struct StoredValuesAtRows{T, I <: AbstractVector{<:Integer}, V <: AbstractVector{T}, R <: AbstractVector{<:Integer}}
    row_index_per_stored::I
    value_per_stored::V
    stored_indices::UnitRange{Int}
    row_indices::R
end

Base.IteratorSize(::Type{<:StoredValuesAtRows}) = Base.SizeUnknown()

function Base.eltype(::Type{<:StoredValuesAtRows{T}}) where {T}  # ONLY SEEMS UNTESTED
    return Tuple{Int, T}
end

function Base.iterate(
    iterator::StoredValuesAtRows{T},
    state::Tuple{Int, Int} = (1, first(iterator.stored_indices)),
)::Union{Nothing, Tuple{Tuple{Int, T}, Tuple{Int, Int}}} where {T}
    position, stored_index = state
    row_index_per_stored = iterator.row_index_per_stored
    last_stored_index = last(iterator.stored_indices)
    @inbounds while position <= length(iterator.row_indices) && stored_index <= last_stored_index
        row_index = iterator.row_indices[position]
        stored_index = first_stored_index_at_or_after(row_index_per_stored, stored_index, last_stored_index, row_index)
        if stored_index <= last_stored_index && row_index_per_stored[stored_index] == row_index
            return ((position, iterator.value_per_stored[stored_index]), (position + 1, stored_index + 1))
        end
        position += 1
    end
    return nothing
end

# The first index from `stored_index` to `last_stored_index` whose row is at least `row_index`, or
# `last_stored_index + 1` if there is none. Gallops forward from `stored_index`, doubling the step, then searches the
# last step.
function first_stored_index_at_or_after(
    row_index_per_stored::AbstractVector{<:Integer},
    stored_index::Integer,
    last_stored_index::Integer,
    row_index::Integer,
)::Int
    @inbounds if row_index_per_stored[stored_index] >= row_index
        return stored_index
    end
    low_index = stored_index
    step = 1
    @inbounds while low_index + step <= last_stored_index && row_index_per_stored[low_index + step] < row_index
        low_index += step
        step *= 2
    end
    high_index = min(low_index + step, last_stored_index + 1)
    return searchsortedfirst(row_index_per_stored, row_index, low_index + 1, high_index - 1, Base.Order.Forward)
end

"""
    gather_sparse_column!(
        destination::AbstractVector,
        matrix::SparseMatrixCSC,
        column_index::Integer,
        row_indices::AbstractVector{<:Integer},
    )::Bool

    gather_sparse_column!(
        destination::AbstractVector,
        column::SparseArrays.SparseVectorUnion,
        row_indices::AbstractVector{<:Integer},
    )::Bool

Gather the values of one column of a sparse `matrix`, or of a sparse `column`, in some of its rows, into a dense
`destination`. The wanted `row_indices` must be sorted in increasing order, with no repetitions. The value of each
wanted row is placed in the `destination` at the row's position in `row_indices`. Returns whether any of the gathered
values is non-zero.

```jldoctest
using SparseArrays
using TanayLabUtilities.SparseColumns

matrix = sparse([0 1; 2 0; 3 4])
destination = Vector{Int}(undef, 2)

@assert gather_sparse_column!(destination, matrix, 1, [2, 3])
@assert destination == [2, 3]

destination = Vector{Int}(undef, 1)
@assert !gather_sparse_column!(destination, matrix, 2, [2])
@assert destination == [0]

@assert gather_sparse_column!(destination, view(matrix, :, 2), [3])
@assert destination == [4]

# output

```
"""
function gather_sparse_column!(
    destination::AbstractVector,
    matrix::SparseMatrixCSC,
    column_index::Integer,
    row_indices::AbstractVector{<:Integer},
)::Bool
    return gather_stored_values!(destination, stored_values_at_rows(matrix, column_index, row_indices))
end

function gather_sparse_column!(
    destination::AbstractVector,
    column::SparseVectorUnion,
    row_indices::AbstractVector{<:Integer},
)::Bool
    return gather_stored_values!(destination, stored_values_at_rows(column, row_indices))
end

# Gather the `stored_values` of the wanted rows into the `destination`, which has an entry per wanted row.
function gather_stored_values!(destination::AbstractVector, stored_values::StoredValuesAtRows)::Bool
    @assert length(destination) == length(stored_values.row_indices)
    fill!(destination, 0)
    is_any_nonzero = false
    for (position, value) in stored_values
        destination[position] = value
        is_any_nonzero |= !iszero(value)
    end
    return is_any_nonzero
end

"""
    sum_sparse_columns_per_group!(
        destination::AbstractMatrix,
        matrix::SparseMatrixCSC,
        row_indices::AbstractVector{<:Integer},
        column_indices_per_group::AbstractVector{<:AbstractVector{<:Integer}},
    )::Nothing

Add the sums of groups of columns of a sparse `matrix`, in some of its rows, into a dense `destination`. The wanted
`row_indices` must be sorted in increasing order, with no repetitions. The `destination` has a row per group and a
column per position in `row_indices`. The value of each wanted row in each group is the sum of the row's values in the
group's `column_indices_per_group`. This is added to the `destination`, which isn't cleared first.

```jldoctest
using SparseArrays
using TanayLabUtilities.SparseColumns

matrix = sparse([1 0 2; 0 3 0; 4 5 6])
destination = zeros(Int, 2, 2)

sum_sparse_columns_per_group!(destination, matrix, [1, 3], [[1, 3], [2]])
@assert destination == [3 10; 0 5]

# output

```
"""
function sum_sparse_columns_per_group!(
    destination::AbstractMatrix,
    matrix::SparseMatrixCSC,
    row_indices::AbstractVector{<:Integer},
    column_indices_per_group::AbstractVector{<:AbstractVector{<:Integer}},
)::Nothing
    @assert size(destination) == (length(column_indices_per_group), length(row_indices))
    for (group_index, column_indices) in enumerate(column_indices_per_group)
        for column_index in column_indices
            for (position, value) in stored_values_at_rows(matrix, column_index, row_indices)
                destination[group_index, position] += value
            end
        end
    end
    return nothing
end

end  # module
