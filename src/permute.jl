"""
In-place permutation kernels for vectors and matrices, dense and sparse.
"""
module Permute

export permute_vector!
export permute_sparse_vector!
export permute_dense_matrix_rows!
export permute_dense_matrix_columns!
export permute_dense_matrix_both!
export permute_sparse_matrix_rows!
export permute_sparse_matrix_columns!
export permute_sparse_matrix_both!

using ..Documentation
using ..ParallelLoops
using ..Types
using ProgressMeter
using SparseArrays

"""
    permute_vector!(;
        destination::AbstractVector,
        source::AbstractVector,
        permutation::AbstractVector{<:Integer},
        progress::Maybe{Progress} = nothing,
    )::Nothing

Fill `destination` such that `destination[index] == source[permutation[index]]` for every `index`. If `progress` is
given, advance it by `length(source)` once after filling is complete.

The caller must pre-allocate `destination` with the same length as `source`. This function never allocates the
destination.

```jldoctest
source = [10, 20, 30, 40]
destination = similar(source)
permutation = [4, 2, 1, 3]
permute_vector!(; destination, source, permutation)
@assert destination == [40, 20, 10, 30]

# output

```
"""
function permute_vector!(;
    destination::AbstractVector,
    source::AbstractVector,
    permutation::AbstractVector{<:Integer},
    progress::Maybe{Progress} = nothing,
)::Nothing
    source_length = length(source)
    @assert length(destination) == source_length
    @assert length(permutation) == source_length
    @inbounds for destination_index in 1:source_length
        destination[destination_index] = source[permutation[destination_index]]
    end
    if progress !== nothing
        next!(progress; step = source_length)  # UNTESTED
    end
    return nothing
end

"""
    permute_sparse_vector!(;
        destination_nzind::AbstractVector{<:Integer},
        destination_nzval::AbstractVector,
        source::SparseVector,
        inverse_permutation::AbstractVector{<:Integer},
        progress::Maybe{Progress} = nothing,
    )::Nothing

Fill the `(nzind, nzval)` backing pair of a sparse vector destination such that
`destination[inverse_permutation[index]] == source[index]` for every `index`. Destination indices and values are
written in ascending `nzind` order so the result is a valid `SparseVector` backing pair. If `progress` is given,
advance it by `length(source)` (the dense length) once after filling is complete.

```jldoctest
using SparseArrays

source = sparsevec([1, 3, 5], [10.0, 30.0, 50.0], 5)
inverse_permutation = [5, 4, 3, 2, 1]
destination_nzind = Vector{Int}(undef, nnz(source))
destination_nzval = Vector{Float64}(undef, nnz(source))
permute_sparse_vector!(; destination_nzind, destination_nzval, source, inverse_permutation)
destination = SparseVector(length(source), destination_nzind, destination_nzval)
@assert destination == sparsevec([1, 3, 5], [50.0, 30.0, 10.0], 5)

# output

```
"""
function permute_sparse_vector!(;
    destination_nzind::AbstractVector{<:Integer},
    destination_nzval::AbstractVector,
    source::SparseVector,
    inverse_permutation::AbstractVector{<:Integer},
    progress::Maybe{Progress} = nothing,
)::Nothing
    source_length = length(source)
    source_nnz = nnz(source)
    @assert length(destination_nzind) == source_nnz
    @assert length(destination_nzval) == source_nnz
    @assert length(inverse_permutation) == source_length

    source_nzind = SparseArrays.nonzeroinds(source)
    source_nzval = nonzeros(source)

    permuted_nzind = Vector{Int}(undef, source_nnz)
    @inbounds for scan_index in 1:source_nnz
        permuted_nzind[scan_index] = inverse_permutation[source_nzind[scan_index]]
    end
    sort_order = sortperm(permuted_nzind)
    @inbounds for destination_index in 1:source_nnz
        ordered_scan_index = sort_order[destination_index]
        destination_nzind[destination_index] = permuted_nzind[ordered_scan_index]
        destination_nzval[destination_index] = source_nzval[ordered_scan_index]
    end

    if progress !== nothing
        next!(progress; step = source_length)  # UNTESTED
    end
    return nothing
end

"""
    permute_dense_matrix_columns!(;
        destination::AbstractMatrix,
        source::AbstractMatrix,
        columns_permutation::AbstractVector{<:Integer},
        progress::Maybe{Progress} = nothing,
    )::Nothing

Fill `destination` such that `destination[:, destination_column] == source[:, columns_permutation[destination_column]]`
for every `destination_column`. Parallelized per destination column. If `progress` is given, advance it by `n_rows` per
destination column completed.

```jldoctest
source = Float64[1 2 3; 4 5 6; 7 8 9]
destination = similar(source)
columns_permutation = [3, 1, 2]
permute_dense_matrix_columns!(; destination, source, columns_permutation)
@assert destination == [3.0 1.0 2.0; 6.0 4.0 5.0; 9.0 7.0 8.0]

# output

```
"""
function permute_dense_matrix_columns!(;
    destination::AbstractMatrix,
    source::AbstractMatrix,
    columns_permutation::AbstractVector{<:Integer},
    progress::Maybe{Progress} = nothing,
)::Nothing
    n_rows, n_columns = size(source)
    @assert size(destination) == (n_rows, n_columns)
    @assert length(columns_permutation) == n_columns

    parallel_loop_wo_rng(1:n_columns; name = "permute_dense_matrix_columns!") do destination_column
        source_column = columns_permutation[destination_column]
        @inbounds @views copyto!(destination[:, destination_column], source[:, source_column])
        if progress !== nothing
            next!(progress; step = n_rows)  # UNTESTED
        end
        return nothing
    end
    return nothing
end

"""
    permute_dense_matrix_rows!(;
        destination::AbstractMatrix,
        source::AbstractMatrix,
        rows_permutation::AbstractVector{<:Integer},
        progress::Maybe{Progress} = nothing,
    )::Nothing

Fill `destination` such that `destination[destination_row, :] == source[rows_permutation[destination_row], :]` for every
`destination_row`. Parallelized per destination column. If `progress` is given, advance it by `n_rows` per destination
column completed.

```jldoctest
source = Float64[1 2; 3 4; 5 6]
destination = similar(source)
rows_permutation = [3, 1, 2]
permute_dense_matrix_rows!(; destination, source, rows_permutation)
@assert destination == [5.0 6.0; 1.0 2.0; 3.0 4.0]

# output

```
"""
function permute_dense_matrix_rows!(;
    destination::AbstractMatrix,
    source::AbstractMatrix,
    rows_permutation::AbstractVector{<:Integer},
    progress::Maybe{Progress} = nothing,
)::Nothing
    n_rows, n_columns = size(source)
    @assert size(destination) == (n_rows, n_columns)
    @assert length(rows_permutation) == n_rows

    parallel_loop_wo_rng(1:n_columns; name = "permute_dense_matrix_rows!") do column_index
        @inbounds for destination_row in 1:n_rows
            destination[destination_row, column_index] = source[rows_permutation[destination_row], column_index]
        end
        if progress !== nothing
            next!(progress; step = n_rows)  # UNTESTED
        end
        return nothing
    end
    return nothing
end

"""
    permute_dense_matrix_both!(;
        destination::AbstractMatrix,
        source::AbstractMatrix,
        rows_permutation::AbstractVector{<:Integer},
        columns_permutation::AbstractVector{<:Integer},
        progress::Maybe{Progress} = nothing,
    )::Nothing

Fill `destination` such that `destination[destination_row, destination_column] ==
source[rows_permutation[destination_row], columns_permutation[destination_column]]` for every `(destination_row,
destination_column)`. Single-pass; parallelized per destination column. If `progress` is given, advance it by `n_rows`
per destination column completed.

```jldoctest
source = Float64[1 2 3; 4 5 6]
destination = similar(source)
rows_permutation = [2, 1]
columns_permutation = [3, 1, 2]
permute_dense_matrix_both!(; destination, source, rows_permutation, columns_permutation)
@assert destination == [6.0 4.0 5.0; 3.0 1.0 2.0]

# output

```
"""
function permute_dense_matrix_both!(;
    destination::AbstractMatrix,
    source::AbstractMatrix,
    rows_permutation::AbstractVector{<:Integer},
    columns_permutation::AbstractVector{<:Integer},
    progress::Maybe{Progress} = nothing,
)::Nothing
    n_rows, n_columns = size(source)
    @assert size(destination) == (n_rows, n_columns)
    @assert length(rows_permutation) == n_rows
    @assert length(columns_permutation) == n_columns

    parallel_loop_wo_rng(1:n_columns; name = "permute_dense_matrix_both!") do destination_column
        source_column = columns_permutation[destination_column]
        @inbounds for destination_row in 1:n_rows
            destination[destination_row, destination_column] = source[rows_permutation[destination_row], source_column]
        end
        if progress !== nothing
            next!(progress; step = n_rows)  # UNTESTED
        end
        return nothing
    end
    return nothing
end

"""
    permute_sparse_matrix_columns!(;
        destination_colptr::AbstractVector{<:Integer},
        destination_rowval::AbstractVector{<:Integer},
        destination_nzval::AbstractVector,
        source::SparseMatrixCSC,
        columns_permutation::AbstractVector{<:Integer},
        progress::Maybe{Progress} = nothing,
    )::Nothing

Fill the `(colptr, rowval, nzval)` backing triplet of a sparse destination such that `destination[:, destination_column]
== source[:, columns_permutation[destination_column]]` for every `destination_column`. Parallelized per destination
column. If `progress` is given, advance it by the column nnz per destination column completed.

```jldoctest
using SparseArrays

source = sparse([1.0 0.0 2.0; 0.0 3.0 0.0; 4.0 5.0 0.0])
destination_colptr = Vector{Int}(undef, size(source, 2) + 1)
destination_rowval = Vector{Int}(undef, nnz(source))
destination_nzval = Vector{Float64}(undef, nnz(source))
columns_permutation = [3, 1, 2]
permute_sparse_matrix_columns!(;
    destination_colptr,
    destination_rowval,
    destination_nzval,
    source,
    columns_permutation,
)
destination = SparseMatrixCSC(
    size(source, 1),
    size(source, 2),
    destination_colptr,
    destination_rowval,
    destination_nzval,
)
@assert destination == sparse([2.0 1.0 0.0; 0.0 0.0 3.0; 0.0 4.0 5.0])

# output

```
"""
function permute_sparse_matrix_columns!(;
    destination_colptr::AbstractVector{<:Integer},
    destination_rowval::AbstractVector{<:Integer},
    destination_nzval::AbstractVector,
    source::SparseMatrixCSC,
    columns_permutation::AbstractVector{<:Integer},
    progress::Maybe{Progress} = nothing,
)::Nothing
    n_columns = size(source, 2)
    source_nnz = nnz(source)
    @assert length(destination_colptr) == n_columns + 1
    @assert length(destination_rowval) == source_nnz
    @assert length(destination_nzval) == source_nnz
    @assert length(columns_permutation) == n_columns

    source_colptr = source.colptr
    source_rowval = source.rowval
    source_nzval = source.nzval

    destination_colptr[1] = 1
    @inbounds for destination_column in 1:n_columns
        source_column = columns_permutation[destination_column]
        column_length = source_colptr[source_column + 1] - source_colptr[source_column]
        destination_colptr[destination_column + 1] = destination_colptr[destination_column] + column_length
    end

    parallel_loop_wo_rng(1:n_columns; name = "permute_sparse_matrix_columns!") do destination_column
        source_column = columns_permutation[destination_column]
        source_start = source_colptr[source_column]
        source_stop = source_colptr[source_column + 1] - 1
        destination_start = destination_colptr[destination_column]
        column_nnz = source_stop - source_start + 1
        @inbounds for column_offset in 0:(column_nnz - 1)
            destination_rowval[destination_start + column_offset] = source_rowval[source_start + column_offset]
            destination_nzval[destination_start + column_offset] = source_nzval[source_start + column_offset]
        end
        if progress !== nothing
            next!(progress; step = column_nnz)  # UNTESTED
        end
        return nothing
    end
    return nothing
end

"""
    permute_sparse_matrix_rows!(;
        destination_colptr::AbstractVector{<:Integer},
        destination_rowval::AbstractVector{<:Integer},
        destination_nzval::AbstractVector,
        source::SparseMatrixCSC,
        inverse_rows_permutation::AbstractVector{<:Integer},
        progress::Maybe{Progress} = nothing,
    )::Nothing

Fill the `(colptr, rowval, nzval)` backing triplet of a sparse destination such that
`destination[inverse_rows_permutation[source_row], :] == source[source_row, :]` for every `source_row`. The destination
has the same `colptr` as the source. Parallelized per destination column. If `progress` is given, advance it by the
column nnz per column completed.

```jldoctest
using SparseArrays

source = sparse([1.0 2.0; 3.0 0.0; 0.0 4.0])
destination_colptr = Vector{Int}(undef, size(source, 2) + 1)
destination_rowval = Vector{Int}(undef, nnz(source))
destination_nzval = Vector{Float64}(undef, nnz(source))
inverse_rows_permutation = [2, 3, 1]
permute_sparse_matrix_rows!(;
    destination_colptr,
    destination_rowval,
    destination_nzval,
    source,
    inverse_rows_permutation,
)
destination = SparseMatrixCSC(
    size(source, 1),
    size(source, 2),
    destination_colptr,
    destination_rowval,
    destination_nzval,
)
@assert destination == sparse([0.0 4.0; 1.0 2.0; 3.0 0.0])

# output

```
"""
function permute_sparse_matrix_rows!(;
    destination_colptr::AbstractVector{<:Integer},
    destination_rowval::AbstractVector{<:Integer},
    destination_nzval::AbstractVector,
    source::SparseMatrixCSC,
    inverse_rows_permutation::AbstractVector{<:Integer},
    progress::Maybe{Progress} = nothing,
)::Nothing
    n_rows, n_columns = size(source)
    source_nnz = nnz(source)
    @assert length(destination_colptr) == n_columns + 1
    @assert length(destination_rowval) == source_nnz
    @assert length(destination_nzval) == source_nnz
    @assert length(inverse_rows_permutation) == n_rows

    source_colptr = source.colptr
    source_rowval = source.rowval
    source_nzval = source.nzval

    @inbounds for colptr_index in 1:(n_columns + 1)
        destination_colptr[colptr_index] = source_colptr[colptr_index]
    end

    parallel_loop_wo_rng(1:n_columns; name = "permute_sparse_matrix_rows!") do column_index
        source_start = source_colptr[column_index]
        source_stop = source_colptr[column_index + 1] - 1
        column_nnz = source_stop - source_start + 1
        if column_nnz > 0
            permuted_rowval = Vector{Int}(undef, column_nnz)
            @inbounds for column_offset in 0:(column_nnz - 1)
                permuted_rowval[column_offset + 1] =
                    inverse_rows_permutation[source_rowval[source_start + column_offset]]
            end
            sort_order = sortperm(permuted_rowval)
            destination_start = destination_colptr[column_index]
            @inbounds for column_offset in 0:(column_nnz - 1)
                ordered_scan_index = sort_order[column_offset + 1]
                destination_rowval[destination_start + column_offset] = permuted_rowval[ordered_scan_index]
                destination_nzval[destination_start + column_offset] =
                    source_nzval[source_start + ordered_scan_index - 1]
            end
        end
        if progress !== nothing
            next!(progress; step = column_nnz)  # UNTESTED
        end
        return nothing
    end
    return nothing
end

"""
    permute_sparse_matrix_both!(;
        destination_colptr::AbstractVector{<:Integer},
        destination_rowval::AbstractVector{<:Integer},
        destination_nzval::AbstractVector,
        source::SparseMatrixCSC,
        inverse_rows_permutation::AbstractVector{<:Integer},
        columns_permutation::AbstractVector{<:Integer},
        progress::Maybe{Progress} = nothing,
    )::Nothing

Fill the `(colptr, rowval, nzval)` backing triplet of a sparse destination such that
`destination[inverse_rows_permutation[source_row], destination_column] == source[source_row,
columns_permutation[destination_column]]` for every `(source_row, destination_column)`. Single-pass; parallelized per
destination column. If `progress` is given, advance it by the column nnz per destination column completed.

```jldoctest
using SparseArrays

source = sparse([1.0 2.0 0.0; 0.0 3.0 4.0])
destination_colptr = Vector{Int}(undef, size(source, 2) + 1)
destination_rowval = Vector{Int}(undef, nnz(source))
destination_nzval = Vector{Float64}(undef, nnz(source))
inverse_rows_permutation = [2, 1]
columns_permutation = [3, 1, 2]
permute_sparse_matrix_both!(;
    destination_colptr,
    destination_rowval,
    destination_nzval,
    source,
    inverse_rows_permutation,
    columns_permutation,
)
destination = SparseMatrixCSC(
    size(source, 1),
    size(source, 2),
    destination_colptr,
    destination_rowval,
    destination_nzval,
)
@assert destination == sparse([4.0 0.0 3.0; 0.0 1.0 2.0])

# output

```
"""
function permute_sparse_matrix_both!(;
    destination_colptr::AbstractVector{<:Integer},
    destination_rowval::AbstractVector{<:Integer},
    destination_nzval::AbstractVector,
    source::SparseMatrixCSC,
    inverse_rows_permutation::AbstractVector{<:Integer},
    columns_permutation::AbstractVector{<:Integer},
    progress::Maybe{Progress} = nothing,
)::Nothing
    n_rows, n_columns = size(source)
    source_nnz = nnz(source)
    @assert length(destination_colptr) == n_columns + 1
    @assert length(destination_rowval) == source_nnz
    @assert length(destination_nzval) == source_nnz
    @assert length(inverse_rows_permutation) == n_rows
    @assert length(columns_permutation) == n_columns

    source_colptr = source.colptr
    source_rowval = source.rowval
    source_nzval = source.nzval

    destination_colptr[1] = 1
    @inbounds for destination_column in 1:n_columns
        source_column = columns_permutation[destination_column]
        column_length = source_colptr[source_column + 1] - source_colptr[source_column]
        destination_colptr[destination_column + 1] = destination_colptr[destination_column] + column_length
    end

    parallel_loop_wo_rng(1:n_columns; name = "permute_sparse_matrix_both!") do destination_column
        source_column = columns_permutation[destination_column]
        source_start = source_colptr[source_column]
        source_stop = source_colptr[source_column + 1] - 1
        column_nnz = source_stop - source_start + 1
        if column_nnz > 0
            permuted_rowval = Vector{Int}(undef, column_nnz)
            @inbounds for column_offset in 0:(column_nnz - 1)
                permuted_rowval[column_offset + 1] =
                    inverse_rows_permutation[source_rowval[source_start + column_offset]]
            end
            sort_order = sortperm(permuted_rowval)
            destination_start = destination_colptr[destination_column]
            @inbounds for column_offset in 0:(column_nnz - 1)
                ordered_scan_index = sort_order[column_offset + 1]
                destination_rowval[destination_start + column_offset] = permuted_rowval[ordered_scan_index]
                destination_nzval[destination_start + column_offset] =
                    source_nzval[source_start + ordered_scan_index - 1]
            end
        end
        if progress !== nothing
            next!(progress; step = column_nnz)  # UNTESTED
        end
        return nothing
    end
    return nothing
end

end  # module
