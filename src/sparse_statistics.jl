"""
Median, quantile, variance and standard-deviation reductions optimized for sparse vectors and matrices.

A quantile of a sparse vector with `nnz` non-zero values out of `n` total entries can be located in `O(nnz)` expected
time instead of the dense `O(n * log(n))` time spent on a full sort. The key observation is that the values, if
conceptually sorted, lay out as `[sorted negatives... | zeros... | sorted positives...]`, where any explicit zero entries
that may be stored in `nzval` are folded together with the structural zeros. Once we know how many negatives and how
many positives are stored in `nzval`, mapping a quantile position to a value just requires running an in-place
quickselect on (a copy of) `nzval` to extract the one or two values needed.
The implementation is allocation-free when the caller provides a `scratch` buffer of the appropriate size.

The variance is computed in `O(nnz)` by running Welford's algorithm over `nzval` and merging the resulting stream stats
`(n, mean, M2)` with the all-zeros stream `(n_zero, 0, 0)` via the standard pairwise variance combiner; this is exact
and numerically stable, with no scratch buffer needed. The standard deviation is just the square root of the variance.

If the data is known to be non-negative (e.g., UMI counts), passing `positive = true` to the median/quantile functions
skips the negativity scan entirely and yields a small additional speedup.

The functions exposed here mirror the calling convention of `Statistics.var` and `Statistics.std`: they accept either an
`AbstractVector` (returning a single value) or an `AbstractMatrix` together with a `dims` keyword argument (returning a
vector with one of the dimensions reduced to a single entry, or a scalar when `dims` is omitted).

The same `[negatives... | zeros... | positives...]` layout also speeds up the one-vs-rest AUC (area under the ROC curve)
of a feature across labeled samples: only the stored non-zero values need to be ranked, the many zero samples sharing a
single tied rank. See [`auc_per_group!`](@ref) and [`auc_per_category`](@ref).
"""
module SparseStatistics

export auc_per_category
export auc_per_group!
export sparse_median
export sparse_quantile
export sparse_std
export sparse_var

using LinearAlgebra
using ProgressMeter

import ProgressMeter.AbstractProgress  # NOLINT
using SparseArrays
using StatsBase

using ..Documentation
using ..FlameTime
using ..MatrixFormats
using ..MatrixLayouts
using ..ParallelLoops
using ..Types

import ..MatrixLayouts.check_efficient_action

"""
    sparse_quantile(
        vector::AbstractVector{<:Real},
        p::Real;
        positive::Bool = false,
        scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    )::Float64

    sparse_quantile(
        matrix::AbstractMatrix{<:Real},
        p::Real;
        dims::Maybe{Integer} = nothing,
        positive::Bool = false,
        result::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
        scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
        progress::Maybe{AbstractProgress} = nothing,
        progress_chunk::Maybe{Integer} = nothing,
    )::Union{Float64, AbstractVector{<:AbstractFloat}}

Compute the `p`-quantile (`0 <= p <= 1`) of the values of a `vector`, or of each column (`dims = 1` / [`Rows`](@ref))
or each row (`dims = 2` / [`Columns`](@ref)) of a `matrix`, or of all the elements of a `matrix` when `dims` is omitted.
With `dims` given, the matrix variant returns a vector of length `n_columns` (`dims = Rows`) or `n_rows`
(`dims = Columns`), holding one quantile value per slice; without `dims`, it returns a scalar `Float64`.

The quantile is computed via the same linear interpolation rule as `Statistics.quantile` (corresponding to `R`'s default
"type 7"), so the result matches `quantile(Vector(input), p)` exactly.
For a sparse `vector` (or each sparse column slice of a column-major sparse `matrix`, or each sparse row slice of a
row-major sparse `matrix`), the quantile is located in `O(nnz)` expected time using an in-place quickselect over a copy
of the stored non-zero values.

If `positive` is `true`, all input values are assumed to be non-negative; this lets the implementation skip the scan
that locates the boundary between negative and non-negative values. No validation is performed on this assumption.

If `scratch` is given, no allocation is performed. The required `scratch` length is `nnz(input)` for sparse data and
`length(input)` for dense vectors (or `n_rows * n_columns` for dense matrices). If `result` is given for the matrix
variant (length `n_columns` for `dims = Rows`, or `n_rows` for `dims = Columns`), no result allocation is performed
either.

When operating on a `matrix`, the iteration is parallelized via [`parallel_loop_wo_rng`](@ref). Operating against the
[`major_axis`](@ref) of a sparse `matrix` will trigger the [`GLOBAL_INEFFICIENT_ACTION_HANDLER`](@ref) and fall back to
allocating per-iteration views.

If `progress` is given (matrix variant only), it is advanced once per processed slice; that is, by `n_columns` ticks
when `dims = Rows`, and by `n_rows` ticks when `dims = Columns`. The `progress_chunk` is passed through to
[`parallel_loop_wo_rng`](@ref) to throttle the rate of `progress` updates.

```jldoctest
println(sparse_quantile([0, 0, 1, 2, 3], 0.5))
println(sparse_quantile([0, 0, 1, 2, 3], 0.5; positive = true))
println(sparse_quantile([-2, 0, 0, 1, 3], 0.25))

# output

1.0
1.0
0.0
```

```jldoctest
using SparseArrays

matrix = sparse([0 1 0; 2 0 3; 0 4 0; 5 0 6])
sparse_quantile(matrix, 0.5; dims = Rows)

# output

3-element Vector{Float64}:
 1.0
 0.5
 1.5
```

```jldoctest
sparse_quantile([1.0 4.0; 2.0 5.0; 3.0 6.0], 0.5; dims = Rows)

# output

2-element Vector{Float64}:
 2.0
 5.0
```
"""
function sparse_quantile(
    vector::AbstractVector{<:Real},
    p::Real;
    positive::Bool = false,
    scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
)::Float64
    @assert 0 <= p <= 1 "quantile p: $(p) is not in [0, 1]"
    return flame_timed("sparse_quantile") do
        return compute_sparse_quantile_of_vector(vector, p; positive, scratch)
    end
end

function sparse_quantile(
    matrix::AbstractMatrix{<:Real},
    p::Real;
    dims::Maybe{Integer} = nothing,
    positive::Bool = false,
    result::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    progress::Maybe{AbstractProgress} = nothing,
    progress_chunk::Maybe{Integer} = nothing,
)::Union{Float64, AbstractVector{<:AbstractFloat}}
    @assert 0 <= p <= 1 "quantile p: $(p) is not in [0, 1]"
    if dims === nothing
        @assert result === nothing "result is only valid with dims"
        @assert progress === nothing "progress is only valid with dims"
        @assert progress_chunk === nothing "progress_chunk is only valid with dims"
        return flame_timed("sparse_quantile") do
            return compute_sparse_quantile_of_flat_matrix(matrix, p; positive, scratch)
        end
    end
    @assert dims == Rows || dims == Columns "invalid dims: $(dims)"
    return flame_timed("sparse_quantile") do
        return compute_sparse_quantile_of_matrix(matrix, p; dims, positive, result, scratch, progress, progress_chunk)
    end
end

"""
    sparse_median(
        vector::AbstractVector{<:Real};
        positive::Bool = false,
        scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    )::Float64

    sparse_median(
        matrix::AbstractMatrix{<:Real};
        dims::Maybe{Integer} = nothing,
        positive::Bool = false,
        result::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
        scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
        progress::Maybe{AbstractProgress} = nothing,
        progress_chunk::Maybe{Integer} = nothing,
    )::Union{Float64, AbstractVector{<:AbstractFloat}}

Compute the median of the values of a `vector`, or of each column (`dims = 1` / [`Rows`](@ref)) or each row
(`dims = 2` / [`Columns`](@ref)) of a `matrix`, or of all the elements of a `matrix` when `dims` is omitted. This is
equivalent to [`sparse_quantile`](@ref) with `p = 0.5` and shares its calling convention, optimizations, `positive`
flag, optional `result`/`scratch` buffers, and `progress` / `progress_chunk` reporting.

```jldoctest
using SparseArrays

vector = sparse([0, 0, 1, 2, 3])
println(sparse_median(vector))
println(sparse_median(vector; positive = true))

# output

1.0
1.0
```

```jldoctest
using SparseArrays

matrix = sparse([0 1 0; 2 0 3; 0 4 0; 5 0 6])
sparse_median(matrix; dims = Rows)

# output

3-element Vector{Float64}:
 1.0
 0.5
 1.5
```
"""
function sparse_median(
    vector::AbstractVector{<:Real};
    positive::Bool = false,
    scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
)::Float64
    return flame_timed("sparse_median") do
        return compute_sparse_quantile_of_vector(vector, 0.5; positive, scratch)
    end
end

function sparse_median(
    matrix::AbstractMatrix{<:Real};
    dims::Maybe{Integer} = nothing,
    positive::Bool = false,
    result::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    progress::Maybe{AbstractProgress} = nothing,
    progress_chunk::Maybe{Integer} = nothing,
)::Union{Float64, AbstractVector{<:AbstractFloat}}
    if dims === nothing
        @assert result === nothing "result is only valid with dims"
        @assert progress === nothing "progress is only valid with dims"
        @assert progress_chunk === nothing "progress_chunk is only valid with dims"
        return flame_timed("sparse_median") do
            return compute_sparse_quantile_of_flat_matrix(matrix, 0.5; positive, scratch)
        end
    end
    @assert dims == Rows || dims == Columns "invalid dims: $(dims)"
    return flame_timed("sparse_median") do
        return compute_sparse_quantile_of_matrix(matrix, 0.5; dims, positive, result, scratch, progress, progress_chunk)
    end
end

"""
    sparse_var(
        vector::AbstractVector{<:Real};
        corrected::Bool = true,
    )::Float64

    sparse_var(
        matrix::AbstractMatrix{<:Real};
        dims::Maybe{Integer} = nothing,
        corrected::Bool = true,
        result::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
        progress::Maybe{AbstractProgress} = nothing,
        progress_chunk::Maybe{Integer} = nothing,
    )::Union{Float64, AbstractVector{<:AbstractFloat}}

Compute the variance of the values of a `vector`, or of each column (`dims = 1` / [`Rows`](@ref)) or each row
(`dims = 2` / [`Columns`](@ref)) of a `matrix`, or of all the elements of a `matrix` when `dims` is omitted. The shape
conventions match `Statistics.var`: with `dims` the matrix variant returns a vector of length `n_columns` (`dims = Rows`)
or `n_rows` (`dims = Columns`); without `dims` it returns a scalar `Float64`.

If `corrected` is `true` (the default), the bias-corrected sample variance is returned (sum of squared deviations
divided by `n - 1`); otherwise the population variance is returned (divided by `n`).

The implementation runs Welford's algorithm over the stored non-zero values and merges the result with the all-zeros
stream `(n_zero, 0, 0)` using the standard pairwise variance combiner. This is `O(nnz)` for sparse inputs, numerically
stable, and allocation-free apart from the result vector. No `scratch` buffer is needed.

When operating on a `matrix`, the iteration is parallelized via [`parallel_loop_wo_rng`](@ref). Operating against the
[`major_axis`](@ref) of a sparse `matrix` will trigger the [`GLOBAL_INEFFICIENT_ACTION_HANDLER`](@ref).

If `progress` is given (matrix variant only), it is advanced once per processed slice (`n_columns` ticks for
`dims = Rows`, `n_rows` for `dims = Columns`). The `progress_chunk` is passed through to [`parallel_loop_wo_rng`](@ref).

```jldoctest
println(sparse_var([1.0, 2.0, 3.0, 4.0]))
println(sparse_var([1.0, 2.0, 3.0, 4.0]; corrected = false))

# output

1.6666666666666667
1.25
```

```jldoctest
using SparseArrays

matrix = sparse([0 1 0; 2 0 3; 0 4 0; 5 0 6])
sparse_var(matrix; dims = Rows)

# output

3-element Vector{Float64}:
 5.583333333333333
 3.5833333333333335
 8.25
```
"""
function sparse_var(vector::AbstractVector{<:Real}; corrected::Bool = true)::Float64
    return flame_timed("sparse_var") do
        return compute_sparse_var_of_vector(vector, corrected)
    end
end

function sparse_var(
    matrix::AbstractMatrix{<:Real};
    dims::Maybe{Integer} = nothing,
    corrected::Bool = true,
    result::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    progress::Maybe{AbstractProgress} = nothing,
    progress_chunk::Maybe{Integer} = nothing,
)::Union{Float64, AbstractVector{<:AbstractFloat}}
    if dims === nothing
        @assert result === nothing "result is only valid with dims"
        @assert progress === nothing "progress is only valid with dims"
        @assert progress_chunk === nothing "progress_chunk is only valid with dims"
        return flame_timed("sparse_var") do
            return compute_sparse_var_of_flat_matrix(matrix, corrected)
        end
    end
    @assert dims == Rows || dims == Columns "invalid dims: $(dims)"
    return flame_timed("sparse_var") do
        return compute_sparse_var_of_matrix(matrix, corrected; dims, result, progress, progress_chunk)
    end
end

"""
    sparse_std(
        vector::AbstractVector{<:Real};
        corrected::Bool = true,
    )::Float64

    sparse_std(
        matrix::AbstractMatrix{<:Real};
        dims::Maybe{Integer} = nothing,
        corrected::Bool = true,
        result::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
        progress::Maybe{AbstractProgress} = nothing,
        progress_chunk::Maybe{Integer} = nothing,
    )::Union{Float64, AbstractVector{<:AbstractFloat}}

Compute the standard deviation of the values; equivalent to taking `sqrt` of [`sparse_var`](@ref). Shares the
calling convention, `corrected` flag, and optional `result` / `progress` / `progress_chunk` parameters of
[`sparse_var`](@ref).

```jldoctest
using SparseArrays

matrix = sparse([0 1 0; 2 0 3; 0 4 0; 5 0 6])
sparse_std(matrix; dims = Rows)

# output

3-element Vector{Float64}:
 2.362907813126304
 1.8929694486000912
 2.8722813232690143
```
"""
function sparse_std(vector::AbstractVector{<:Real}; corrected::Bool = true)::Float64
    return flame_timed("sparse_std") do
        return sqrt(compute_sparse_var_of_vector(vector, corrected))
    end
end

function sparse_std(
    matrix::AbstractMatrix{<:Real};
    dims::Maybe{Integer} = nothing,
    corrected::Bool = true,
    result::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    progress::Maybe{AbstractProgress} = nothing,
    progress_chunk::Maybe{Integer} = nothing,
)::Union{Float64, AbstractVector{<:AbstractFloat}}
    if dims === nothing
        @assert result === nothing "result is only valid with dims"
        @assert progress === nothing "progress is only valid with dims"
        @assert progress_chunk === nothing "progress_chunk is only valid with dims"
        return flame_timed("sparse_std") do
            return sqrt(compute_sparse_var_of_flat_matrix(matrix, corrected))
        end
    end
    @assert dims == Rows || dims == Columns "invalid dims: $(dims)"
    return flame_timed("sparse_std") do
        variances = compute_sparse_var_of_matrix(matrix, corrected; dims, result, progress, progress_chunk)
        variances .= sqrt.(variances)
        return variances
    end
end

function compute_sparse_var_of_vector(vector::AbstractVector{<:Real}, corrected::Bool)::Float64
    if issparse(vector)
        return welford_variance(nzval(vector), length(vector), corrected)
    else
        return welford_variance(vector, length(vector), corrected)
    end
end

function compute_sparse_var_of_flat_matrix(matrix::AbstractMatrix{<:Real}, corrected::Bool)::Float64
    if issparse(matrix)
        return welford_variance(nzval(matrix), length(matrix), corrected)
    else
        return welford_variance(vec(matrix), length(matrix), corrected)
    end
end

function compute_sparse_var_of_matrix(
    matrix::AbstractMatrix{<:Real},
    corrected::Bool;
    dims::Integer,
    result::Maybe{AbstractVector{<:AbstractFloat}},
    progress::Maybe{AbstractProgress},
    progress_chunk::Maybe{Integer},
)::AbstractVector{<:AbstractFloat}
    n_rows, n_columns = size(matrix)
    process_axis = other_axis(dims)
    if major_axis(matrix) !== nothing
        check_efficient_action(@source_location()..., "matrix", matrix, process_axis)
    end

    result_length = dims == Rows ? n_columns : n_rows
    if result === nothing
        result = Vector{Float64}(undef, result_length)
    else
        @assert length(result) == result_length "result length: $(length(result)) is not the expected: $(result_length)"
    end

    if issparse(matrix) && major_axis(matrix) == process_axis
        compute_sparse_var_of_sparse_matrix!(result, matrix, corrected; dims, progress, progress_chunk)
    else
        compute_sparse_var_of_dense_matrix!(result, matrix, corrected; dims, progress, progress_chunk)
    end

    return result
end

function compute_sparse_var_of_sparse_matrix!(
    result::AbstractVector{<:AbstractFloat},
    matrix::AbstractMatrix{<:Real},
    corrected::Bool;
    dims::Integer,
    progress::Maybe{AbstractProgress},
    progress_chunk::Maybe{Integer},
)::Nothing
    n_rows, n_columns = size(matrix)
    if dims == Rows
        column_major_matrix = matrix
        n_iterations = n_columns
        n_total_per_iteration = n_rows
    else
        column_major_matrix = flip(matrix)
        n_iterations = n_rows
        n_total_per_iteration = n_columns
    end

    column_offsets = colptr(column_major_matrix)
    nonzero_values = nzval(column_major_matrix)

    parallel_loop_wo_rng(1:n_iterations; name = "sparse_var", progress, progress_chunk) do iteration_index
        slice_first = Int(column_offsets[iteration_index])
        slice_last = Int(column_offsets[iteration_index + 1]) - 1
        @views slice = nonzero_values[slice_first:slice_last]
        result[iteration_index] = welford_variance(slice, n_total_per_iteration, corrected)
        return nothing
    end

    return nothing
end

function compute_sparse_var_of_dense_matrix!(
    result::AbstractVector{<:AbstractFloat},
    matrix::AbstractMatrix{<:Real},
    corrected::Bool;
    dims::Integer,
    progress::Maybe{AbstractProgress},
    progress_chunk::Maybe{Integer},
)::Nothing
    n_rows, n_columns = size(matrix)
    if dims == Rows
        parallel_loop_wo_rng(1:n_columns; name = "sparse_var", progress, progress_chunk) do column_index
            @views slice = matrix[:, column_index]
            result[column_index] = welford_variance(slice, n_rows, corrected)
            return nothing
        end
    else
        parallel_loop_wo_rng(1:n_rows; name = "sparse_var", progress, progress_chunk) do row_index
            @views slice = matrix[row_index, :]
            result[row_index] = welford_variance(slice, n_columns, corrected)
            return nothing
        end
    end
    return nothing
end

function welford_variance(values, n_total::Integer, corrected::Bool)::Float64
    n_a = 0
    mean_a = 0.0
    m2_a = 0.0
    @inbounds for value in values
        x = Float64(value)
        n_a += 1
        delta = x - mean_a
        mean_a += delta / n_a
        m2_a += delta * (x - mean_a)
    end
    n_zero = n_total - n_a
    @assert n_zero >= 0
    n = n_total

    # Pairwise merge of stream A = `(n_a, mean_a, m2_a)` over the stored values with stream B
    # `(n_zero, 0, 0)` (the implicit zeros). The Chan/Golub/LeVeque combiner gives:
    #   M2 = M2_A + M2_B + (mean_B - mean_A)^2 * n_A * n_B / n
    # Here `mean_B = 0` and `M2_B = 0`, so this collapses to `m2_a + mean_a^2 * n_a * n_zero / n`.
    if n_a > 0 && n_zero > 0
        m2 = m2_a + mean_a^2 * n_a * n_zero / n
    else
        m2 = m2_a
    end

    if corrected
        return n < 2 ? NaN : m2 / (n - 1)
    else
        return n < 1 ? NaN : m2 / n
    end
end

function compute_sparse_quantile_of_vector(
    vector::AbstractVector{<:Real},
    p::Real;
    positive::Bool,
    scratch::Maybe{AbstractVector{<:AbstractFloat}},
)::Float64
    if issparse(vector)
        nonzero_values = nzval(vector)
        n_nonzero = length(nonzero_values)
        n_implicit_zero = length(vector) - n_nonzero
    else
        nonzero_values = vector
        n_nonzero = length(vector)
        n_implicit_zero = 0
    end

    if scratch === nothing
        scratch = Vector{Float64}(undef, n_nonzero)
    else
        @assert length(scratch) >= n_nonzero "scratch length: $(length(scratch)) is below required: $(n_nonzero)"
    end

    @views scratch[1:n_nonzero] .= nonzero_values

    return sparse_quantile_from_buffer!(scratch, n_nonzero, n_implicit_zero, p, positive)
end

function compute_sparse_quantile_of_flat_matrix(
    matrix::AbstractMatrix{<:Real},
    p::Real;
    positive::Bool,
    scratch::Maybe{AbstractVector{<:AbstractFloat}},
)::Float64
    if issparse(matrix)
        nonzero_values = nzval(matrix)
        n_nonzero = length(nonzero_values)
        n_implicit_zero = length(matrix) - n_nonzero
    else
        nonzero_values = vec(matrix)
        n_nonzero = length(nonzero_values)
        n_implicit_zero = 0
    end

    if scratch === nothing
        scratch = Vector{Float64}(undef, n_nonzero)
    else
        @assert length(scratch) >= n_nonzero "scratch length: $(length(scratch)) is below required: $(n_nonzero)"
    end

    @views scratch[1:n_nonzero] .= nonzero_values

    return sparse_quantile_from_buffer!(scratch, n_nonzero, n_implicit_zero, p, positive)
end

function compute_sparse_quantile_of_matrix(
    matrix::AbstractMatrix{<:Real},
    p::Real;
    dims::Integer,
    positive::Bool,
    result::Maybe{AbstractVector{<:AbstractFloat}},
    scratch::Maybe{AbstractVector{<:AbstractFloat}},
    progress::Maybe{AbstractProgress},
    progress_chunk::Maybe{Integer},
)::AbstractVector{<:AbstractFloat}
    n_rows, n_columns = size(matrix)
    process_axis = other_axis(dims)
    if major_axis(matrix) !== nothing
        check_efficient_action(@source_location()..., "matrix", matrix, process_axis)
    end

    result_length = dims == Rows ? n_columns : n_rows
    if result === nothing
        result = Vector{Float64}(undef, result_length)
    else
        @assert length(result) == result_length "result length: $(length(result)) is not the expected: $(result_length)"
    end

    if issparse(matrix) && major_axis(matrix) == process_axis
        compute_sparse_quantile_of_sparse_matrix!(result, matrix, p; dims, positive, scratch, progress, progress_chunk)
    else
        compute_sparse_quantile_of_dense_matrix!(result, matrix, p; dims, positive, scratch, progress, progress_chunk)
    end

    return result
end

function compute_sparse_quantile_of_sparse_matrix!(
    result::AbstractVector{<:AbstractFloat},
    matrix::AbstractMatrix{<:Real},
    p::Real;
    dims::Integer,
    positive::Bool,
    scratch::Maybe{AbstractVector{<:AbstractFloat}},
    progress::Maybe{AbstractProgress},
    progress_chunk::Maybe{Integer},
)::Nothing
    n_rows, n_columns = size(matrix)
    if dims == Rows
        column_major_matrix = matrix
        n_iterations = n_columns
        n_total_per_iteration = n_rows
    else
        column_major_matrix = flip(matrix)
        n_iterations = n_rows
        n_total_per_iteration = n_columns
    end

    column_offsets = colptr(column_major_matrix)
    nonzero_values = nzval(column_major_matrix)
    n_nonzero = length(nonzero_values)

    if scratch === nothing
        scratch = Vector{Float64}(undef, n_nonzero)
    else
        @assert length(scratch) >= n_nonzero "scratch length: $(length(scratch)) is below required: $(n_nonzero)"
    end

    @views scratch[1:n_nonzero] .= nonzero_values

    parallel_loop_wo_rng(1:n_iterations; name = "sparse_quantile", progress, progress_chunk) do iteration_index
        slice_first = Int(column_offsets[iteration_index])
        slice_last = Int(column_offsets[iteration_index + 1]) - 1
        slice_length = slice_last - slice_first + 1
        @views slice = scratch[slice_first:slice_last]
        n_implicit_zero = n_total_per_iteration - slice_length
        result[iteration_index] = sparse_quantile_from_buffer!(slice, slice_length, n_implicit_zero, p, positive)
        return nothing
    end

    return nothing
end

function compute_sparse_quantile_of_dense_matrix!(
    result::AbstractVector{<:AbstractFloat},
    matrix::AbstractMatrix{<:Real},
    p::Real;
    dims::Integer,
    positive::Bool,
    scratch::Maybe{AbstractVector{<:AbstractFloat}},
    progress::Maybe{AbstractProgress},
    progress_chunk::Maybe{Integer},
)::Nothing
    n_rows, n_columns = size(matrix)
    if dims == Rows
        n_iterations = n_columns
        slice_length = n_rows
    else
        n_iterations = n_rows
        slice_length = n_columns
    end

    n_required_scratch = slice_length * n_iterations
    if scratch === nothing
        scratch = Vector{Float64}(undef, n_required_scratch)
    else
        @assert length(scratch) >= n_required_scratch "scratch length: $(length(scratch)) is below required: $(n_required_scratch)"
    end

    @views scratch_matrix = reshape(scratch[1:n_required_scratch], slice_length, n_iterations)
    if dims == Rows
        copyto!(scratch_matrix, matrix)
    else
        transpose!(scratch_matrix, matrix)
    end

    parallel_loop_wo_rng(1:n_iterations; name = "sparse_quantile", progress, progress_chunk) do iteration_index
        slice_first = (iteration_index - 1) * slice_length + 1
        slice_last = iteration_index * slice_length
        @views slice = scratch[slice_first:slice_last]
        result[iteration_index] = sparse_quantile_from_buffer!(slice, slice_length, 0, p, positive)
        return nothing
    end

    return nothing
end

function sparse_quantile_from_buffer!(
    buffer::AbstractVector{<:AbstractFloat},
    n_in_buffer::Integer,
    n_implicit_zero::Integer,
    p::Real,
    positive::Bool,
)::Float64
    n_total = n_in_buffer + n_implicit_zero
    @assert n_total >= 1 "empty input"
    @assert n_in_buffer >= 0
    @assert n_implicit_zero >= 0

    if n_in_buffer == 0
        return 0.0
    end

    n_negative = 0
    n_positive = 0
    if positive
        @inbounds for index in 1:n_in_buffer
            if buffer[index] > 0
                n_positive += 1
            end
        end
    else
        @inbounds for index in 1:n_in_buffer
            value = buffer[index]
            if value < 0
                n_negative += 1
            elseif value > 0
                n_positive += 1
            end
        end
    end
    n_explicit_zero = n_in_buffer - n_negative - n_positive
    n_zero_total = n_implicit_zero + n_explicit_zero

    fractional_position = (n_total - 1) * Float64(p) + 1.0
    lower_position = floor(Int, fractional_position)

    if lower_position >= n_total
        return value_at_conceptual_position!(buffer, n_total, n_in_buffer, n_negative, n_zero_total, n_implicit_zero)
    end

    if lower_position == fractional_position
        return value_at_conceptual_position!(
            buffer,
            lower_position,
            n_in_buffer,
            n_negative,
            n_zero_total,
            n_implicit_zero,
        )
    end

    lower_value, upper_value = values_at_conceptual_positions!(
        buffer,
        lower_position,
        lower_position + 1,
        n_in_buffer,
        n_negative,
        n_zero_total,
        n_implicit_zero,
    )
    return lower_value + (fractional_position - lower_position) * (upper_value - lower_value)
end

@inline function buffer_index_for_conceptual_position(
    position::Integer,
    n_negative::Integer,
    n_zero_total::Integer,
    n_implicit_zero::Integer,
)::Int
    if position <= n_negative
        return Int(position)
    elseif position <= n_negative + n_zero_total
        return 0
    else
        return Int(position - n_implicit_zero)
    end
end

function value_at_conceptual_position!(
    buffer::AbstractVector{<:AbstractFloat},
    position::Integer,
    n_in_buffer::Integer,
    n_negative::Integer,
    n_zero_total::Integer,
    n_implicit_zero::Integer,
)::Float64
    buffer_index = buffer_index_for_conceptual_position(position, n_negative, n_zero_total, n_implicit_zero)
    if buffer_index == 0
        return 0.0
    end
    quickselect!(buffer, buffer_index, 1, n_in_buffer)
    return Float64(buffer[buffer_index])
end

function values_at_conceptual_positions!(
    buffer::AbstractVector{<:AbstractFloat},
    lower_position::Integer,
    upper_position::Integer,
    n_in_buffer::Integer,
    n_negative::Integer,
    n_zero_total::Integer,
    n_implicit_zero::Integer,
)::Tuple{Float64, Float64}
    lower_buffer_index = buffer_index_for_conceptual_position(lower_position, n_negative, n_zero_total, n_implicit_zero)
    upper_buffer_index = buffer_index_for_conceptual_position(upper_position, n_negative, n_zero_total, n_implicit_zero)

    if lower_buffer_index == 0 && upper_buffer_index == 0
        return (0.0, 0.0)
    end

    if lower_buffer_index == 0
        quickselect!(buffer, upper_buffer_index, 1, n_in_buffer)
        return (0.0, Float64(buffer[upper_buffer_index]))
    end

    if upper_buffer_index == 0
        quickselect!(buffer, lower_buffer_index, 1, n_in_buffer)
        return (Float64(buffer[lower_buffer_index]), 0.0)
    end

    quickselect!(buffer, lower_buffer_index, 1, n_in_buffer)
    if upper_buffer_index != lower_buffer_index
        quickselect!(buffer, upper_buffer_index, lower_buffer_index + 1, n_in_buffer)
    end
    return (Float64(buffer[lower_buffer_index]), Float64(buffer[upper_buffer_index]))
end

function quickselect!(
    buffer::AbstractVector{<:AbstractFloat},
    target_index::Integer,
    low::Integer,
    high::Integer,
)::Nothing
    @assert low <= target_index <= high
    while low < high
        if high - low < 16
            insertion_sort!(buffer, low, high)
            return nothing
        end

        middle = (low + high) >> 1
        if buffer[middle] < buffer[low]
            buffer[low], buffer[middle] = buffer[middle], buffer[low]
        end
        if buffer[high] < buffer[low]
            buffer[low], buffer[high] = buffer[high], buffer[low]
        end
        if buffer[high] < buffer[middle]
            buffer[middle], buffer[high] = buffer[high], buffer[middle]
        end
        pivot = buffer[middle]
        buffer[middle], buffer[high - 1] = buffer[high - 1], buffer[middle]

        low_index = low
        high_index = high - 1
        while true
            low_index += 1
            while buffer[low_index] < pivot
                low_index += 1
            end
            high_index -= 1
            while buffer[high_index] > pivot
                high_index -= 1
            end
            if low_index >= high_index
                break
            end
            buffer[low_index], buffer[high_index] = buffer[high_index], buffer[low_index]
        end
        buffer[low_index], buffer[high - 1] = buffer[high - 1], buffer[low_index]

        if target_index == low_index
            return nothing
        elseif target_index < low_index
            high = low_index - 1
        else
            low = low_index + 1
        end
    end
    return nothing
end

function insertion_sort!(buffer::AbstractVector{<:AbstractFloat}, low::Integer, high::Integer)::Nothing
    @inbounds for outer_index in (low + 1):high
        value = buffer[outer_index]
        inner_index = outer_index - 1
        while inner_index >= low && buffer[inner_index] > value
            buffer[inner_index + 1] = buffer[inner_index]
            inner_index -= 1
        end
        buffer[inner_index + 1] = value
    end
    return nothing
end

"""
    auc_per_group!(;
        auc_per_group::AbstractVector{Float64},
        feature_per_sample::AbstractVector{<:Real},
        group_index_per_sample::AbstractVector{<:Integer},
        n_samples_per_group::AbstractVector{<:Integer},
        positive::Bool = false,
    )::AbstractVector{Float64}

Fill `auc_per_group` with the one-vs-rest AUC of the `feature_per_sample`, distinguishing each group (the samples whose
`group_index_per_sample` holds that group's `1`-based index) from the rest of the included samples. Samples with a group
index of `0` are excluded from the population. `n_samples_per_group[group]` is the number of samples in each group,
supplied by the caller so that scoring many features against the same grouping computes it just once.

The AUC measures how predictive the feature is of membership in the group, regardless of direction: it is
`max(a, 1 - a)` of the Mann-Whitney statistic `a = (rank_sum - n * (n + 1) / 2) / (n * (N - n))` (tie-averaged ranks),
where `n` is the group size and `N` the number of included samples. It is thus always `>= 0.5`, so a group covering none
or all of the included samples - where the AUC is undefined - is reported as `0`. Being rank based, any monotonic
transform of the feature (e.g. a logarithm) yields the same result.

When `feature_per_sample` is sparse this costs `O(nnz * log(nnz))` rather than the dense `O(N * log(N))`: the many zero
samples share the single tied rank of the `[negatives... | zeros... | positives...]` layout, so only the stored non-zero
values are ranked. Pass `positive = true` when the feature is known to be non-negative (e.g. UMI counts) to skip the
negative-value handling.
"""
function auc_per_group!(;
    auc_per_group::AbstractVector{Float64},
    feature_per_sample::AbstractVector{<:Real},
    group_index_per_sample::AbstractVector{<:Integer},
    n_samples_per_group::AbstractVector{<:Integer},
    positive::Bool = false,
)::AbstractVector{Float64}
    n_groups = length(n_samples_per_group)
    @assert length(auc_per_group) == n_groups "auc_per_group length: $(length(auc_per_group)) is not the number of groups: $(n_groups)"
    @assert length(group_index_per_sample) == length(feature_per_sample) "group_index_per_sample length: $(length(group_index_per_sample)) is not the feature length: $(length(feature_per_sample))"
    return flame_timed("auc_per_group") do
        n_included_samples = sum(n_samples_per_group)
        rank_sum_per_group = zeros(Float64, n_groups)
        if issparse(feature_per_sample)
            sparse_rank_sum_per_group!(;
                rank_sum_per_group,
                feature_per_sample,
                group_index_per_sample,
                n_samples_per_group,
                n_included_samples,
                positive,
            )
        else
            dense_rank_sum_per_group!(; rank_sum_per_group, feature_per_sample, group_index_per_sample)
        end
        finalize_auc_per_group!(; auc_per_group, rank_sum_per_group, n_samples_per_group, n_included_samples)
        return auc_per_group
    end
end

"""
    auc_per_category(
        feature_per_sample::AbstractVector{<:Real},
        category_per_sample::AbstractVector;
        positive::Bool = false,
    )::Tuple{Vector, Vector{Float64}}

Compute the one-vs-rest AUC of the `feature_per_sample` for each distinct value of `category_per_sample` (an integer or
string label per sample). Returns the distinct categories (in order of first appearance) aligned with their AUCs. See
[`auc_per_group!`](@ref) for the definition and the sparse-feature optimization; use it directly to avoid re-deriving the
categories when scoring many features against the same categories.

```jldoctest
using SparseArrays

feature = sparse([0, 0, 0, 1, 2, 3])
category = [1, 2, 3, 1, 2, 3]
categories, aucs = auc_per_category(feature, category)
println(categories)
println(aucs)

# output

[1, 2, 3]
[0.625, 0.5, 0.625]
```
"""
function auc_per_category(
    feature_per_sample::AbstractVector{<:Real},
    category_per_sample::AbstractVector;
    positive::Bool = false,
)::Tuple{Vector, Vector{Float64}}
    @assert length(feature_per_sample) == length(category_per_sample) "feature length: $(length(feature_per_sample)) is not the category length: $(length(category_per_sample))"
    category_per_group = unique(category_per_sample)
    n_groups = length(category_per_group)
    group_index_per_category =
        Dict(category => group_index for (group_index, category) in enumerate(category_per_group))
    group_index_per_sample = [group_index_per_category[category] for category in category_per_sample]
    n_samples_per_group = zeros(Int, n_groups)
    for group_index in group_index_per_sample
        n_samples_per_group[group_index] += 1
    end
    auc_per_group = Vector{Float64}(undef, n_groups)
    auc_per_group!(; auc_per_group, feature_per_sample, group_index_per_sample, n_samples_per_group, positive)
    return (category_per_group, auc_per_group)
end

# Gather the included samples (group index above zero), rank the feature over them, and sum the ranks per group.
function dense_rank_sum_per_group!(;
    rank_sum_per_group::AbstractVector{Float64},
    feature_per_sample::AbstractVector{<:Real},
    group_index_per_sample::AbstractVector{<:Integer},
)::Nothing
    n_included = count(>(0), group_index_per_sample)
    feature_per_included = Vector{Float64}(undef, n_included)
    group_index_per_included = Vector{Int}(undef, n_included)
    included = 0
    for sample in eachindex(feature_per_sample)
        group_index = group_index_per_sample[sample]
        if group_index > 0
            included += 1
            feature_per_included[included] = feature_per_sample[sample]
            group_index_per_included[included] = group_index
        end
    end
    rank_per_included = tiedrank(feature_per_included)
    for included in 1:n_included
        rank_sum_per_group[group_index_per_included[included]] += rank_per_included[included]
    end
    return nothing
end

# Sum the ranks per group for a sparse feature: rank only the stored non-zero values (negatives and positives ranked
# separately), while the zero samples share the tied rank of the `[negatives... | zeros... | positives...]` block.
function sparse_rank_sum_per_group!(;
    rank_sum_per_group::AbstractVector{Float64},
    feature_per_sample::AbstractVector{<:Real},
    group_index_per_sample::AbstractVector{<:Integer},
    n_samples_per_group::AbstractVector{<:Integer},
    n_included_samples::Integer,
    positive::Bool,
)::Nothing
    n_groups = length(n_samples_per_group)
    stored_sample_per_stored = nzind(feature_per_sample)
    stored_value_per_stored = nzval(feature_per_sample)

    negative_value_per_negative = Float64[]
    group_index_per_negative = Int[]
    positive_value_per_positive = Float64[]
    group_index_per_positive = Int[]
    nonzero_count_per_group = zeros(Int, n_groups)

    for stored in eachindex(stored_sample_per_stored)
        group_index = group_index_per_sample[stored_sample_per_stored[stored]]
        if group_index == 0
            continue  # UNTESTED
        end
        value = Float64(stored_value_per_stored[stored])
        if value > 0
            push!(positive_value_per_positive, value)
            push!(group_index_per_positive, group_index)
            nonzero_count_per_group[group_index] += 1
        elseif value < 0 && !positive
            push!(negative_value_per_negative, value)
            push!(group_index_per_negative, group_index)
            nonzero_count_per_group[group_index] += 1
        end
        # A value of zero (an explicitly stored zero) is folded into the zeros block, like a structural zero.
    end

    n_negative = length(negative_value_per_negative)
    n_positive = length(positive_value_per_positive)
    n_zero = n_included_samples - n_negative - n_positive
    average_zero_rank = n_negative + (n_zero + 1) / 2

    if n_negative > 0
        rank_per_negative = tiedrank(negative_value_per_negative)
        for negative in 1:n_negative
            rank_sum_per_group[group_index_per_negative[negative]] += rank_per_negative[negative]
        end
    end
    if n_positive > 0
        rank_per_positive = tiedrank(positive_value_per_positive)
        positive_rank_offset = n_negative + n_zero
        for positive in 1:n_positive
            rank_sum_per_group[group_index_per_positive[positive]] += rank_per_positive[positive] + positive_rank_offset
        end
    end
    for group_index in 1:n_groups
        n_zero_in_group = n_samples_per_group[group_index] - nonzero_count_per_group[group_index]
        rank_sum_per_group[group_index] += n_zero_in_group * average_zero_rank
    end
    return nothing
end

# Turn the per-group rank-sums into AUCs, folded to `max(a, 1 - a)` so they are always `>= 0.5` (`0` for a group
# covering none or all of the population, where the AUC is undefined).
function finalize_auc_per_group!(;
    auc_per_group::AbstractVector{Float64},
    rank_sum_per_group::AbstractVector{Float64},
    n_samples_per_group::AbstractVector{<:Integer},
    n_included_samples::Integer,
)::Nothing
    for group_index in eachindex(n_samples_per_group)
        n_in_group = n_samples_per_group[group_index]
        n_out_of_group = n_included_samples - n_in_group
        if n_in_group == 0 || n_out_of_group == 0
            auc_per_group[group_index] = 0.0
        else
            auc = (rank_sum_per_group[group_index] - n_in_group * (n_in_group + 1) / 2) / (n_in_group * n_out_of_group)
            auc_per_group[group_index] = max(auc, 1 - auc)
        end
    end
    return nothing
end

end  # module
