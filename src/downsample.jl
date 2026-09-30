"""
Downsampling of data. The idea is that you have a vector containing a total of some (large) `K` counts of samples with
values `1..N` drawn from a multinomial distribution (with different probabilities for getting each of the `1..N`
values). Generate a vector with a total of some (smaller) `k` samples. This is typically done to a set of vectors,
typically to all columns of a matrix (each with its own `K(j)`), to get a set of vectors with the same `k`.

This is useful for meaningfully comparing the vectors (for example, computing correlations between them). Without
downsampling, distance measures between such vectors are biases by the sampling depth `K`. For example, correlations
with deeper (higher total samples) vectors will tend to be higher.

Downsampling discards data so we'd like the target `k` to be as large as possible. Typically this isn't the minimal
`K(j)` to avoid a few shallow sampled vectors from ruining the quality of the results; we accept that a small fraction
of the vectors will keep their original `K(j)` samples when this is less than the chosen `k`.

Downsampling only works on integer data. If given fractional data, one should first use [`round_counts`](@ref) to
convert it to integers.
"""
module Downsample

export downsample
export downsamples
export round_counts

using ..Brief
using ..Documentation
using ..MatrixFormats
using ..MatrixLayouts
using ..ParallelLoops
using ..SparseStatistics
using ..Types
using Base.Threads
using Random
using SparseArrays

import ..MatrixLayouts.check_efficient_action
import Random.default_rng

"""
    downsample(
        vector::AbstractVector{<:Integer},
        samples::Integer;
        rng::AbstractRNG = default_rng(),
        output::Maybe{AbstractVector} = nothing,
    )::AbstractVector

    downsample(
        matrix::AbstractMatrix{<:Integer},
        samples::Integer;
        dims::Integer,
        rng::AbstractRNG = default_rng(),
        output::Maybe{AbstractMatrix} = nothing,
    )::AbstractMatrix

Given a `vector` of integer non-negative data values, return a new vector such that the sum of entries in it is
`samples`. Think of the original vector as containing a number of marbles in each entry. We randomly pick `samples`
marbles from this vector; each time we pick a marble we take it out of the original vector and move it to the same
position in the result.

If the sum of the entries of a vector is less than `samples`, it is copied to the output. If `output` is not specified,
it is allocated automatically using the same element type as the input. For sparse data, it only examines the non-zero
entries.

When downsampling a `matrix`, then `dims` must be specified to be `1`/`Rows` to separately downsample each row, or
`2`/`Columns` to separately downsample each column.

```jldoctest
# Columns

data = rand(1:100, 10, 5)
samples_per_column = vec(sum(data; dims = 1))

for samples in (100, 250, 500, 750, 1000)
    downsampled = downsample(data, samples; dims = 2)
    downsamples_per_column = vec(sum(downsampled; dims = 1))
    @assert all(downsamples_per_column .== min.(samples_per_column, samples))
    too_small_mask = samples_per_column .<= samples
    @assert all(downsampled[:, too_small_mask] .== data[:, too_small_mask])
end

# Rows

data = flip(data)
samples_per_row = samples_per_column

for samples in (100, 250, 500, 750, 1000)
    downsampled = downsample(data, samples; dims = 1)
    downsamples_per_row = vec(sum(downsampled; dims = 2))
    @assert all(downsamples_per_row .== min.(samples_per_row, samples))
    too_small_mask = samples_per_row .<= samples
    @assert all(downsampled[too_small_mask, :] .== data[too_small_mask, :])
end

# output

```
"""
function downsample(
    vector::AbstractVector{<:Integer},
    samples::Integer;
    rng::AbstractRNG = default_rng(),
    output::Maybe{AbstractVector} = nothing,
)::AbstractVector
    if output === nothing
        output = similar_array(vector)
    end

    @assert length(output) == length(vector)

    if issparse(vector)
        output .= 0
        downsample_values!(output, nzind(vector), nzval(vector), samples, rng)
    else
        downsample_values!(output, eachindex(vector), vector, samples, rng)
    end

    return output
end

function downsample(
    matrix::AbstractMatrix{<:Integer},
    samples::Integer;
    dims::Integer,
    rng::AbstractRNG = default_rng(),
    output::Maybe{AbstractMatrix} = nothing,
)::AbstractMatrix
    @assert 1 <= dims <= 2

    if major_axis(matrix) !== nothing
        check_efficient_action(@source_location()..., "matrix", matrix, dims)
    end

    if output === nothing
        output = similar_array(matrix; default_major_axis = dims)
    else
        @assert size(output) == size(matrix)  # UNTESTED
        if major_axis(output) !== nothing  # UNTESTED
            check_efficient_action(@source_location()..., "output", output, dims)  # UNTESTED
        end
    end

    parallel_loop_on_slices(output, matrix; dims, name = "downsample", rng) do output_vector, positions, values, rng
        downsample_values!(output_vector, positions, values, samples, rng)
        return nothing
    end

    return output
end

# Downsample the `values` into the `positions` of the `output`. Other entries of the `output` are not modified.
function downsample_values!(
    output::AbstractVector,
    positions::AbstractVector{<:Integer},
    values::AbstractVector{<:Integer},
    samples::Integer,
    rng::AbstractRNG,
)::Nothing
    n_values = length(values)

    if n_values > 0
        @assert minimum(values) >= 0 "Downsampling a vector with negative values"
    end

    if n_values == 1
        output[positions[1]] = min(samples, values[1])

    elseif n_values > 1
        tree = initialize_tree(values)

        if tree[end] <= samples
            @views output[positions] .= values

        else
            @views output[positions] .= 0
            for _ in 1:samples
                output[positions[random_sample!(tree, rand(rng, 1:tree[end]))]] += 1
            end
        end
    end

    return nothing
end

"""
    round_counts(
        vector::AbstractVector{<:Real};
        rng::AbstractRNG = default_rng(),
        output::Maybe{AbstractVector{<:Integer}} = nothing,
    )::AbstractVector

    round_counts(
        matrix::AbstractMatrix{<:Real};
        dims::Integer,
        rng::AbstractRNG = default_rng(),
        output::Maybe{AbstractMatrix{<:Integer}} = nothing,
    )::AbstractMatrix

Given a `vector` of non-negative, possibly fractional, counts, return a new vector of integer counts. Each entry is
randomly rounded to either the `floor` or the `ceil` of its value. The expected value of each entry is its original
value. If the total of the original vector is an integer, the result has exactly this total. Otherwise, the total of
the result is either the `floor` or the `ceil` of the original total.

This allows downsampling fractional counts, by first rounding them and then calling [`downsample`](@ref). Integer
counts are copied as-is, so for them this is identical to calling [`downsample`](@ref) directly.

This uses the ordered pivotal method (Deville and Tillé, 1998). It takes `O(N)` time and uses one random number per
fractional entry. For sparse data, it only examines the non-zero entries. If `output` is not specified, it is
allocated automatically with an `Int` element type.

When rounding a `matrix`, then `dims` must be specified to be `1`/`Rows` to separately round each row, or `2`/`Columns`
to separately round each column.

```jldoctest
# Columns

data = rand(10, 5) .* 10
rounded = round_counts(data; dims = 2)
@assert all((rounded .== floor.(data)) .| (rounded .== ceil.(data)))

sums_per_column = vec(sum(data; dims = 1))
rounded_sums_per_column = vec(sum(rounded; dims = 1))
@assert all(
    (rounded_sums_per_column .== floor.(sums_per_column)) .| (rounded_sums_per_column .== ceil.(sums_per_column))
)

integers = rand(0:10, 10, 5)
@assert round_counts(integers; dims = 2) == integers

# Rows

data = flip(data)
rounded = round_counts(data; dims = 1)
@assert all((rounded .== floor.(data)) .| (rounded .== ceil.(data)))

sums_per_row = vec(sum(data; dims = 2))
rounded_sums_per_row = vec(sum(rounded; dims = 2))
@assert all((rounded_sums_per_row .== floor.(sums_per_row)) .| (rounded_sums_per_row .== ceil.(sums_per_row)))

integers = flip(integers)
@assert round_counts(integers; dims = 1) == integers

# output

```
"""
function round_counts(
    vector::AbstractVector{<:Real};
    rng::AbstractRNG = default_rng(),
    output::Maybe{AbstractVector{<:Integer}} = nothing,
)::AbstractVector
    if output === nothing
        output = similar_array(vector; eltype = Int)
    end

    @assert length(output) == length(vector)

    if issparse(vector)
        output .= 0
        round_counts_of_values!(output, nzind(vector), nzval(vector), rng)
    else
        round_counts_of_values!(output, eachindex(vector), vector, rng)
    end

    return output
end

function round_counts(
    matrix::AbstractMatrix{<:Real};
    dims::Integer,
    rng::AbstractRNG = default_rng(),
    output::Maybe{AbstractMatrix{<:Integer}} = nothing,
)::AbstractMatrix
    @assert 1 <= dims <= 2

    if major_axis(matrix) !== nothing
        check_efficient_action(@source_location()..., "matrix", matrix, dims)
    end

    if output === nothing
        output = similar_array(matrix; eltype = Int, default_major_axis = dims)
    else
        @assert size(output) == size(matrix)
        if major_axis(output) !== nothing
            check_efficient_action(@source_location()..., "output", output, dims)
        end
    end

    parallel_loop_on_slices(round_counts_of_values!, output, matrix; dims, name = "round_counts", rng)

    return output
end

# Call `body(output_vector, positions, values, rng)` in parallel for each row (`dims = Rows`) or column
# (`dims = Columns`) of the `matrix`. If the `matrix` is sparse with a major axis of `dims`, the `output_vector` is
# zero-filled and the `positions` and `values` are only of the non-zero entries. Otherwise, they are of all the entries.
function parallel_loop_on_slices(
    body::Function,
    output::AbstractMatrix,
    matrix::AbstractMatrix;
    dims::Integer,
    name::AbstractString,
    rng::AbstractRNG,
)::Nothing
    n_rows, n_columns = size(matrix)

    if issparse(matrix) && major_axis(matrix) == dims
        if dims == Columns
            column_major_matrix = matrix
            column_major_output = output
        else
            column_major_matrix = flip(matrix)
            column_major_output = flip(output)
        end

        column_offsets = colptr(column_major_matrix)
        row_indices = rowval(column_major_matrix)
        nonzero_values = nzval(column_major_matrix)
        n_iterations = size(column_major_matrix, 2)

        parallel_loop_with_rng(1:n_iterations; name, rng) do iteration_index, rng
            slice_first = Int(column_offsets[iteration_index])
            slice_last = Int(column_offsets[iteration_index + 1]) - 1
            @views output_vector = column_major_output[:, iteration_index]
            output_vector .= 0
            @views body(output_vector, row_indices[slice_first:slice_last], nonzero_values[slice_first:slice_last], rng)
            return nothing
        end

    elseif dims == Rows
        parallel_loop_with_rng(1:n_rows; name, rng) do row_index, rng
            @views row_vector = matrix[row_index, :]
            @views output_vector = output[row_index, :]
            body(output_vector, eachindex(row_vector), row_vector, rng)
            return nothing
        end

    elseif dims == Columns
        parallel_loop_with_rng(1:n_columns; name, rng) do column_index, rng
            @views column_vector = matrix[:, column_index]
            @views output_vector = output[:, column_index]
            body(output_vector, eachindex(column_vector), column_vector, rng)
            return nothing
        end

    else
        @assert false
    end

    return nothing
end

# Ordered pivotal method. The fractional parts are paired one by one with a single carried fraction. Each pairing moves
# the fractional mass between the two so that one of them becomes an integer, while preserving the expected values.
function round_counts_of_values!(
    output::AbstractVector{<:Integer},
    positions::AbstractVector{<:Integer},
    values::AbstractVector{<:Real},
    rng::AbstractRNG,
)::Nothing
    carry_position = 0
    carry_fraction = zero(eltype(values))

    for (position, value) in zip(positions, values)
        @assert value >= 0 "Rounding a vector with negative values"
        integer_part = floor(value)
        fraction = value - integer_part
        output[position] = integer_part

        if fraction > 0
            if carry_position == 0
                carry_position = position
                carry_fraction = fraction

            else
                total_fraction = carry_fraction + fraction
                if total_fraction < 1
                    if rand(rng) >= carry_fraction / total_fraction
                        carry_position = position
                    end
                    carry_fraction = total_fraction

                else
                    if rand(rng) < (1 - fraction) / (2 - total_fraction)
                        output[carry_position] += 1
                        carry_position = position
                    else
                        output[position] += 1
                    end
                    carry_fraction = total_fraction - 1
                    if carry_fraction == 0
                        carry_position = 0
                    end
                end
            end
        end
    end

    if carry_position > 0 && rand(rng) < carry_fraction
        output[carry_position] += 1
    end

    return nothing
end

function initialize_tree(input::AbstractVector{T})::AbstractVector{T} where {T <: Integer}
    n_values = length(input)
    @assert n_values > 1

    n_values_in_level = ceil_power_of_two(n_values)
    tree_size = 2 * n_values_in_level - 1

    tree = Vector{T}(undef, tree_size)

    tree[1:n_values] .= input
    tree[(n_values + 1):end] .= 0

    tree_of_level = tree

    while (n_values_in_level > 1)
        @assert iseven(n_values_in_level)

        @views input_of_level = tree_of_level[1:n_values_in_level]
        @views tree_of_level = tree_of_level[(n_values_in_level + 1):end]
        n_values_in_level = div(n_values_in_level, 2)

        @assert length(tree_of_level) >= n_values_in_level

        for index_in_level in 1:n_values_in_level
            left_value = input_of_level[index_in_level * 2 - 1]
            right_value = input_of_level[index_in_level * 2]
            tree_of_level[index_in_level] = left_value + right_value
        end
    end

    @assert length(tree_of_level) == 1

    return tree
end

function ceil_power_of_two(size::Integer)::Integer
    return 2^Int(ceil(log2(size)))
end

function random_sample!(tree::AbstractVector{<:Integer}, random::Integer)::Integer
    size_of_level = 1
    base_of_level = length(tree)

    index_in_level = 1
    index_in_tree = base_of_level + index_in_level - 1

    while true
        @assert tree[index_in_tree] > 0
        tree[index_in_tree] -= 1

        size_of_level *= 2
        base_of_level -= size_of_level

        if base_of_level <= 0
            return index_in_level
        end

        index_in_level = index_in_level * 2 - 1
        index_in_tree = base_of_level + index_in_level - 1
        right_random = random - tree[index_in_tree]

        if right_random > 0
            index_in_level += 1
            index_in_tree += 1
            random = right_random
        end
    end
end

"""
    downsamples(
        samples_per_vector::AbstractVector{<:Integer};
        min_downsamples::Integer = $(DEFAULT.min_downsamples),
        min_downsamples_quantile::AbstractFloat = $(DEFAULT.min_downsamples_quantile),
        max_downsamples_quantile::AbstractFloat = $(DEFAULT.max_downsamples_quantile),
    )::Integer

When downsampling multiple vectors (the amount of data in each available in `samples_per_vector`), we need to pick a
"reasonable" number of samples to downsample to. We have conflicting requirements, so this is a compromise. First, we
want most vectors to have at least the target number of samples, so we start with the `min_downsamples_quantile` of the
`samples_per_vector`. Second, we also want to have at least `min_downsamples` to ensure we don't throw away too much
data even if many vectors are sparse, so we increase the target to this value. Finally, we don't want a target which is
too big for too many vectors, so so we reduce the result to the `max_downsamples_quantile` of the `samples_per_vector`.

!!! note

    The defaults (especially `min_downsamples`) were chosen to fit our needs (downsampling UMIs of sc-RNA-seq data). You
    will need to tweak them when using this for other purposes.

```jldoctest
downsamples([100, 500, 1000])

# output

500
```
"""
@documented function downsamples(
    samples_per_vector::AbstractVector{<:Integer};
    min_downsamples::Integer = 750,
    min_downsamples_quantile::AbstractFloat = 0.05,
    max_downsamples_quantile::AbstractFloat = 0.5,
)::Integer
    @assert 0 <= min_downsamples_quantile <= max_downsamples_quantile <= 1
    return Int(  # NOJET
        round(
            min(
                max(min_downsamples, sparse_quantile(samples_per_vector, min_downsamples_quantile)),
                sparse_quantile(samples_per_vector, max_downsamples_quantile),
            ),
        ),
    )
end

end
