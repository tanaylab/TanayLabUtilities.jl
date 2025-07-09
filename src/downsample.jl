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
"""
module Downsample

export downsample
export downsamples

using ..Brief
using ..Documentation
using ..MatrixFormats
using ..MatrixLayouts
using ..ParallelRNG
using ..Types
using Base.Threads
using Random
using Statistics

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
it is allocated automatically using the same element type as the input.

When downsampling a `matrix`, then `dims` must be specified to be `1`/`Rows` to separately downsample each row, or
`2`/`Columns` to separately downsample each column.

```jldoctest
using Test

# Columns

data = rand(1:100, 10, 5)
samples_per_column = vec(sum(data; dims = 1))

for samples in (100, 250, 500, 750, 1000)
    downsampled = downsample(data, samples; dims = 2)
    downsamples_per_column = vec(sum(downsampled; dims = 1))
    @test all(downsamples_per_column .== min.(samples_per_column, samples))
    too_small_mask = samples_per_column .<= samples
    @test all(downsampled[:, too_small_mask] .== data[:, too_small_mask])
end

# Rows

data = transpose(data)
samples_per_row = samples_per_column

for samples in (100, 250, 500, 750, 1000)
    downsampled = downsample(data, samples; dims = 1)
    downsamples_per_row = vec(sum(downsampled; dims = 2))
    @test all(downsamples_per_row .== min.(samples_per_row, samples))
    too_small_mask = samples_per_row .<= samples
    @test all(downsampled[too_small_mask, :] .== data[too_small_mask, :])
end

println("OK")

# output

OK
```
"""
function downsample(
    vector::AbstractVector{<:Integer},
    samples::Integer;
    rng::AbstractRNG = default_rng(),
    output::Maybe{AbstractVector} = nothing,
)::AbstractVector
    n_values = length(vector)

    if output === nothing
        output = similar_array(vector)  # UNTESTED
    end

    @assert length(output) == length(vector)

    if n_values > 0
        @assert minimum(vector) >= 0 "Downsampling a vector with negative values"
    end

    if n_values == 1
        output[1] = min(samples, vector[1])  # UNTESTED

    elseif n_values > 1
        tree = initialize_tree(vector)

        if tree[end] <= samples
            output .= vector

        else
            output .= 0
            for _ in 1:samples
                output[random_sample!(tree, rand(rng, 1:tree[end]))] += 1
            end
        end
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
    n_rows, n_columns = size(matrix)

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

    if dims == Rows
        parallel_loop_with_rng(1:n_rows; rng) do row_index, rng
            @views row_vector = matrix[row_index, :]
            @views output_vector = output[row_index, :]
            return downsample(row_vector, samples; rng, output = output_vector)
        end

    elseif dims == Columns
        parallel_loop_with_rng(1:n_columns; rng) do column_index, rng
            @views column_vector = matrix[:, column_index]
            @views output_vector = output[:, column_index]
            return downsample(column_vector, samples; rng, output = output_vector)
        end
    else
        @assert false
    end

    return output
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
                max(min_downsamples, quantile(samples_per_vector, min_downsamples_quantile)),
                quantile(samples_per_vector, max_downsamples_quantile),
            ),
        ),
    )
end

end
