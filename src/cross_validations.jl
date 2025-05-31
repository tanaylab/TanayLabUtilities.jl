"""
Cross validation functions.
"""
module CrossValidations

export pick_cross_validation_indices

using Random

import Random.default_rng

"""
    @kwdef struct CrossValidationIndices
        n_parts::Integer
        train_indices_per_part::AbstractVector{<:AbstractVector{<:Integer}}
        test_indices_per_part::AbstractVector{<:AbstractVector{<:Integer}}
    end

A set of cross-validation indices; `n_parts`, where each part is a combination of distinct test and training indices,
such that the training indices for each part are the union of the test indices of all the other parts.
"""
@kwdef struct CrossValidationIndices
    n_parts::Integer
    train_indices_per_part::AbstractVector{<:AbstractVector{<:Integer}}
    test_indices_per_part::AbstractVector{<:AbstractVector{<:Integer}}
end

"""
    function pick_cross_validation_indices(;
        full_indices::AbstractVector{<:Integer},
        cross_validation_parts::Integer,
        rng::AbstractRNG,
    )::CrossValidationIndices

Given a vector of `full_indices`, split them into `cross_validation_parts` where each part has distinct training and
testing indices.

```jldoctest
using Test
using Random

cross_validation_indices = pick_cross_validation_indices(;
    full_indices = collect(1:12),
    cross_validation_parts = 3,
    rng = Random.default_rng()
)
@test all([
    length(test_indices)
    for test_indices
    in cross_validation_indices.test_indices_per_part
] .== 4)
@test all([
    length(train_indices)
    for train_indices
    in cross_validation_indices.train_indices_per_part
] .== 8)
@test all([
    length(union(Set(test_indices), Set(train_indices)))
    for (test_indices, train_indices)
    in zip(cross_validation_indices.test_indices_per_part, cross_validation_indices.train_indices_per_part)
] .== 12)

println("OK")

# output

OK
```
"""
function pick_cross_validation_indices(;
    full_indices::AbstractVector{<:Integer},
    cross_validation_parts::Integer,
    rng::AbstractRNG,
)::CrossValidationIndices
    n_full = length(full_indices)
    @assert n_full >= cross_validation_parts

    train_indices_per_part = Vector{Int}[]
    test_indices_per_part = Vector{Int}[]

    shuffled_full_indices = shuffle(rng, full_indices)
    parts_size = n_full / cross_validation_parts

    for part_index in 1:cross_validation_parts
        first_test_position = Int(round((part_index - 1) * parts_size)) + 1
        last_test_position = Int(round(part_index * parts_size))
        test_positions = first_test_position:last_test_position

        train_indices = vcat(
            shuffled_full_indices[1:(first_test_position - 1)],
            shuffled_full_indices[(last_test_position + 1):end],
        )
        test_indices = shuffled_full_indices[test_positions]
        @assert length(test_indices) + length(train_indices) == n_full

        push!(train_indices_per_part, train_indices)
        push!(test_indices_per_part, test_indices)
    end

    return CrossValidationIndices(; n_parts = cross_validation_parts, train_indices_per_part, test_indices_per_part)
end

end  # module
