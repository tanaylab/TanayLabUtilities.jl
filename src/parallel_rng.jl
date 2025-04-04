"""
Reproducible random number generator for parallel loops. Yes, the `TaskLocalRNG` is supposed to do this, but,
it actually depends on the way tasks are allocated to threads. The implementation here will give the same results
regardless of the thread scheduling policy.
"""
module ParallelRNG

export parallel_loop_with_rng

using ..Types

using Base.Threads
using Random

import Random.default_rng

"""
    parallel_loop_with_rng(
        body::Function,
        size::Maybe{Integer} = nothing;
        policy::Symbol = :greedy,
        seed::Maybe{Integer} = nothing,
        rng::Maybe{AbstractRNG} = nothing
    )::Nothing

Run the `body` in parallel (using the `@threads policy`) `size` times, passing it the iteration index, and a separate
`rng` that is seeded to a reproducible state regardless of the allocation of tasks to threads. A copy of this `rng` is
given to each iteration, after being reset to `seed + index` for reproducibility. If no `seed` is specified, it is just
sampled `rng` before the loop starts. If the `rng` isn't given, then this uses (and sets for each iteration) the
`default_rng()`. In this case passing it to the body is redundant but is still done for consistency.

```jldoctest
using Test
using Random

size = 10

function collect_rng(rng::AbstractRNG)::Vector{Float64}
    results = zeros(Float64, size)
    parallel_loop_with_rng(size; rng) do index, rng
        results[index] = rand(rng)
    end
    @test results[1] != results[2]
    return results
end

@test collect_rng(MersenneTwister(1)) == collect_rng(MersenneTwister(1))

function collect_default_rng()::Vector{Float64}
    results = zeros(Float64, size)
    parallel_loop_with_rng(size; seed = 123456, policy = :dynamic) do index, _
        results[index] = rand()
    end
    @test results[1] != results[2]
    return results
end

@test collect_default_rng() == collect_default_rng()

println("OK")

# output

OK
```
"""
function parallel_loop_with_rng(  # NOJET
    body::Function,
    size::Maybe{Integer} = nothing;
    policy::Symbol = :greedy,
    seed::Maybe{Integer} = nothing,
    rng::Maybe{AbstractRNG} = nothing,
)::Nothing
    @assert policy in (:greedy, :static, :dynamic)

    if seed === nothing
        seed = rand(copy(rng === nothing ? default_rng() : rng), Int64)
    end

    if policy == :greedy
        @threads :greedy for index in 1:size
            loop_body(body, index, seed, rng)
        end
    elseif policy == :static
        @threads :static for index in 1:size  # UNTESTED
            loop_body(body, index, seed, rng)
        end
    elseif policy == :dynamic
        @threads :dynamic for index in 1:size
            loop_body(body, index, seed, rng)
        end
    else
        @assert false
    end
end

function loop_body(body::Function, index::Integer, loop_seed::Integer, loop_rng::Maybe{AbstractRNG})::Nothing
    if loop_rng === nothing
        iteration_rng = default_rng()
    else
        iteration_rng = copy(loop_rng)
    end
    Random.seed!(iteration_rng, loop_seed + index)  # NOJET
    body(index, iteration_rng)
    return nothing
end

end
