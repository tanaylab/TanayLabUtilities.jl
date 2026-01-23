"""
Parallel loops.
"""
module ParallelLoops

export parallel_loop_with_rng
export parallel_loop_wo_rng
export DebugProgress

using ..Types
using ..FlameTime

using Base.Threads
using Logging
using ProgressMeter
using Random

import Random.default_rng

"""
    parallel_loop_wo_rng(
        body::Function,
        indices::AbstractVector{<:Integer};
        name::AbstractString = ".loop",
        policy::Symbol = :greedy,
        progress::Maybe{Progress} = nothing,
    )::Nothing

Run the `body` in parallel, passing it the iteration `index`. The `policy` is passed to `@threads` if it is one of (the
default `:greedy`, `:dynamic`, or `:static`). If it is `:serial`, then the loop is not run in parallel (useful for
debugging).

If this is nested (and the containing loop is parallel), `policy` is ignored and the loop is executed serially. This
allows functions to be parallel if invoked from the main thread, but become serial if invoked from inside an already
parallel loop.

The `name` is used for [`flame_timed`](@ref) for the whole loop. If the loop is parallel, the whole loop is a line in
the serial flame file, and each iteration is a line in the parallel flame file. The serial file therefore gives a view
of the elapsed time of the top-level loops, to identify what actually matters, and the parallel file shows the internal
breakdown of the computations in each loop. It is sadly impossible to include threaded tasks in the same flamegraph in a
meaningful way (at least not without extending the flamegraph format and visualization).

If `progress` is specified, it is updated for each iteration.

!!! note

    The code inside the loop body should not use any random number generation. Use [`parallel_loop_with_rng`](@ref) if
    random number generation is needed.

```jldoctest
using Test
using Random

size = 10

for policy in (:serial, :greedy, :dynamic, :static)
    results = zeros(Int, size)
    parallel_loop_wo_rng(1:size; policy) do index
        results[index] = index
        return nothing
    end
    @test results == collect(1:size)
end

println("OK")

# output

OK
```
"""
function parallel_loop_wo_rng(  # NOJET
    body::Function,
    indices::AbstractVector{<:Integer};
    name::AbstractString = ".loop",
    policy::Symbol = :greedy,
    progress::Maybe{Progress} = nothing,  # NOLINT
)::Nothing
    @assert policy in (:greedy, :static, :dynamic, :serial)

    base_private_storage = task_local_storage()
    base_is_in_parallel = get(base_private_storage, :is_in_parallel, false)
    base_flame_stack = FlameTime.get_flame_stack(base_private_storage)
    if base_flame_stack !== nothing
        if startswith(name, ".")  # UNTESTED
            @assert length(base_flame_stack) > 0  # UNTESTED
            name = base_flame_stack[end] * name  # UNTESTED
        end
        base_flame_stack = join(base_flame_stack, ";")  # UNTESTED
    elseif startswith(name, ".")
        name = name[2:end]
    end

    if base_is_in_parallel
        progress = nothing  # UNTESTED
        if policy != :serial  # UNTESTED
            policy = :serial  # UNTESTED
        end
    end

    flame_timed(name; iterations = policy == :serial ? length(indices) : -length(indices)) do
        if policy == :serial
            for index in indices
                body(index)
                if progress !== nothing
                    next!(progress)  # UNTESTED
                end
            end
        elseif policy == :greedy
            @threads :greedy for index in indices
                index_private_storage = task_local_storage()
                @assert index_private_storage !== base_private_storage
                index_private_storage[:is_in_parallel] = true
                index_flame_stack = [base_flame_stack]
                index_private_storage[:flame_stack] = index_flame_stack

                flame_timed(name) do
                    body(index)
                    if progress !== nothing
                        next!(progress)  # NOJET # UNTESTED
                    end
                    return nothing
                end
            end

        elseif policy == :static
            @threads :static for index in indices
                index_private_storage = task_local_storage()
                @assert index_private_storage !== base_private_storage
                index_private_storage[:is_in_parallel] = true
                index_flame_stack = [base_flame_stack]
                index_private_storage[:flame_stack] = index_flame_stack

                flame_timed(name) do
                    body(index)
                    if progress !== nothing
                        next!(progress)  # NOJET # UNTESTED
                    end
                    return nothing
                end
            end

        elseif policy == :dynamic
            @threads :dynamic for index in indices
                index_private_storage = task_local_storage()
                @assert index_private_storage !== base_private_storage
                index_private_storage[:is_in_parallel] = true
                index_flame_stack = [base_flame_stack]
                index_private_storage[:flame_stack] = index_flame_stack

                flame_timed(name) do
                    body(index)
                    if progress !== nothing
                        next!(progress)  # NOJET # UNTESTED
                    end
                    return nothing
                end
            end

        else
            @assert false
        end
    end

    return nothing
end

"""
    parallel_loop_with_rng(
        body::Function,
        indices::AbstractVector{<:Integer};
        name::AbstractString = ".loop",
        policy::Symbol = :greedy,
        progress::Maybe{Progress} = nothing,
        seed::Maybe{Integer} = nothing,
        rng::Maybe{AbstractRNG} = nothing
    )::Nothing

Run the `body` in parallel, passing it the iteration `index` and a separate `rng` that is seeded to a reproducible
state regardless of the allocation of tasks to threads. A copy of this `rng` is given to each iteration, after being
reset to `seed + index` for reproducibility. If no `seed` is specified, it is just sampled `rng` before the loop starts.
If the `rng` isn't given, then this uses (and sets for each iteration) the `default_rng()`. In this case passing it to
the body is redundant but is still done for consistency.

The `policy` is passed to `@threads` if it is one of (the default `:greedy`, `:dynamic`, or `:static`). If it is
`:serial`, then the loop is not run in parallel (useful for debugging).

If this is nested (and the containing loop is parallel), `policy` is ignored and the loop is executed serially. This
allows functions to be parallel if invoked from the main thread, but become serial if invoked from inside an already
parallel loop.

The `name` is used for [`flame_timed`](@ref) for the whole loop. If the loop is parallel, the whole loop is a line in
the serial flame file, and each iteration is a line in the parallel flame file. The serial file therefore gives a view
of the elapsed time of the top-level loops, to identify what actually matters, and the parallel file shows the internal
breakdown of the computations in each loop. It is sadly impossible to include threaded tasks in the same flamegraph in a
meaningful way (at least not without extending the flamegraph format and visualization).

If `progress` is specified, it is updated for each iteration.

!!! note

    Yes, the `TaskLocalRNG` is supposed to do this, but, it actually depends on the way tasks are allocated to threads.
    The implementation here will give the same results regardless of the thread scheduling policy. Sigh.

```jldoctest
using Test
using Random

size = 10

function collect_rng(rng::AbstractRNG)::Vector{Float64}
    results = zeros(Float64, size)
    parallel_loop_with_rng(1:size; rng) do index, rng
        results[index] = rand(rng)
    end
    @test results[1] != results[2]
    return results
end

@test collect_rng(MersenneTwister(1)) == collect_rng(MersenneTwister(1))

function collect_default_rng()::Vector{Float64}
    results = zeros(Float64, size)
    parallel_loop_with_rng(1:size; seed = 123456, policy = :dynamic) do index, _
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
    indices::AbstractVector{<:Integer};
    name::AbstractString = ".loop",
    policy::Symbol = :greedy,
    progress::Maybe{Progress} = nothing,
    seed::Maybe{Integer} = nothing,
    rng::Maybe{AbstractRNG} = nothing,
)::Nothing
    @assert policy in (:greedy, :static, :dynamic, :serial)

    if seed === nothing
        seed = rand(copy(rng === nothing ? default_rng() : rng), Int64)
    end

    parallel_loop_wo_rng(indices; name, policy, progress) do index
        if rng === nothing
            iteration_rng = default_rng()
        else
            iteration_rng = copy(rng)
        end
        Random.seed!(iteration_rng, seed + index)  # NOJET
        body(index, iteration_rng)
        return nothing
    end

    return nothing
end

function is_debug_enabled_for_caller()  # UNTESTED
    logger = Logging.current_logger()
    if logger.min_level > Logging.Debug
        return false
    end
    stack = stacktrace(backtrace())
    if length(stack) < 4
        caller_module = Main
    else
        caller_frame = stack[4]
        if caller_frame === nothing || caller_frame.linfo === nothing || caller_frame.linfo.def === nothing
            caller_module = Main
        else
            caller_module = caller_frame.linfo.def.module  # NOJET
        end
    end
    return Logging.shouldlog(logger, Logging.Debug, caller_module, :debug, :check)
end

"""
    DebugProgress(n::Integer; kwargs...)::Maybe{Progress}

Same as `Progress` in `ProgressMeter`, but returns `nothing` if debug is not enabled for the modules calling
`DebugProgress`.
"""
function DebugProgress(n::Integer; kwargs...)::Maybe{Progress}  # UNTESTED
    if is_debug_enabled_for_caller()
        return Progress(n; kwargs...)
    else
        return nothing
    end
end

end
