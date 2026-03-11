"""
Parallel loops.
"""
module ParallelLoops

export parallel_gc
export parallel_loop_with_rng
export parallel_loop_wo_rng
export DebugProgress

using ..Brief
using ..Types
using ..FlameTime

using Base.Threads
using Logging
using ProgressMeter
using Random

import Random.default_rng

TOTAL_MEMORY = 0

"""
Above what fraction of the total memory to enable GC in parallel loops. If set to a non-zero value, assume the machine
is "mostly" dedicated to the computation, and only trigger GC when working in parallel if (at the end of a parallel loop
iteration) the used memory is at least this fraction of the total memory of the machine.

This is initialized from the `TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION` environment variable.
"""
TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION = 0.0

function __init__()::Nothing
    live_bytes_gc_threshold_fraction = get(ENV, "TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION", nothing)
    global TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION
    if live_bytes_gc_threshold_fraction === nothing
        TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION = 0.0
    else
        TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION = parse(Float64, live_bytes_gc_threshold_fraction)  # UNTESTED
        @assert 0 <= TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION < 1  # UNTESTED
    end
    if TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION != 0.0
        global TOTAL_MEMORY  # UNTESTED
        TOTAL_MEMORY = Sys.total_memory()  # UNTESTED
        @info "Will only GC when using over: $(TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION)" *  # UNTESTED
              " of the $(delimited_number(Int(round(TOTAL_MEMORY / 1e6))))MB memory in parallel loops." _group =
            :tlu_env
    end
    return nothing
end

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

!!! note

    If the code runs in parallel, we disable the GC through it and invoke [`parallel_gc`](@ref) at the start of each
    iteration. In general parallel code "shouldn't" perform "many" memory allocations. Use [`ParallelStorage`](@ref
    TanayLabUtilities.ParallelStorage) if large temporary data is needed.

If this is nested (and the containing loop is parallel), `policy` is ignored and the loop is executed serially. This
allows functions to be parallel if invoked from the main thread, but become serial if invoked from inside an already
parallel loop.

The `name` is used for [`flame_timed`](@ref) for the whole loop. If the loop is parallel, the whole loop is a line in
the serial flame file, and each iteration is a line in the parallel flame file. The serial file therefore gives a view
of the elapsed time of the top-level loops, to identify what actually matters, and the parallel file shows the internal
breakdown of the computations in each loop. It is sadly impossible to include threaded tasks in the same flamegraph in a
meaningful way (at least not without extending the flamegraph format and visualization).

If `progress` is specified, it is updated for each iteration. If `progress_chunk` is specified, then `indices` must be a
range `1:n`, and we throttle calls to the progress bar update to one every `progress_chunk`.

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
    progress_chunk::Maybe{Integer} = nothing,
)::Nothing
    @assert policy in (:greedy, :static, :dynamic, :serial)

    if progress_chunk !== nothing
        @assert indices[1] == 1  # UNTESTED
        @assert indices[end] == length(indices)  # UNTESTED
    end

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
                    if progress_chunk === nothing  # UNTESTED
                        next!(progress)  # UNTESTED
                    else
                        if index % progress_chunk == 0  # UNTESTED
                            next!(progress; step = progress_chunk)  # UNTESTED
                        end
                    end
                end
            end
        else
            @inline function iteration_body(index::Integer)::Nothing
                index_private_storage = task_local_storage()
                @assert index_private_storage !== base_private_storage
                index_private_storage[:is_in_parallel] = true
                index_flame_stack = [base_flame_stack]
                index_private_storage[:flame_stack] = index_flame_stack

                flame_timed(name) do
                    parallel_gc()
                    body(index)
                    if progress !== nothing
                        if progress_chunk === nothing  # UNTESTED
                            next!(progress)  # NOJET # UNTESTED
                        else
                            if index % progress_chunk == 0  # UNTESTED
                                next!(progress; step = progress_chunk)  # UNTESTED
                            end
                        end
                    end
                    return nothing
                end

                return nothing
            end

            if TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION > 0
                GC.enable(false)  # UNTESTED
            end
            try
                if policy == :greedy
                    @threads :greedy for index in indices
                        iteration_body(index)
                    end

                elseif policy == :static
                    @threads :static for index in indices
                        iteration_body(index)
                    end

                elseif policy == :dynamic
                    @threads :dynamic for index in indices
                        iteration_body(index)
                    end

                else
                    @assert false
                end
            finally
                if TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION > 0
                    GC.enable(true)  # UNTESTED
                end
            end
        end
    end

    if progress !== nothing && progress_chunk !== nothing && length(indices) % progress_chunk > 0
        next!(progress; step = length(indices) % progress_chunk)  # UNTESTED
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

function is_debug_enabled_for_caller(group::Maybe{Symbol} = nothing)  # UNTESTED
    logger = Logging.current_logger()
    if logger.min_level <= Logging.Debug
        return true
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
    return Logging.shouldlog(logger, Logging.Debug, caller_module, group, :check)
end

"""
    DebugProgress(n::Integer; kwargs...)::Maybe{Progress}

Same as `Progress` in `ProgressMeter`, but returns `nothing` if debug is not enabled for the modules calling
`DebugProgress`.
"""
function DebugProgress(n::Integer; group::Maybe{Symbol} = nothing, kwargs...)::Maybe{Progress}  # UNTESTED
    if is_debug_enabled_for_caller(group)
        return Progress(n; kwargs...)
    else
        return nothing
    end
end

"""
    parallel_gc()::Nothing

Enable GC inside a parallel loop if the memory pressure is high, that is, if the currently used memory is above
[`TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION`](@ref) of the total system memory. If the memory usage is less, disabled GC. This is
invoked automatically at the start of each iteration. If the parallel code does so many allocations that it makes sense
to sprinkle additional calls to this in the middle of it, then it is worthwhile to reconsider it. Using
[`ParallelStorage`](@ref TanayLabUtilities.ParallelStorage) may help.

This is a no-op unless [`TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION`](@ref) is set to a zero value (the default).
"""
@inline function parallel_gc()::Nothing
    if TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION > 0
        GC.enable(Base.gc_live_bytes() > TOTAL_MEMORY * TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION)  # UNTESTED
    end
    return nothing
end

end
