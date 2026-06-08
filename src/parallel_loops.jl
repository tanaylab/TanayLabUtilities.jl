"""
Parallel loops.
"""
module ParallelLoops

export DebugProgress
export DebugProgressUnknown
export is_in_parallel_loop
export parallel_loop_with_rng
export parallel_loop_wo_rng

using ..Brief
using ..Types
using ..FlameTime

using Base.Threads
using LinearAlgebra
using Logging
using ProgressMeter
using Random

import Random.default_rng

TOTAL_MEMORY = 0

"""
    is_in_parallel_loop()::Bool

Return whether we are inside a [`parallel_loop_wo_rng`](@ref) or [`parallel_loop_with_rng`](@ref).
"""
function is_in_parallel_loop()::Bool
    base_private_storage = task_local_storage()
    return haskey(base_private_storage, :is_in_parallel_loop)
end

# In-place round-robin distribute `order` for `n_threads` static contiguous chunks: chunk `t` ends up holding every
# `n_threads`-th element of the original `order` starting from position `t`. Matches the chunking convention of
# `@threads :static`: the first `n_full_threads = n_threads - n_overflow` threads get the full `chunk_size =
# cld(n, n_threads)` items, the remaining `n_overflow = chunk_size * n_threads - n` threads get one fewer. Uses
# cycle-following with negation as the visited marker - assumes all entries of `order` are positive.
function round_robin_for_static_chunks!(order::AbstractVector{<:Integer}, n_threads::Integer)::Nothing
    n = length(order)
    if n_threads <= 1 || n <= 1
        return nothing  # UNTESTED
    end
    chunk_size = cld(n, n_threads)
    n_overflow = chunk_size * n_threads - n
    n_full_threads = n_threads - n_overflow
    full_threads_total_items = n_full_threads * chunk_size

    @inbounds for cycle_start in 1:n
        if order[cycle_start] < 0
            continue
        end
        saved_value = order[cycle_start]
        pos = cycle_start
        while true
            # Map output `pos` back to the input position whose original value should land at `pos`.
            if pos <= full_threads_total_items
                thread_index = cld(pos, chunk_size)
                position_in_chunk = pos - (thread_index - 1) * chunk_size
            else
                leftover = pos - full_threads_total_items
                partial_chunk_size = chunk_size - 1
                thread_offset = cld(leftover, partial_chunk_size)
                thread_index = n_full_threads + thread_offset
                position_in_chunk = leftover - (thread_offset - 1) * partial_chunk_size
            end
            source = (position_in_chunk - 1) * n_threads + thread_index
            if source == cycle_start
                order[pos] = -saved_value
                break
            end
            order[pos] = -order[source]
            pos = source
        end
    end
    @inbounds for index in 1:n
        order[index] = -order[index]
    end
    return nothing
end

"""
    parallel_loop_wo_rng(
        body::Function,
        indices::AbstractVector{<:Integer};
        name::AbstractString = ".loop",
        policy::Symbol = :greedy_sticky,
        order::Maybe{AbstractVector{<:Integer}} = nothing,
        weights::Maybe{AbstractVector{<:Integer}} = nothing,
        nested::Bool = false,
        progress::Maybe{Progress} = nothing,
    )::Nothing

Run the `body` in parallel, passing it the iteration `index`. The `policy` is one of:

  * `:greedy_sticky` - `nthreads()` sticky worker tasks spawned via `@threads :static for _ in 1:nthreads()`, each
    pulling the next index position from an `Atomic{Int}` counter. Combines the dynamic load balancing of `:greedy`
    with the threadid stability of `:static`: tasks never migrate, so `threadid()`-indexed scratch is safe inside the
    body. Pair with a heaviest-first `order` (or `weights`) to attack the tail-latency problem - heavy items dispatch
    first, light items fill the tail.
  * `:greedy`, `:dynamic`, `:static` - passed straight to `@threads`. Note that `:greedy` and `:dynamic` create
    *non-sticky* tasks that may migrate across threads at yield points, so `threadid()`-indexed scratch is unsafe
    under those.
  * `:serial` - run the loop on the calling thread (useful for debugging).

If `order` is specified, it must be a permutation of the *positions* of `indices` - that is, a length-`length(indices)`
vector whose values are a permutation of `1:length(indices)`. The body still receives the corresponding
`indices[order[k]]` value at each step k; only the visit order is altered. Typical usage is heaviest-work-first (so the
longest items dispatch as early as possible): pass `sortperm(weight_per_index; rev = true)`. Under `:greedy`,
`:dynamic`, `:greedy_sticky`, or `:serial`, the iteration visits the positions in `order` as given. Under `:static`,
`order` is reshuffled in place in a round-robin fashion each contiguous chunk gets one item from each weight quantile,
balancing the static partition. Mutates the caller's `order` array.

If `weights` is specified, it must be a length-`length(indices)` vector of non-negative integers giving the estimated
work for each position in `indices`. When `order` is not given, `weights` computes it to be `sortperm(weights; rev =
true)` so the heaviest items dispatch first. When `progress` is also given, each iteration advances the bar by the
visited position's `weights` entry instead of by 1; the caller is responsible for sizing the progress total to
`sum(weights)` so the percentage reflects work done rather than items done. `weights` and `order` are independent and
may be combined: pass both to keep an explicit visit order while reporting work-weighted progress. Mutually exclusive
with `progress_chunk` (the unweighted-throttling alternative).

If this is invoked from inside another parallel loop and `nested` is `false` (the default), `policy` is ignored and the
loop is executed serially. This makes functions safe to compose: parallel at the top level, serial when called from
inside another parallel loop. Pass `nested = true` to opt out of the safety demotion - the loop then uses `policy` as if
at the top level. The caller is responsible for the scratch contract: under `:static`/`:greedy_sticky` each spawned
sub-task has a stable `threadid()`, but multiple outer iterations may be inside their nested loops at the same time, so
per-thread scratch indexed by inner `threadid()` must be local to the outer iteration (not a global per-thread array
shared across all outer iterations).

The `name` is used for [`flame_timed`](@ref) for the whole loop. If the loop is parallel, the whole loop is a line in
the serial flame file, and each iteration is a line in the parallel flame file. The serial file therefore gives a view
of the elapsed time of the top-level loops, to identify what actually matters, and the parallel file shows the internal
breakdown of the computations in each loop. It is sadly impossible to include threaded tasks in the same flamegraph in a
meaningful way (at least not without extending the flamegraph format and visualization).

If `progress` is specified, it is updated for each iteration. If `progress_chunk` is specified, then `indices` must be a
range `1:n`, `order` must be `nothing`, and we throttle calls to the progress bar update to one every `progress_chunk`.

!!! note

    The code inside the loop body should not use any random number generation. Use [`parallel_loop_with_rng`](@ref) if
    random number generation is needed.

```jldoctest
using Random

size = 10

for policy in (:serial, :greedy, :dynamic, :static, :greedy_sticky)
    results = zeros(Int, size)
    parallel_loop_wo_rng(1:size; policy) do index
        results[index] = index
        return nothing
    end
    @assert results == collect(1:size)
end

# output

```
"""
function parallel_loop_wo_rng(  # NOJET
    body::Function,
    indices::AbstractVector{<:Integer};
    name::AbstractString = ".loop",
    policy::Symbol = :greedy_sticky,
    order::Maybe{AbstractVector{<:Integer}} = nothing,
    weights::Maybe{AbstractVector{<:Integer}} = nothing,
    nested::Bool = false,
    progress::Maybe{Progress} = nothing,  # NOLINT
    progress_chunk::Maybe{Integer} = nothing,
)::Nothing
    @assert policy in (:greedy, :static, :dynamic, :greedy_sticky, :serial)

    if weights !== nothing
        @assert length(weights) == length(indices)
        @assert progress_chunk === nothing
        if order === nothing
            order = sortperm(weights; rev = true)  # NOJET
        end
    end

    if order !== nothing
        @assert length(order) == length(indices)
        @assert progress_chunk === nothing
    end

    if progress_chunk !== nothing
        @assert indices[1] == 1
        @assert indices[end] == length(indices)
    end

    base_private_storage = task_local_storage()
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

    if nthreads() == 1
        policy = :serial  # UNTESTED
    else
        if !nested && haskey(base_private_storage, :is_in_parallel_loop)
            progress = nothing  # UNTESTED
            if policy != :serial  # UNTESTED
                policy = :serial  # UNTESTED
            end
        end
    end

    # When `:greedy_sticky` would have at most one iteration per thread, the atomic counter buys nothing - degrade to
    # `:static` (which gives each iteration its own thread directly).
    if policy == :greedy_sticky && length(indices) <= nthreads()
        policy = :static
    end

    # Resolve the iteration order once: under `:static`, round-robin-shuffle `order` in place so each contiguous chunk
    # gets one item from each weight quantile. Under any other policy, the work queue consumes items in the order they
    # appear, so an already-sorted (heaviest-first) `order` directly dispatches the heaviest items earliest. The body
    # always receives `indices[order[k]]` - `view(indices, order)` projects positions to values without allocating.
    if order !== nothing && policy == :static
        round_robin_for_static_chunks!(order, nthreads())
    end
    iteration_indices = order === nothing ? indices : view(indices, order)
    n_index_positions = length(iteration_indices)

    # Per-position progress step. Built after the `:static` round-robin shuffle so `step_per_position[k]` matches the
    # `iteration_indices[k]` actually visited at step `k`. `nothing` when no `weights` were given, in which case the
    # iteration body falls back to `+1` per step (or to `progress_chunk` throttling).
    step_per_position = weights === nothing ? nothing : (order === nothing ? weights : weights[order])

    flame_timed(name; iterations = policy == :serial ? length(indices) : -length(indices)) do
        try
            if progress !== nothing
                disable_logging(Logging.Debug)
                if progress.counter == 0
                    update!(progress, 0; force = true)  # NOJET
                end
            end

            # Advance `progress` by the visited position's contribution: weighted step if `weights` were given, one
            # per item under no chunking, or one per `progress_chunk` items when throttling.
            @inline function advance_progress(position::Integer)::Nothing
                if progress === nothing
                    return nothing
                end
                if step_per_position !== nothing
                    next!(progress; step = Int(step_per_position[position]))  # NOJET
                elseif progress_chunk === nothing
                    next!(progress)  # NOJET
                elseif position % progress_chunk == 0
                    next!(progress; step = progress_chunk)
                end
                return nothing
            end

            if policy == :serial
                for position in 1:n_index_positions
                    body(iteration_indices[position])
                    advance_progress(position)
                end
            else
                @inline function iteration_body(position::Integer)::Nothing
                    index_private_storage = task_local_storage()
                    @assert index_private_storage !== base_private_storage
                    index_private_storage[:is_in_parallel_loop] = true
                    index_flame_stack = [base_flame_stack]
                    index_private_storage[:flame_stack] = index_flame_stack

                    flame_timed(name) do
                        body(iteration_indices[position])
                        advance_progress(position)
                        return nothing
                    end

                    return nothing
                end

                saved_blas_threads = BLAS.get_num_threads()
                BLAS.set_num_threads(1)
                try
                    if policy == :greedy
                        @threads :greedy for position in 1:n_index_positions
                            iteration_body(position)
                        end

                    elseif policy == :static
                        @threads :static for position in 1:n_index_positions
                            iteration_body(position)
                        end

                    elseif policy == :dynamic
                        @threads :dynamic for position in 1:n_index_positions
                            iteration_body(position)
                        end

                    elseif policy == :greedy_sticky
                        # `nthreads()` sticky workers (one per thread) pull positions from a shared atomic counter.
                        # Each iteration runs on its starting thread for the duration of the body, so
                        # `threadid()`-indexed scratch is safe - unlike `:greedy` / `:dynamic` which create
                        # non-sticky tasks.
                        next_position = Atomic{Int}(0)
                        @threads :static for _ in 1:nthreads()
                            while true
                                position = atomic_add!(next_position, 1) + 1
                                if position > n_index_positions
                                    break
                                end
                                iteration_body(position)
                            end
                        end

                    else
                        @assert false
                    end
                finally
                    BLAS.set_num_threads(saved_blas_threads)
                end
            end

        finally
            if progress !== nothing
                disable_logging(Logging.BelowMinLevel)
            end
        end
    end

    if progress !== nothing && progress_chunk !== nothing && length(indices) % progress_chunk > 0
        next!(progress; step = length(indices) % progress_chunk)
    end

    return nothing
end

"""
    parallel_loop_with_rng(
        body::Function,
        indices::AbstractVector{<:Integer};
        name::AbstractString = ".loop",
        policy::Symbol = :greedy,
        order::Maybe{AbstractVector{<:Integer}} = nothing,
        weights::Maybe{AbstractVector{<:Integer}} = nothing,
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

If `order` and/or `weights` are specified, they are forwarded to [`parallel_loop_wo_rng`](@ref) - see there for the
position-permutation semantics, the in-place round-robin shuffle under `:static`, and the work-weighted progress
accounting. The body still receives the same `indices[order[k]]` value at each step k, and the seed is
`seed + indices[order[k]]` so reproducibility is independent of the visit order.

If this is invoked from inside another parallel loop and `nested` is `false` (the default), `policy` is ignored and
the loop is executed serially. This makes functions safe to compose: parallel at the top level, serial when called
from inside another parallel loop. Pass `nested = true` to opt out of the safety demotion - the loop then uses
`policy` as if at the top level. The caller is responsible for the scratch contract: under `:static`/`:greedy_sticky`
each spawned sub-task has a stable `threadid()`, but multiple outer iterations may be inside their nested loops at
the same time, so per-thread scratch indexed by inner `threadid()` must be local to the outer iteration (not a global
per-thread array shared across all outer iterations).

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
using Random

size = 10

function collect_rng(rng::AbstractRNG)::Vector{Float64}
    results = zeros(Float64, size)
    parallel_loop_with_rng(1:size; rng) do index, rng
        results[index] = rand(rng)
    end
    @assert results[1] != results[2]
    return results
end

@assert collect_rng(MersenneTwister(1)) == collect_rng(MersenneTwister(1))

function collect_default_rng()::Vector{Float64}
    results = zeros(Float64, size)
    parallel_loop_with_rng(1:size; seed = 123456, policy = :dynamic) do index, _
        results[index] = rand()
    end
    @assert results[1] != results[2]
    return results
end

@assert collect_default_rng() == collect_default_rng()

# output

```
"""
function parallel_loop_with_rng(  # NOJET
    body::Function,
    indices::AbstractVector{<:Integer};
    name::AbstractString = ".loop",
    policy::Symbol = :greedy_sticky,
    order::Maybe{AbstractVector{<:Integer}} = nothing,
    weights::Maybe{AbstractVector{<:Integer}} = nothing,
    nested::Bool = false,
    progress::Maybe{Progress} = nothing,
    seed::Maybe{Integer} = nothing,
    rng::Maybe{AbstractRNG} = nothing,
)::Nothing
    @assert policy in (:greedy, :static, :dynamic, :greedy_sticky, :serial)

    if seed === nothing
        seed = rand(copy(rng === nothing ? default_rng() : rng), Int64)
    end

    parallel_loop_wo_rng(indices; name, policy, order, weights, nested, progress) do index
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
    if get(ENV, "JULIA_DEBUG", "") == ""
        return false
    end
    logger = Logging.current_logger()
    if logger.min_level <= Logging.Debug
        return true
    end
    stack = stacktrace(backtrace())
    if length(stack) < 4
        caller_module = Main
    else
        caller_frame = stack[4]
        linfo = caller_frame.linfo
        if linfo === nothing
            caller_module = Main
        elseif linfo isa Module
            caller_module = linfo
        elseif linfo isa Core.MethodInstance
            def = linfo.def
            caller_module = def isa Method ? def.module : def isa Module ? def : Main  # NOJET
        else
            caller_module = Main
        end
    end
    return Logging.shouldlog(logger, Logging.Debug, caller_module, group, :check)
end

"""
    DebugProgress(n::Integer; kwargs...)::Maybe{Progress}

Same as `Progress` in `ProgressMeter`, but returns `nothing` if debug is not enabled for the modules calling
`DebugProgress`.
"""
function DebugProgress(n::Integer; group::Maybe{Symbol} = nothing, kwargs...)::Maybe{Progress}
    if is_debug_enabled_for_caller(group)
        return Progress(n; kwargs...)
    else
        return nothing
    end
end

"""
    DebugProgressUnknown(; kwargs...)::Maybe{ProgressUnknown}

Same as `ProgressUnknown` in `ProgressMeter`, but returns `nothing` if debug is not enabled for the modules calling
`DebugProgressUnknown`. Use this for progress bars where the total amount of work is not known in advance (for example,
a planning phase that walks a data set to discover the set of properties to act on).
"""
function DebugProgressUnknown(; group::Maybe{Symbol} = nothing, kwargs...)::Maybe{ProgressUnknown}
    if is_debug_enabled_for_caller(group)
        return ProgressUnknown(; kwargs...)
    else
        return nothing
    end
end

end
