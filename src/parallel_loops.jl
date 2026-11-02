"""
Parallel loops.
"""
module ParallelLoops

export ColumnReader
export DebugProgress
export DebugProgressUnknown
export column_memory_ranges
export column_scratch
export copy_columns!
export fill_column!
export is_in_parallel_loop
export new_column_reader_per_thread
export parallel_foreach_column
export parallel_loop_with_rng
export parallel_loop_wo_rng
export read_column
export read_column!
export start_hang_watchdog

using ..Brief
using ..Types
using ..FlameTime
using ..Handlers
using ..MatrixLayouts

using Base.Threads
using LinearAlgebra
using Logging
using ProgressMeter
using SparseArrays

import ..MatrixLayouts.is_flipped_matrix
import ..MatrixLayouts.is_plain_array
import Base.Threads.maxthreadid
import ProgressMeter.AbstractProgress  # NOLINT
using Random

import Random.default_rng
import SparseArrays.AbstractSparseMatrixCSC  # NOLINT

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
        return nothing
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

# The positions of a parallel loop, claimed in chunks by the threads that run them. A nested loop is listed in
# `OPEN_LOOPS` while it runs, so that the idle workers of the top-level loop can help it. The `run_position` is what a
# helping worker runs for each position. The `n_active` counts the tasks that may still run positions of the loop.
mutable struct ClaimedLoop
    run_position::Function
    n_positions::Int
    next_position::Atomic{Int}
    n_active::Atomic{Int}
    is_idle_event::Event
    failure::Maybe{CapturedException}
end

function ClaimedLoop(run_position::Function, n_positions::Integer, n_active::Integer)::ClaimedLoop
    return ClaimedLoop(run_position, n_positions, Atomic{Int}(1), Atomic{Int}(n_active), Event(), nothing)
end

const OPEN_LOOPS = ClaimedLoop[]
const OPEN_LOOPS_CONDITION = Threads.Condition()
const OPEN_LOOPS_VERSION = Atomic{Int}(0)

# About the cost of blocking a task and waking it up again, which is how long a wait spins before it blocks.
const SPIN_LIMIT_NS = 5_000

# Claim the next position of the `loop`, or return `nothing` if none are left. A fetch-and-add never retries, so a claim
# costs one atomic operation.
function claim_position!(loop::ClaimedLoop)::Maybe{Int}
    position = atomic_add!(loop.next_position, 1)
    return position <= loop.n_positions ? position : nothing
end

# Run claimed positions of the `loop` until none are left. A failure stops the claims of all the tasks.
function run_claimed_positions!(run_position::Function, loop::ClaimedLoop)::Nothing
    try
        while true
            position = claim_position!(loop)
            if position === nothing
                return nothing
            end
            run_position(position)
        end
    catch
        loop.next_position[] = loop.n_positions + 1
        rethrow()
    end
end

# Spin until `is_done()`, for at most `SPIN_LIMIT_NS`. Return whether it is done.
function spin_until(is_done::Function)::Bool
    start_ns = time_ns()
    while !is_done()
        if time_ns() - start_ns > SPIN_LIMIT_NS
            return false
        end
        GC.safepoint()
    end
    return true
end

# Help the nested `loop` as an idle worker. A failure is kept for the caller of the loop to throw.
function help_loop!(loop::ClaimedLoop)::Nothing
    try
        run_claimed_positions!(loop.run_position, loop)
    catch exception
        failure = CapturedException(exception, catch_backtrace())
        lock(OPEN_LOOPS_CONDITION) do
            if loop.failure === nothing
                loop.failure = failure
            end
            return nothing
        end
    finally
        # The caller of the loop holds one count until it stops claiming positions. So the last helper wakes it up. The
        # event stays set, so this works even if the caller only waits later.
        if atomic_sub!(loop.n_active, 1) == 1
            notify(loop.is_idle_event)
        end
    end
    return nothing
end

# Help the open nested loops as an idle worker, until the top-level loop `is_done`. Waiting for a loop to open spins,
# then blocks.
function help_open_loops!(is_done::Atomic{Bool})::Nothing
    while true
        version = OPEN_LOOPS_VERSION[]
        loop = lock(OPEN_LOOPS_CONDITION) do
            for open_loop in OPEN_LOOPS
                if open_loop.next_position[] <= open_loop.n_positions
                    atomic_add!(open_loop.n_active, 1)
                    return open_loop
                end
            end
            return nothing
        end
        if loop !== nothing
            help_loop!(loop)
        elseif is_done[]
            return nothing
        elseif !spin_until(() -> OPEN_LOOPS_VERSION[] != version || is_done[])
            lock(OPEN_LOOPS_CONDITION) do
                while OPEN_LOOPS_VERSION[] == version && !is_done[]
                    wait(OPEN_LOOPS_CONDITION)
                end
                return nothing
            end
        end
    end
end

# Wake up the idle workers, because a loop opened or a top-level loop is done.
function wake_idle_workers()::Nothing
    lock(OPEN_LOOPS_CONDITION) do
        atomic_add!(OPEN_LOOPS_VERSION, 1)
        notify(OPEN_LOOPS_CONDITION)
        return nothing
    end
    return nothing
end

# Run a nested loop. Its caller runs its positions using `run_position`, while idle workers of the top-level loop help
# using `help_position`. The caller returns only after the helpers are done, and throws the failure of any of them. The
# loop stays in `OPEN_LOOPS` until then, so a report of a hang shows a caller waiting for its helpers. Helpers skip it
# once it has no positions left.
function run_nested_loop(run_position::Function, help_position::Function, n_positions::Integer)::Nothing
    loop = ClaimedLoop(help_position, n_positions, 1)
    lock(OPEN_LOOPS_CONDITION) do
        push!(OPEN_LOOPS, loop)
        return nothing
    end
    wake_idle_workers()

    try
        run_claimed_positions!(run_position, loop)
    finally
        if atomic_sub!(loop.n_active, 1) != 1 && !spin_until(() -> loop.n_active[] == 0)
            wait(loop.is_idle_event)
        end
        lock(OPEN_LOOPS_CONDITION) do
            loop_index = findfirst(open_loop -> open_loop === loop, OPEN_LOOPS)
            @assert loop_index !== nothing
            deleteat!(OPEN_LOOPS, loop_index)
            return nothing
        end
    end

    failure = loop.failure
    if failure !== nothing
        throw(failure)
    end
    return nothing
end

# The positions of the `worker_index` under `:static`, using the same split as `@threads :static`.
function static_positions(n_positions::Integer, worker_index::Integer)::UnitRange{Int}
    chunk_size, n_larger_chunks = divrem(n_positions, nthreads())
    first_position = 1 + (worker_index - 1) * chunk_size + min(worker_index - 1, n_larger_chunks)
    last_position = first_position + chunk_size - 1 + (worker_index <= n_larger_chunks ? 1 : 0)
    return first_position:last_position
end

# Run a top-level loop with one worker per thread. A worker runs its own positions, then helps the open nested loops
# until all the positions of the top-level loop are done.
function run_top_level_loop(run_position::Function, n_positions::Integer, policy::Symbol)::Nothing
    is_static = policy == :static
    loop = ClaimedLoop(run_position, n_positions, is_static ? nthreads() : 0)
    is_done = Atomic{Bool}(false)

    function run_worker(worker_index::Integer)::Nothing
        if !is_static
            atomic_add!(loop.n_active, 1)
        end
        try
            if is_static
                for position in static_positions(n_positions, worker_index)
                    run_position(position)
                end
            else
                run_claimed_positions!(run_position, loop)
            end
        finally
            if atomic_sub!(loop.n_active, 1) == 1 && (is_static || loop.next_position[] > n_positions)
                is_done[] = true
                wake_idle_workers()
            end
        end
        help_open_loops!(is_done)
        return nothing
    end

    if policy == :greedy
        @threads :greedy for worker_index in 1:nthreads()
            run_worker(worker_index)
        end
    elseif policy == :dynamic
        @threads :dynamic for worker_index in 1:nthreads()
            run_worker(worker_index)
        end
    else
        @assert policy in (:static, :greedy_sticky)
        @threads :static for worker_index in 1:nthreads()
            run_worker(worker_index)
        end
    end
    return nothing
end

"""
    start_hang_watchdog(; idle_seconds::Real = 60, check_seconds::Real = 5)::Timer

Start a watchdog for hangs, e.g. of a task waiting for a wake-up that never comes. Every `check_seconds`, it samples
the CPU time of the process. If this grows by less than 1% of the elapsed time for `idle_seconds`, the process looks
hung. The watchdog then prints to `stderr` the state of the open nested parallel loops, and the backtraces of all the
tasks. It prints once per stall, and again only after the process works and then stalls again. Close the returned
`Timer` to stop it.

This is for tests and long runs. A process that waits on purpose (e.g. an interactive session) looks hung too.
"""
function start_hang_watchdog(; idle_seconds::Real = 60, check_seconds::Real = 5)::Timer
    stall_seconds = Ref(time())
    stall_cpu_seconds = Ref(process_cpu_seconds())
    is_reported = Ref(false)
    return Timer(check_seconds; interval = check_seconds) do _
        now_seconds = time()
        now_cpu_seconds = process_cpu_seconds()
        if now_cpu_seconds - stall_cpu_seconds[] > 0.01 * (now_seconds - stall_seconds[])
            stall_seconds[] = now_seconds
            stall_cpu_seconds[] = now_cpu_seconds
            is_reported[] = false
        elseif !is_reported[] && now_seconds - stall_seconds[] >= idle_seconds
            is_reported[] = true
            report_hang(now_seconds - stall_seconds[])
            # The report itself uses CPU, which isn't work of the process.
            stall_seconds[] = time()
            stall_cpu_seconds[] = process_cpu_seconds()
        end
        return nothing
    end
end

# The user and system CPU time of the process, in seconds.
function process_cpu_seconds()::Float64
    # The `uv_rusage_t` starts with the user and the system `uv_timeval_t`, each of seconds and microseconds.
    rusage = zeros(Int64, 32)
    @assert ccall(:uv_getrusage, Cint, (Ptr{Int64},), rusage) == 0
    return rusage[1] + rusage[2] / 1e6 + rusage[3] + rusage[4] / 1e6
end

# Print the state of the open nested loops, and the backtraces of all the tasks, after `stalled_seconds` without work.
# This doesn't lock `OPEN_LOOPS_CONDITION`, since a hung task may hold it.
function report_hang(stalled_seconds::Real)::Nothing
    open_loops = copy(OPEN_LOOPS)
    println(
        stderr,
        "The process used almost no CPU for $(round(Int, stalled_seconds)) seconds. It may be hung. ",
        "Open nested loops: $(length(open_loops)), version: $(OPEN_LOOPS_VERSION[]).",
    )
    for (loop_index, loop) in enumerate(open_loops)
        println(
            stderr,
            "Open nested loop $(loop_index): next position: $(loop.next_position[]) of $(loop.n_positions), ",
            "active: $(loop.n_active[]).",
        )
    end
    flush(stderr)
    ccall(:jl_print_task_backtraces, Cvoid, (Cint,), 0)
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
        progress::Maybe{AbstractProgress} = nothing,
    )::Nothing

Run the `body` in parallel, passing it the iteration `index`. A parallel loop runs one worker task per thread, started
by `@threads` with the `policy`. The `policy` is one of:

  - `:greedy_sticky` - sticky workers, started by `@threads :static`, which claim the next index positions from a
    shared atomic counter. Combines the dynamic load balancing of `:greedy` with the threadid stability of `:static`:
    tasks never migrate, so `threadid()`-indexed scratch is safe inside the body. Pair with a heaviest-first `order`
    (or `weights`) to attack the tail-latency problem - heavy items dispatch first, light items fill the tail.
  - `:static` - sticky workers, each running a fixed contiguous chunk of the positions, split as `@threads :static`
    does.
  - `:greedy`, `:dynamic` - workers which claim positions like `:greedy_sticky`, but are *non-sticky* tasks that may
    migrate across threads at yield points, so `threadid()`-indexed scratch is unsafe under those.
  - `:serial` - run the loop on the calling thread (useful for debugging).

Each claim takes a single position, so the positions are claimed in their order.

If `order` is specified, it must be a permutation of the *positions* of `indices` - that is, a length-`length(indices)`
vector whose values are a permutation of `1:length(indices)`. The body still receives the corresponding
`indices[order[k]]` value at each step k; only the visit order is altered. Typical usage is heaviest-work-first (so the
longest items dispatch as early as possible): pass `sortperm(weight_per_index; rev = true)`. Under `:greedy`,
`:dynamic`, `:greedy_sticky`, or `:serial`, the iteration visits the positions in `order` as given. Under `:static`,
`order` is reshuffled in place in a round-robin fashion each contiguous chunk gets one item from each weight quantile,
balancing the static partition. Mutates the caller's `order` array.

If `weights` is specified, it must be a length-`length(indices)` vector of non-negative integers giving the estimated
work for each position in `indices`. When `order` is not given, `weights` computes it to be
`sortperm(weights; rev = true)` so the heaviest items dispatch first. When `progress` is also given, each iteration
advances the bar by the visited position's `weights` entry instead of by 1; the caller is responsible for sizing the
progress total to `sum(weights)` so the percentage reflects work done rather than items done. `weights` and `order` are
independent and may be combined: pass both to keep an explicit visit order while reporting work-weighted progress.
Mutually exclusive with `progress_chunk` (the unweighted-throttling alternative).

If this is invoked from inside another parallel loop and `nested` is `false` (the default), `policy` is ignored and the
loop is executed serially. This makes functions safe to compose: parallel at the top level, serial when called from
inside another parallel loop. Pass `nested = true` to opt out of the safety demotion. The caller then runs the positions
of the loop itself, claiming them as above, and the workers of the top-level loop that have no positions of their own
left help it. This starts no new tasks, and `policy` is ignored. This balances the work of an outer loop whose
iterations differ a lot in size: the last, heavy iterations get the help of the threads that are done with the rest.

The caller is responsible for the scratch contract. Each thread runs one task at a time, so under a sticky top-level
`:static`/`:greedy_sticky` the body of the nested loop has a stable `threadid()`. However, the caller runs some of the
positions in its own task, in the middle of its own outer iteration, so per-thread scratch indexed by inner `threadid()`
must not be the scratch of the outer iteration. The caller keeps its own flame stack and its own `default_rng()` state
across the positions it runs.

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
function parallel_loop_wo_rng(
    body::Function,
    indices::AbstractVector{<:Integer};
    name::AbstractString = ".loop",
    policy::Symbol = :greedy_sticky,
    order::Maybe{AbstractVector{<:Integer}} = nothing,
    weights::Maybe{AbstractVector{<:Integer}} = nothing,
    nested::Bool = false,
    progress::Maybe{AbstractProgress} = nothing,
    progress_chunk::Maybe{Integer} = nothing,
)::Nothing
    @assert policy in (:greedy, :static, :dynamic, :greedy_sticky, :serial)

    if weights !== nothing
        @assert length(weights) == length(indices)
        @assert progress_chunk === nothing
        if order === nothing
            order = sortperm!(Vector{Int}(undef, length(weights)), weights; rev = true)
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
    flame_stack_names = FlameTime.get_flame_stack(base_private_storage)
    loop_name, base_flame_stack = if flame_stack_names === nothing
        (startswith(name, ".") ? name[2:end] : name, nothing)
    else
        @assert !startswith(name, ".") || length(flame_stack_names) > 0
        (startswith(name, ".") ? flame_stack_names[end] * name : name, join(flame_stack_names, ";"))
    end

    # A loop nested in another parallel loop runs serially, without its own progress. With `nested`, its caller runs it
    # with the help of the idle workers of the top-level loop.
    is_in_outer_parallel_loop = nthreads() > 1 && haskey(base_private_storage, :is_in_parallel_loop)
    is_nested_in_parallel_loop = is_in_outer_parallel_loop && !nested
    is_helped_nested_loop = is_in_outer_parallel_loop && nested
    loop_progress = is_nested_in_parallel_loop ? nothing : progress

    # When `:greedy_sticky` would have at most one iteration per thread, the atomic counter buys nothing - degrade to
    # `:static` (which gives each iteration its own thread directly).
    loop_policy = if nthreads() == 1 || is_nested_in_parallel_loop
        :serial
    elseif policy == :greedy_sticky && length(indices) <= nthreads()
        :static
    else
        policy
    end

    # Resolve the iteration order once: under `:static`, round-robin-shuffle `order` in place so each contiguous chunk
    # gets one item from each weight quantile. Under any other policy, and in a helped nested loop, the positions are
    # claimed in the order they appear, so an already-sorted (heaviest-first) `order` directly dispatches the heaviest
    # items earliest. The body always receives `indices[order[k]]` - `view(indices, order)` projects positions to values
    # without allocating.
    if order !== nothing && loop_policy == :static && !is_helped_nested_loop
        round_robin_for_static_chunks!(order, nthreads())
    end
    iteration_indices = order === nothing ? indices : view(indices, order)
    n_index_positions = length(iteration_indices)

    # Per-position progress step. Built after the `:static` round-robin shuffle so `step_per_position[k]` matches the
    # `iteration_indices[k]` actually visited at step `k`. `nothing` when no `weights` were given, in which case the
    # iteration body falls back to `+1` per step (or to `progress_chunk` throttling).
    step_per_position = weights === nothing ? nothing : (order === nothing ? weights : weights[order])

    # A negative count marks the whole of a parallel loop, top-level or nested, so flame timing scales its iterations.
    flame_timed(loop_name; iterations = loop_policy == :serial ? length(indices) : -length(indices)) do
        try
            if loop_progress !== nothing
                disable_logging(Logging.Debug)
                if loop_progress.counter == 0
                    update!(loop_progress, 0; force = true)  # NOJET
                end
            end

            # Advance `loop_progress` by the visited position's contribution: weighted step if `weights` were given,
            # one per item under no chunking, or one per `progress_chunk` items when throttling.
            @inline function advance_progress(position::Integer)::Nothing
                if loop_progress === nothing
                    return nothing
                end
                if step_per_position !== nothing
                    next!(loop_progress; step = Int(step_per_position[position]))
                elseif progress_chunk === nothing
                    next!(loop_progress)
                elseif position % progress_chunk == 0
                    next!(loop_progress; step = progress_chunk)
                end
                return nothing
            end

            if loop_policy == :serial
                for position in 1:n_index_positions
                    body(iteration_indices[position])
                    advance_progress(position)
                end
            else
                @inline function iteration_body(position::Integer)::Nothing
                    index_private_storage = task_local_storage()
                    @assert index_private_storage !== base_private_storage
                    index_private_storage[:is_in_parallel_loop] = true
                    # The task keeps its flame stack between positions, since `flame_timed` pops what it pushes.
                    index_flame_stack = get(index_private_storage, :flame_stack, nothing)
                    if !(
                        index_flame_stack isa Vector{Maybe{AbstractString}} &&
                        length(index_flame_stack) == 1 &&
                        index_flame_stack[1] === base_flame_stack
                    )
                        index_private_storage[:flame_stack] = Maybe{AbstractString}[base_flame_stack]
                    end

                    flame_timed(loop_name) do
                        body(iteration_indices[position])
                        advance_progress(position)
                        return nothing
                    end

                    return nothing
                end

                if is_helped_nested_loop
                    # The caller runs positions in its own task. It keeps its own flame stack and random generator.
                    @inline function caller_iteration_body(position::Integer)::Nothing
                        saved_flame_stack = base_private_storage[:flame_stack]
                        saved_rng = copy(default_rng())
                        try
                            base_private_storage[:flame_stack] = [base_flame_stack]
                            flame_timed(loop_name) do
                                body(iteration_indices[position])
                                advance_progress(position)
                                return nothing
                            end
                        finally
                            base_private_storage[:flame_stack] = saved_flame_stack
                            copy!(default_rng(), saved_rng)
                        end
                        return nothing
                    end

                    run_nested_loop(caller_iteration_body, iteration_body, n_index_positions)

                else
                    saved_blas_threads = BLAS.get_num_threads()
                    BLAS.set_num_threads(1)
                    try
                        run_top_level_loop(iteration_body, n_index_positions, loop_policy)
                    finally
                        BLAS.set_num_threads(saved_blas_threads)
                    end
                end
            end

        finally
            if loop_progress !== nothing
                disable_logging(Logging.BelowMinLevel)
            end
        end
    end

    if loop_progress !== nothing && progress_chunk !== nothing && length(indices) % progress_chunk > 0
        next!(loop_progress; step = length(indices) % progress_chunk)
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
        progress::Maybe{AbstractProgress} = nothing,
        seed::Maybe{Integer} = nothing,
        rng::Maybe{AbstractRNG} = nothing
    )::Nothing

Run the `body` in parallel, passing it the iteration `index` and a separate `rng` that is seeded to a reproducible
state regardless of the allocation of tasks to threads. A copy of this `rng` is given to each iteration, after being
reset to `seed + index` for reproducibility. If no `seed` is specified, it is just sampled `rng` before the loop starts.
If the `rng` isn't given, then this uses (and sets for each iteration) the `default_rng()`. In this case passing it to
the body is redundant but is still done for consistency.

The `policy` is one of those of [`parallel_loop_wo_rng`](@ref). If it is `:serial`, then the loop is not run in parallel
(useful for debugging).

If `order` and/or `weights` are specified, they are forwarded to [`parallel_loop_wo_rng`](@ref) - see there for the
position-permutation semantics, the in-place round-robin shuffle under `:static`, and the work-weighted progress
accounting. The body still receives the same `indices[order[k]]` value at each step k, and the seed is
`seed + indices[order[k]]` so reproducibility is independent of the visit order.

If this is invoked from inside another parallel loop and `nested` is `false` (the default), `policy` is ignored and
the loop is executed serially. This makes functions safe to compose: parallel at the top level, serial when called
from inside another parallel loop. Pass `nested = true` to have the caller run the loop with the help of the idle
workers of the top-level loop - see [`parallel_loop_wo_rng`](@ref) for how, and for the scratch contract. The caller
keeps its own `default_rng()` state, even though the positions it runs reseed it.

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
        return results[index] = rand(rng)
    end
    @assert results[1] != results[2]
    return results
end

@assert collect_rng(MersenneTwister(1)) == collect_rng(MersenneTwister(1))

function collect_default_rng()::Vector{Float64}
    results = zeros(Float64, size)
    parallel_loop_with_rng(1:size; seed = 123456, policy = :dynamic) do index, _
        return results[index] = rand()
    end
    @assert results[1] != results[2]
    return results
end

@assert collect_default_rng() == collect_default_rng()

# output

```
"""
function parallel_loop_with_rng(
    body::Function,
    indices::AbstractVector{<:Integer};
    name::AbstractString = ".loop",
    policy::Symbol = :greedy_sticky,
    order::Maybe{AbstractVector{<:Integer}} = nothing,
    weights::Maybe{AbstractVector{<:Integer}} = nothing,
    nested::Bool = false,
    progress::Maybe{AbstractProgress} = nothing,
    seed::Maybe{Integer} = nothing,
    rng::Maybe{AbstractRNG} = nothing,
)::Nothing
    @assert policy in (:greedy, :static, :dynamic, :greedy_sticky, :serial)

    loop_seed = seed === nothing ? rand(copy(rng === nothing ? default_rng() : rng), Int64) : seed

    parallel_loop_wo_rng(indices; name, policy, order, weights, nested, progress) do index
        if rng === nothing
            iteration_rng = default_rng()
        else
            iteration_rng = copy(rng)
        end
        Random.seed!(iteration_rng, loop_seed + index)
        body(index, iteration_rng)
        return nothing
    end

    return nothing
end

"""
    abstract type ColumnReader end

Reads the columns of a matrix one at a time, for a loop that visits some of its columns. Each thread needs its own
reader, from [`new_column_reader_per_thread`](@ref). The columns are read by [`read_column`](@ref).
"""
abstract type ColumnReader end

# Each reader holds the `matrix` it reads, so a loop given readers can check they are its own.

# Reads the columns of plain storage as views.
struct PlainColumnReader{A <: AbstractMatrix, M <: AbstractMatrix} <: ColumnReader
    matrix::A
    storage::M
end

# Reads the columns of plain storage, which have gaps, by copying them into a contiguous buffer.
struct CopiedColumnReader{A <: AbstractMatrix, M <: AbstractMatrix, T} <: ColumnReader
    matrix::A
    storage::M
    buffer::Vector{T}
end

# Reads the columns of a lazy matrix through its column scratch.
struct LazyColumnReader{A <: AbstractMatrix, M <: AbstractMatrix, S} <: ColumnReader
    matrix::A
    lazy_matrix::M
    scratch::S
end

# Reads the columns of a view of a matrix through a reader of its parent. The `buffer` holds the column of a view of
# some of the rows: a `Vector` for a contiguous dense column, a `SparseVector` for a sparse column, or `nothing`
# otherwise. For sparse rows that were sorted, it is a `NamedTuple` of the `column`, the `order` that sorted the rows,
# and the `renumbered` buffer.
struct ViewColumnReader{
    A <: AbstractMatrix,
    R <: ColumnReader,
    I <: AbstractVector{<:Integer},
    C <: AbstractVector{<:Integer},
    B <: Union{Nothing, Vector, SparseVector, NamedTuple},
} <: ColumnReader
    matrix::A
    parent_reader::R
    parent_row_indices::I
    parent_column_indices::C
    buffer::B
end

"""
    new_column_reader_per_thread(
        matrix::AbstractMatrix;
        contiguous::Bool = false,
        policy::Symbol = :greedy_sticky,
        nested::Bool = false,
    )::AbstractVector{<:ColumnReader}

A [`ColumnReader`](@ref) of the `matrix` for each thread, indexed by `threadid()`, for a parallel loop with the
`policy`. This is useful when each iteration of the loop reads several columns, e.g. the columns of one group. The
`matrix` and the `contiguous` flag are handled as in [`parallel_foreach_column`](@ref), which reads its columns through
these readers.

The readers are created before the loop, so the `policy` must keep each task on its thread (`:greedy_sticky`, `:static`
or `:serial`). If the loop runs serially, all threads share one reader. A loop inside another parallel loop runs
serially, unless it is `nested` (see [`parallel_loop_wo_rng`](@ref)). Set `nested` for the readers of such a loop. If
this is invoked from inside another parallel loop and the readers need buffers, it invokes the
[`GLOBAL_INEFFICIENT_ACTION_HANDLER`](@ref). To avoid this, create them before the other loop.

```jldoctest
using SparseArrays

matrix = sparse([0 1 2; 3 0 4])
columns_per_group = [[1, 3], [2]]
n_stored_per_group = zeros(Int, 2)
column_reader_per_thread = new_column_reader_per_thread(matrix)
parallel_loop_wo_rng(1:2) do group_index
    column_reader = column_reader_per_thread[Threads.threadid()]
    for column_index in columns_per_group[group_index]
        n_stored_per_group[group_index] += length(nonzeros(read_column(column_reader, column_index)))
    end
    return nothing
end
@assert n_stored_per_group == [3, 1]

# output

```
"""
function new_column_reader_per_thread(
    matrix::AbstractMatrix;
    contiguous::Bool = false,
    policy::Symbol = :greedy_sticky,
    nested::Bool = false,
)::AbstractVector{<:ColumnReader}
    @assert policy in (:greedy_sticky, :static, :serial)
    is_serial = is_serial_loop(policy; nested)

    canonical_matrix = canonical_array(matrix)
    if canonical_matrix isa SubArray && is_read_by_parent_columns(canonical_matrix)
        parent_reader_per_thread =
            new_parent_column_reader_per_thread(canonical_matrix, contiguous, policy, nested, is_serial)
        if parent_reader_per_thread !== nothing
            return new_view_column_reader_per_thread(
                matrix,
                canonical_matrix,
                parent_reader_per_thread,
                contiguous,
                is_serial,
            )
        end
    end

    scratch = column_scratch(canonical_matrix)
    if scratch !== nothing
        # A lazy matrix is read through its columns, so it is column-major.
        @assert major_axis(canonical_matrix) == Columns
        return new_lazy_column_reader_per_thread(canonical_matrix, scratch, matrix, is_serial)
    end

    if major_axis(canonical_matrix) === nothing
        handle_abnormal(GLOBAL_INEFFICIENT_ACTION_HANDLER) do
            return """
                copying a matrix without a layout
                for matrix: $(brief(matrix))
                """
        end
    end

    if canonical_matrix isa BitMatrix
        # A `BitMatrix` is read through views of its columns. Its `plain_array` would copy all of it.
        plain_storage = canonical_matrix
    elseif is_flipped_matrix(canonical_matrix) && !is_plain_array(parent(canonical_matrix))
        # A flip of a lazy matrix (or of a view) is relayouted through its columns. Its `plain_array` would hold all of
        # it in memory.
        @assert_matrix(matrix, Columns)
        plain_storage = relayout(matrix)
    else
        plain_storage = plain_array(matrix)
    end
    @assert_matrix(plain_storage, Columns)
    storage = issparse(plain_storage) && major_axis(plain_storage) != Columns ? relayout(plain_storage) : plain_storage
    if contiguous && !issparse(storage) && size(storage, 2) > 0 && !is_contiguous_array(view(storage, :, 1))
        create_copied_reader =
            () -> CopiedColumnReader(matrix, storage, Vector{eltype(storage)}(undef, size(storage, 1)))
        return new_scratch_per_thread(create_copied_reader(), create_copied_reader, matrix, is_serial)
    end
    return fill(PlainColumnReader(matrix, storage), maxthreadid())
end

"""
    read_column(reader::ColumnReader, column_index::Integer)::AbstractVector

Return the values of the column `column_index` through the `reader`. The result is valid until the next call with the
same `reader`, and must not be modified. See [`parallel_foreach_column`](@ref) for the kinds of columns it may return.
"""
function read_column end

function read_column(reader::PlainColumnReader, column_index::Integer)::AbstractVector
    return view(reader.storage, :, column_index)
end

function read_column(reader::CopiedColumnReader, column_index::Integer)::AbstractVector
    return copyto!(reader.buffer, view(reader.storage, :, column_index))
end

function read_column(reader::LazyColumnReader, column_index::Integer)::AbstractVector
    return read_column!(reader.scratch, reader.lazy_matrix, column_index)  # NOLINT # NOJET
end

function read_column(reader::ViewColumnReader, column_index::Integer)::AbstractVector
    parent_column = read_column(reader.parent_reader, reader.parent_column_indices[column_index])
    return gathered_view_rows(parent_column, reader.parent_row_indices, reader.buffer)
end

# Whether the columns of a `matrix_view` are read through the columns of its parent, instead of through `plain_array`. A
# strided view of an `Array` is plain storage by itself. A sparse view of some of the rows needs them to have no
# repetitions. Rows in any other order are sorted once, and the stored values of each column are reordered.
function is_read_by_parent_columns(matrix_view::SubArray)::Bool
    if ndims(matrix_view) != 2 || (parent(matrix_view) isa Array && matrix_view isa StridedArray)
        return false
    end
    row_indices, column_indices = parentindices(matrix_view)
    if !(row_indices isa AbstractVector{<:Integer} && column_indices isa AbstractVector{<:Integer})
        return false
    end
    return !issparse(parent(matrix_view)) ||
           row_indices isa Base.Slice ||
           issorted(row_indices; lt = <=) ||
           allunique(row_indices)
end

# The readers of the columns of the parent of a `matrix_view`, if the parent is plain storage or a lazy matrix. Reading
# the columns of any other parent would copy all of it. Such a view is read through `plain_array` instead, which copies
# only its entries.
function new_parent_column_reader_per_thread(
    matrix_view::SubArray,
    contiguous::Bool,
    policy::Symbol,
    nested::Bool,
    is_serial::Bool,
)::Maybe{AbstractVector{<:ColumnReader}}
    parent_matrix = parent(matrix_view)
    if parent_matrix isa Union{Array, BitMatrix, SparseMatrixCSC}
        is_all_rows = parentindices(matrix_view)[1] isa Base.Slice
        return new_column_reader_per_thread(parent_matrix; contiguous = contiguous && is_all_rows, policy, nested)
    end
    scratch = column_scratch(parent_matrix)
    if scratch === nothing
        return nothing
    end
    return new_lazy_column_reader_per_thread(parent_matrix, scratch, parent_matrix, is_serial)
end

# The readers of the columns of the `matrix`, whose lazy `canonical_matrix` is read, the first of them using the
# `scratch`. The `matrix` is reported if the scratch is allocated inside another parallel loop.
function new_lazy_column_reader_per_thread(
    canonical_matrix::AbstractMatrix,
    scratch::Any,
    matrix::AbstractMatrix,
    is_serial::Bool,
)::AbstractVector{<:ColumnReader}
    create_lazy_reader = () -> LazyColumnReader(matrix, canonical_matrix, column_scratch(canonical_matrix))
    first_reader = LazyColumnReader(matrix, canonical_matrix, scratch)
    return new_scratch_per_thread(first_reader, create_lazy_reader, matrix, is_serial)
end

# The readers of the columns of the `matrix`, whose canonical `matrix_view` is read through the readers of its parent.
function new_view_column_reader_per_thread(
    matrix::AbstractMatrix,
    matrix_view::SubArray,
    parent_reader_per_thread::AbstractVector{<:ColumnReader},
    contiguous::Bool,
    is_serial::Bool,
)::AbstractVector{<:ColumnReader}
    view_row_indices, column_indices = parentindices(matrix_view)
    row_indices = view_row_indices isa Base.LogicalIndex ? collect(view_row_indices) : view_row_indices
    is_all_rows = row_indices isa Base.Slice
    parent_matrix = parent(matrix_view)

    # Sparse rows in any other order are gathered in increasing order, then renumbered by the `order` that sorted them.
    order = is_all_rows || !issparse(parent_matrix) || issorted(row_indices; lt = <=) ? nothing : sortperm(row_indices)
    reader_row_indices = order === nothing ? row_indices : row_indices[order]

    create_buffer = nothing
    if !is_all_rows && issparse(parent_matrix)
        index_type = SparseArrays.indtype(parent_matrix)
        value_type = eltype(parent_matrix)
        n_rows = length(row_indices)
        create_column = () -> SparseVector(n_rows, Vector{index_type}(), Vector{value_type}())
        if order === nothing
            create_buffer = create_column
        else
            create_buffer =
                () -> (column = create_column(), order = order, renumbered = Tuple{index_type, value_type}[])
        end
    elseif !is_all_rows && contiguous
        create_buffer = () -> Vector{eltype(parent_matrix)}(undef, length(row_indices))
    end

    buffer_per_thread = if create_buffer === nothing
        fill(nothing, maxthreadid())
    else
        new_scratch_per_thread(create_buffer(), create_buffer, matrix_view, is_serial)
    end

    return [
        ViewColumnReader(
            matrix,
            parent_reader_per_thread[thread_index],
            reader_row_indices,
            column_indices,
            buffer_per_thread[thread_index],
        ) for thread_index in 1:maxthreadid()
    ]
end

# The values of a `parent_column` in the rows of a view, given by the `row_indices` of the parent.
function gathered_view_rows(parent_column::AbstractVector, ::Base.Slice, ::Nothing)::AbstractVector
    return parent_column
end

function gathered_view_rows(
    parent_column::AbstractVector,
    row_indices::AbstractVector{<:Integer},
    ::Nothing,
)::AbstractVector
    return view(parent_column, row_indices)
end

function gathered_view_rows(
    parent_column::AbstractVector,
    row_indices::AbstractVector{<:Integer},
    buffer::Vector,
)::AbstractVector
    return copyto!(buffer, view(parent_column, row_indices))
end

# The method for a sparse column, whose `buffer` is a `SparseVector`, is in `SparseColumns`.

"""
    parallel_foreach_column(
        visit::Function,
        matrix::AbstractMatrix;
        columns::AbstractVector{<:Integer} = 1:size(matrix, 2),
        contiguous::Bool = false,
        name::AbstractString = ".columns",
        policy::Symbol = :greedy_sticky,
        progress::Maybe{AbstractProgress} = nothing,
        progress_chunk::Maybe{Integer} = nothing,
        nested::Bool = false,
        column_reader_per_thread::Maybe{AbstractVector{<:ColumnReader}} = nothing,
    )::Nothing

Run `visit(position, column_index, column)` in parallel for each of the `columns` of the `matrix`. The `position` is the
index of the `column_index` in `columns`, so `visit` can write its results by position. This is [`plain_array`](@ref)
one column at a time: the `column` holds the values of the column in plain storage, so `visit` needs no `plain_array` of
its own. The `column` is valid only during the call to `visit`, and must not be modified.

A dense `column` is a view of a column of plain storage, or a buffer holding the column. A sparse `column` is read
through `nonzeroinds` and `nonzeros`. It is a view of a column of a `SparseMatrixCSC`, or a `SparseVector` holding the
column.

A lazy matrix (which reads its data on demand) is read one column at a time through [`column_scratch`](@ref) and
[`read_column!`](@ref), so all of it is never held in memory.

A view of some of the rows and columns of plain storage or of a lazy matrix (e.g., by a mask) is read through the
columns of its parent. Only the parent's columns in the view are read, and a view of a lazy matrix is never held in
memory either. A dense column of a view of some of the rows is a view of the parent's column. A sparse one is a
`SparseVector` holding the parent's stored values in these rows. The rows may be in any order, but a sparse view of
repeated rows is read through `plain_array`. So is a view of any other matrix (e.g. a transpose), which copies only the
entries of the view.

A `BitMatrix` is read through views of its columns. Any other matrix is read once through `plain_array`, which copies
nothing for plain storage, read-only and named wrappers, and strided views. A matrix stored in rows (e.g. a transpose)
is read against the grain, which invokes the [`GLOBAL_INEFFICIENT_ACTION_HANDLER`](@ref). A sparse one is relayouted to
columns first. So is a flip of a lazy matrix (or of a view), which is relayouted through its columns, so all of it is
never held in memory twice. A matrix without a layout (one that isn't strided) is copied, which also invokes the
handler.

Set `contiguous` to copy each column with gaps (of a strided view or of a transpose) into a buffer, for code that needs
contiguous memory.

Buffers are allocated before the loop, one per thread, and dropped after it. They are indexed by `threadid()`, so the
`policy` must keep each task on its thread (`:greedy_sticky`, `:static` or `:serial`). If this is invoked from inside
another parallel loop, it runs serially (see [`parallel_loop_wo_rng`](@ref)) and allocates a single buffer. If it is
`nested`, the idle threads help it, and it allocates a buffer per thread. Either way, this allocates inside the other
loop, so it invokes the [`GLOBAL_INEFFICIENT_ACTION_HANDLER`](@ref).

A loop that runs this many times over the same `matrix` (e.g. once per group of columns) can create the readers once
with [`new_column_reader_per_thread`](@ref), and pass them as `column_reader_per_thread`. These must be readers of this
`matrix`, created for the same `policy` and `nested`, and `contiguous` is then ignored. A `nested` loop given readers
created before the other loop allocates nothing. The caller runs some of its columns in its own task, through the reader
of its own thread, so these readers must not be the ones the iteration of the other loop uses.

The columns are visited in order, one at a time. So a destination written in column order (e.g. a packed matrix) holds
few columns at a time, and lazy or memory-mapped data is read from start to end. The `name`, the `policy`, the `nested`
flag, the `progress` and the `progress_chunk` are passed to [`parallel_loop_wo_rng`](@ref).

```jldoctest
using SparseArrays

matrix = sparse([0 1 2; 3 0 4])
n_stored_per_column = zeros(Int, 2)
parallel_foreach_column(matrix; columns = [3, 1]) do position, column_index, column
    n_stored_per_column[position] = length(nonzeros(column))
    return nothing
end
@assert n_stored_per_column == [2, 1]

# output

```
"""
function parallel_foreach_column(
    visit::Function,
    matrix::AbstractMatrix;
    columns::AbstractVector{<:Integer} = 1:size(matrix, 2),
    contiguous::Bool = false,
    name::AbstractString = ".columns",
    policy::Symbol = :greedy_sticky,
    progress::Maybe{AbstractProgress} = nothing,
    progress_chunk::Maybe{Integer} = nothing,
    nested::Bool = false,
    column_reader_per_thread::Maybe{AbstractVector{<:ColumnReader}} = nothing,
)::Nothing
    @assert policy in (:greedy_sticky, :static, :serial)
    @assert column_reader_per_thread === nothing || column_reader_per_thread[1].matrix === matrix
    if isempty(columns)
        return nothing
    end

    if column_reader_per_thread === nothing
        column_reader_per_thread = new_column_reader_per_thread(matrix; contiguous, policy, nested)
    end
    foreach_column_through_readers(
        visit,
        column_reader_per_thread,
        columns;
        name,
        policy,
        nested,
        progress,
        progress_chunk,
    )
    return nothing
end

# Whether a parallel loop with the `policy` runs serially, which is also the case inside another parallel loop, unless
# it is `nested`.
function is_serial_loop(policy::Symbol; nested::Bool = false)::Bool
    return policy == :serial || nthreads() == 1 || (is_in_parallel_loop() && !nested)
end

# The scratch of each thread, starting with the `first_scratch`, the rest made by `create_scratch`. A serial loop shares
# the `first_scratch` between all threads. Inside another parallel loop, this allocation is inside that loop, which
# invokes the `GLOBAL_INEFFICIENT_ACTION_HANDLER`.
function new_scratch_per_thread(
    first_scratch::Any,
    create_scratch::Function,
    matrix::AbstractMatrix,
    is_serial::Bool,
)::AbstractVector
    if is_in_parallel_loop()
        handle_abnormal(GLOBAL_INEFFICIENT_ACTION_HANDLER) do
            return """
                allocating column scratch inside a parallel loop
                for matrix: $(brief(matrix))
                """
        end
    end
    if is_serial
        return fill(first_scratch, maxthreadid())
    else
        return [thread_index == 1 ? first_scratch : create_scratch() for thread_index in 1:maxthreadid()]
    end
end

# Run `visit(position, column_index, column)` for each of the `columns`, reading the column through the reader of the
# current thread.
function foreach_column_through_readers(
    visit::Function,
    column_reader_per_thread::AbstractVector{<:ColumnReader},
    columns::AbstractVector{<:Integer};
    name::AbstractString,
    policy::Symbol,
    nested::Bool,
    progress::Maybe{AbstractProgress},
    progress_chunk::Maybe{Integer},
)::Nothing
    parallel_loop_wo_rng(1:length(columns); name, policy, nested, progress, progress_chunk) do position
        column_index = columns[position]
        visit(position, column_index, read_column(column_reader_per_thread[threadid()], column_index))
        return nothing
    end
    return nothing
end

"""
    column_scratch(matrix::AbstractMatrix)::Any

The scratch [`parallel_foreach_column`](@ref) needs for reading the columns of the `matrix` one at a time through
[`read_column!`](@ref), or `nothing` if it reads the matrix through [`plain_array`](@ref) instead. This is `nothing`
except for a lazy matrix type (which reads its data on demand), which adds a method for both functions. Such a type
returns `nothing` when all its data is already in memory, since `plain_array` then returns it without reading it.
"""
function column_scratch(::AbstractMatrix)::Any
    return nothing
end

"""
    read_column!(scratch::Any, matrix::AbstractMatrix, column_index::Integer)::AbstractVector

Return the values of the column `column_index` of a lazy `matrix`, using the `scratch` from [`column_scratch`](@ref).
The result is valid until the next call with the same `scratch`. See [`parallel_foreach_column`](@ref) for the kinds of
columns it may return.
"""
function read_column! end

"""
    column_memory_ranges(matrix::AbstractMatrix, column_index::Integer)::Maybe{Tuple{Vararg{UnitRange{UInt}}}}

The ranges of memory addresses that hold the values of the column `column_index` of the `matrix`, or `nothing` if they
aren't known. There is one range per array that holds the data of the matrix, in the same order for every column: one
for a dense matrix, and two for a sparse matrix (its `rowval`, then its `nzval`). A lazy matrix type adds a method
returning the memory of the encoded data of the column.
"""
function column_memory_ranges(::AbstractMatrix, ::Integer)::Nothing
    return nothing
end

function column_memory_ranges(matrix::DenseMatrix{T}, column_index::Integer)::Tuple{UnitRange{UInt}} where {T}
    n_rows = size(matrix, 1)
    first_address = UInt(pointer(matrix, (column_index - 1) * n_rows + 1))
    return (first_address:(first_address + n_rows * sizeof(T) - 1),)
end

function column_memory_ranges(
    matrix::SparseMatrixCSC{T, I},
    column_index::Integer,
)::NTuple{2, UnitRange{UInt}} where {T, I}
    column_pointers = SparseArrays.getcolptr(matrix)
    first_stored = Int(column_pointers[column_index])
    n_stored = Int(column_pointers[column_index + 1]) - first_stored
    first_row_address = UInt(pointer(rowvals(matrix), first_stored))
    first_value_address = UInt(pointer(nonzeros(matrix), first_stored))
    return (
        first_row_address:(first_row_address + n_stored * sizeof(I) - 1),
        first_value_address:(first_value_address + n_stored * sizeof(T) - 1),
    )
end

"""
    fill_column!(fill::Function, matrix::AbstractMatrix, column_index::Integer)::Nothing

Write the column `column_index` of the `matrix` by calling `fill(column)`, which must set every entry of the `column`
without keeping a reference to it. By default, the `column` is a view of the column of the `matrix`. A matrix type that
can't be written through such a view (e.g., one that encodes each column as it is written) adds a method, which may
pass a buffer instead. This is the write counterpart of [`read_column!`](@ref), used by [`copy_columns!`](@ref).
"""
function fill_column!(fill::Function, matrix::AbstractMatrix, column_index::Integer)::Nothing
    fill(view(matrix, :, column_index))
    return nothing
end

"""
    copy_columns!(
        destination::AbstractMatrix,
        source::AbstractMatrix;
        column_offset::Integer = 0,
    )::Nothing
    copy_columns!(
        colptr::AbstractVector{<:Integer},
        rowval::AbstractVector{<:Integer},
        nzval::AbstractVector,
        source::AbstractMatrix;
        column_offset::Integer = 0,
        nnz_offset::Integer = 0,
    )::Nothing

Copy the columns of the `source` into the columns of the `destination` that follow its first `column_offset` columns.
The columns are copied in parallel through [`parallel_foreach_column`](@ref), so a lazy `source` is never held in
memory as a whole. The values are converted to the element type of the destination.

A dense `destination` is written through [`fill_column!`](@ref). A sparse `source` column is written into it together
with its implicit zeros.

A sparse `destination` is given as its `colptr`, `rowval` and `nzval` (or as a `SparseMatrixCSC`, whose own vectors are
used). The `source` must be sparse too. It may be a view of a sparse matrix, or a row-major one, which is relayouted
first. Its stored values are written after the first `nnz_offset` stored values of the
`destination`, and its `colptr` is offset to match. This also writes the `colptr` entry after the last column of the
`source`. That entry is where the next part of the `destination` starts, so copying several sources into consecutive
parts of the `destination` fills all of its `colptr`.

The `destination` must have the same number of rows as the `source`, and room for its columns (and stored values) after
the offsets.

```jldoctest
using SparseArrays

source = sparse([0 1; 2 0])
destination = zeros(Float32, 2, 3)
copy_columns!(destination, source; column_offset = 1)
@assert destination == Float32[0 0 1; 0 2 0]

colptr = zeros(Int, 3)
rowval = zeros(Int, 2)
nzval = zeros(Int, 2)
copy_columns!(colptr, rowval, nzval, source)
@assert SparseMatrixCSC(2, 2, colptr, rowval, nzval) == source

# output

```
"""
function copy_columns!(destination::AbstractMatrix, source::AbstractMatrix; column_offset::Integer = 0)::Nothing
    @assert size(destination, 1) == size(source, 1)
    @assert 0 <= column_offset && column_offset + size(source, 2) <= size(destination, 2)
    columns = (column_offset + 1):(column_offset + size(source, 2))
    destination_axis = major_axis(destination)
    source_axis = major_axis(source)

    if destination_axis == Rows && source_axis == Rows
        # The columns of both are the rows of their flips.
        copy_columns!(view(flip(destination), columns, :), flip(source))
        return nothing
    end

    if !issparse(source) && source_axis !== nothing && destination_axis == other_axis(source_axis)
        relayout_destination = relayout_columns(destination, columns)
        if relayout_destination !== nothing
            relayout!(relayout_destination, source)
            return nothing
        end
    end

    parallel_foreach_column(source; name = "copy_columns!") do _, column_index, column
        fill_column!(destination, column_offset + column_index) do destination_column
            copy_column!(destination_column, column)
            return nothing
        end
        return nothing
    end
    return nothing
end

# The `columns` of the `destination` as a matrix [`relayout!`](@ref) can write, or `nothing` if it can't write them.
function relayout_columns(destination::AbstractMatrix, columns::UnitRange{<:Integer})::Maybe{AbstractMatrix}
    if destination isa DenseMatrix
        return columns == axes(destination, 2) ? destination : view(destination, :, columns)
    end

    flipped_destination = flip(destination)
    if flipped_destination isa DenseMatrix
        return columns == axes(destination, 2) ? destination : flip(view(flipped_destination, columns, :))
    end

    return nothing
end

function copy_columns!(
    destination::SparseMatrixCSC,
    source::AbstractMatrix;
    column_offset::Integer = 0,
    nnz_offset::Integer = 0,
)::Nothing
    @assert size(destination, 1) == size(source, 1)
    copy_columns!(
        SparseArrays.getcolptr(destination),
        rowvals(destination),
        nonzeros(destination),
        source;
        column_offset,
        nnz_offset,
    )
    return nothing
end

function copy_columns!(
    colptr::AbstractVector{<:Integer},
    rowval::AbstractVector{<:Integer},
    nzval::AbstractVector,
    source::AbstractMatrix;
    column_offset::Integer = 0,
    nnz_offset::Integer = 0,
)::Nothing
    @assert issparse(source)
    if major_axis(source) == Rows
        # Copying a row-major source into columns is a relayout.
        copy_columns!(colptr, rowval, nzval, relayout(source); column_offset, nnz_offset)
        return nothing
    end

    copy_stored_columns!(colptr, rowval, nzval, source, stored_colptr(source); column_offset, nnz_offset)
    return nothing
end

# Copy the stored values of a column-major sparse `source`, whose `colptr` is `source_colptr`, as in `copy_columns!`.
function copy_stored_columns!(
    colptr::AbstractVector{<:Integer},
    rowval::AbstractVector{<:Integer},
    nzval::AbstractVector,
    source::AbstractMatrix,
    source_colptr::AbstractVector{<:Integer};
    column_offset::Integer = 0,
    nnz_offset::Integer = 0,
)::Nothing
    n_columns = size(source, 2)
    @assert 0 <= column_offset && column_offset + n_columns + 1 <= length(colptr)
    @assert length(rowval) == length(nzval)
    @assert 0 <= nnz_offset && nnz_offset + source_colptr[end] - 1 <= length(rowval)

    copy_colptr!(colptr, source_colptr, column_offset, nnz_offset)
    parallel_foreach_column(source; name = "copy_columns!") do _, column_index, column
        first_stored = nnz_offset + source_colptr[column_index]
        last_stored = nnz_offset + source_colptr[column_index + 1] - 1
        rowval[first_stored:last_stored] .= SparseArrays.nonzeroinds(column)
        nzval[first_stored:last_stored] .= nonzeros(column)
        return nothing
    end
    return nothing
end

# The `colptr` of a column-major sparse `source`. A sparse matrix (or a lazy one) has its own. A view of all the rows of
# one takes the number of stored values of each column from its parent. Any other view counts them in a pass over its
# columns.
function stored_colptr(source::AbstractMatrix)::AbstractVector{<:Integer}
    canonical_source = canonical_array(source)
    if canonical_source isa AbstractSparseMatrixCSC
        return SparseArrays.getcolptr(canonical_source)
    end

    @assert canonical_source isa SubArray
    colptr = Vector{Int}(undef, size(source, 2) + 1)
    colptr[1] = 1
    row_indices, column_indices = parentindices(canonical_source)
    parent_matrix = parent(canonical_source)
    if row_indices isa Base.Slice && parent_matrix isa AbstractSparseMatrixCSC
        parent_colptr = SparseArrays.getcolptr(parent_matrix)
        @views colptr[2:end] .= parent_colptr[column_indices .+ 1] .- parent_colptr[column_indices]
    else
        parallel_foreach_column(source; name = "copy_columns!.count") do position, _, column
            colptr[position + 1] = nnz(column)
            return nothing
        end
    end
    return cumsum!(colptr, colptr)
end

# Copy the `source_colptr` into the `colptr`, after its first `column_offset` entries, adding `nnz_offset` to each.
function copy_colptr!(
    colptr::AbstractVector{<:Integer},
    source_colptr::AbstractVector{<:Integer},
    column_offset::Integer,
    nnz_offset::Integer,
)::Nothing
    for (position, source_pointer) in enumerate(source_colptr)
        colptr[column_offset + position] = source_pointer + nnz_offset
    end
    return nothing
end

# Write the values of a `column` of a source into a dense `destination_column`.
function copy_column!(destination_column::AbstractVector, column::AbstractVector)::Nothing
    destination_column .= column
    return nothing
end

function copy_column!(destination_column::AbstractVector, column::SparseArrays.SparseVectorUnion)::Nothing
    destination_column .= zero(eltype(destination_column))
    destination_column[SparseArrays.nonzeroinds(column)] .= nonzeros(column)
    return nothing
end

function is_debug_enabled_for_caller(group::Maybe{Symbol} = nothing)
    if get(ENV, "JULIA_DEBUG", "") == ""
        return false
    end
    logger = Logging.current_logger()
    if logger.min_level <= Logging.Debug
        return true
    end
    stack = stacktrace(backtrace())
    if length(stack) < 4
        caller_module = Main  # UNTESTED
    else
        caller_frame = stack[4]
        linfo = caller_frame.linfo
        if linfo === nothing
            caller_module = Main
        elseif linfo isa Module  # UNTESTED
            caller_module = linfo  # UNTESTED
        elseif linfo isa Core.MethodInstance  # UNTESTED
            def = linfo.def  # UNTESTED
            caller_module = def isa Method ? def.module : def isa Module ? def : Main  # UNTESTED
        else
            caller_module = Main  # UNTESTED
        end
    end
    # This is how `@debug` matches `JULIA_DEBUG` against the group and the module. The logger's `shouldlog` does not.
    return Base.CoreLogging.env_override_minlevel(group, caller_module) &&
           Logging.shouldlog(logger, Logging.Debug, caller_module, group, :check)
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

# The relayout of plain storage. These are methods of the `unnamed_relayout` of `MatrixLayouts`, which comes before
# the parallel loops they use.

function MatrixLayouts.unnamed_relayout(destination::SparseMatrixCSC, source::AbstractMatrix)::SparseMatrixCSC
    @assert size(destination) == size(source)
    @assert issparse(source)
    flipped_source = flip(source)
    if !sparse_relayout_by_bands!(destination, flipped_source)
        @assert LinearAlgebra.transpose!(destination, canonical_array(flipped_source)) === destination
    end
    return destination
end

function MatrixLayouts.flipped(matrix::AbstractMatrix)::AbstractMatrix
    return flame_timed("flipped") do
        axis = require_major_axis(matrix)
        if axis == Rows
            return flip(flipped(flip(matrix)))
        end
        @assert axis == Columns

        n_rows, n_columns = size(matrix)
        if issparse(matrix)
            n_stored = stored_colptr(matrix)[end] - 1
            canonical_matrix = canonical_array(matrix)
            index_type =
                SparseArrays.indtype(canonical_matrix isa SubArray ? parent(canonical_matrix) : canonical_matrix)
            colptr = fill(index_type(n_stored + 1), n_rows + 1)
            colptr[1] = 1
            result = SparseMatrixCSC(
                n_columns,
                n_rows,
                colptr,
                Vector{index_type}(undef, n_stored),
                Vector{eltype(matrix)}(undef, n_stored),
            )
        else
            result = Matrix{eltype(matrix)}(undef, n_columns, n_rows)
        end
        relayout!(flip(result), matrix)

        @assert major_axis(result) == Columns
        @assert size(result, Rows) == size(matrix, Columns)
        @assert size(result, Columns) == size(matrix, Rows)

        return result
    end
end

function MatrixLayouts.unnamed_relayout(destination::DenseMatrix, source::AbstractMatrix)::DenseMatrix
    @assert size(destination) == size(source)
    @assert !issparse(source)
    flipped_source = flip(source)
    if !dense_relayout_by_bands!(destination, flipped_source)
        transpose_columns!(destination, flipped_source)
    end
    return destination
end

# Relayouting in bands needs a contiguous destination, so a view of a dense matrix is relayouted without them.
function MatrixLayouts.unnamed_relayout(
    destination::SubArray{<:Any, 2, <:DenseMatrix},
    source::AbstractMatrix,
)::SubArray
    @assert size(destination) == size(source)
    @assert !issparse(source)
    transpose_columns!(destination, flip(source))
    return destination
end

# Copy the transpose of the column-major `flipped_source` into the `destination`. Plain storage is copied by
# `transpose!`. Any other matrix (e.g. a lazy one) is read through its columns, each written into a row of the
# `destination`.
function transpose_columns!(destination::AbstractMatrix, flipped_source::AbstractMatrix)::Nothing
    canonical_source = canonical_array(flipped_source)
    @assert canonical_source isa AbstractMatrix
    if is_plain_array(canonical_source)
        @assert transpose_values!(destination, canonical_source) === destination
    else
        parallel_foreach_column(flipped_source; name = "relayout.transpose_columns") do _, column_index, column
            view(destination, column_index, :) .= column
            return nothing
        end
    end
    return nothing
end

# Copy the transpose of the `source` into the `destination`. `transpose!` also transposes each element, which only
# numbers support, so other elements are just moved.
function transpose_values!(destination::AbstractMatrix, source::AbstractMatrix)::AbstractMatrix
    if eltype(source) <: Number
        return transpose!(destination, source)
    else
        return permutedims!(destination, source, (2, 1))
    end
end

# Whether to relayout `data_bytes` of data in bands, rather than using `transpose!`.
function is_relayout_by_bands(data_bytes::Integer)::Bool
    return data_bytes > MatrixLayouts.RELAYOUT_BUDGET_KB * 1024
end

# The number of tasks, and the total scratch memory (in bytes), to relayout `data_bytes` of data in bands. The writes of
# each task are at least `RELAYOUT_MIN_WRITE_KB` if its memory is at least `sqrt(2 * data_bytes * min_write_bytes)`. If
# the budget does not allow even a single such task, then it is exceeded.
function relayout_tasks_and_budget(data_bytes::Integer)::Tuple{Int, Int}
    budget_bytes = MatrixLayouts.RELAYOUT_BUDGET_KB * 1024
    task_bytes = ceil(Int, sqrt(2.0 * data_bytes * MatrixLayouts.RELAYOUT_MIN_WRITE_KB * 1024))
    n_tasks = clamp(budget_bytes ÷ task_bytes, 1, nthreads())
    return (n_tasks, max(budget_bytes, n_tasks * task_bytes))
end

# Split the entries `1:(length(cumulative_weights) - 1)` into `n_chunks` contiguous ranges with about the same total
# weight. The `cumulative_weights` are like a sparse matrix `colptr`.
function relayout_chunks(cumulative_weights::AbstractVector{<:Integer}, n_chunks::Integer)::Vector{UnitRange{Int}}
    n_entries = length(cumulative_weights) - 1
    total_weight = cumulative_weights[end] - cumulative_weights[1]
    chunk_firsts = [1]
    for chunk_index in 1:(n_chunks - 1)
        target_weight = cumulative_weights[1] + total_weight * chunk_index / n_chunks
        chunk_first = searchsortedfirst(cumulative_weights, target_weight)
        push!(chunk_firsts, clamp(chunk_first, chunk_firsts[end], n_entries + 1))
    end
    push!(chunk_firsts, n_entries + 1)
    return [chunk_firsts[chunk_index]:(chunk_firsts[chunk_index + 1] - 1) for chunk_index in 1:n_chunks]
end

# The number of bits needed to hold the non-negative `value`.
function n_bits_for(value::Integer)::Int
    return 64 - leading_zeros(UInt64(value))
end

# Relayout the sparse `source` into the `destination` in bands of the `destination` columns, which are the `source`
# rows. The `source` is read through its columns, so a lazy one is never held in memory as a whole. Each pass runs over
# chunks of the `source` columns. Pass 1 counts the entries of each `source` row in each chunk. Pass 2 stores each entry
# with a packed index, which holds both its column in the band and its row. A band is therefore at most as wide as the
# index type allows. Pass 3 sorts each band. Returns whether this was done; a small `SparseMatrixCSC` isn't, so use
# `transpose!` for it instead.
function sparse_relayout_by_bands!(destination::SparseMatrixCSC{Td, Id}, source::AbstractMatrix)::Bool where {Td, Id}
    n_destination_rows, n_destination_columns = size(destination)
    @assert size(source) == (n_destination_columns, n_destination_rows)
    source_colptr = stored_colptr(source)
    n_entries = source_colptr[end] - 1
    entry_bytes = sizeof(Id) + sizeof(Td)
    if canonical_array(source) isa SparseMatrixCSC && !is_relayout_by_bands(n_entries * entry_bytes)
        return false
    elseif n_entries == 0
        destination.colptr .= 1
        return true
    end
    row_bits = n_bits_for(n_destination_rows - 1)
    band_column_bits = 8 * sizeof(Id) - (Id <: Signed ? 1 : 0) - row_bits
    @assert band_column_bits >= 0
    @assert length(destination.rowval) >= n_entries
    @assert length(destination.nzval) >= n_entries

    n_tasks, budget_bytes = relayout_tasks_and_budget(n_entries * entry_bytes)
    chunks = relayout_chunks(source_colptr, n_tasks)
    column_reader_per_thread = new_column_reader_per_thread(source)

    n_entries_per_row_per_chunk = zeros(Int, n_destination_columns, n_tasks)
    parallel_loop_wo_rng(1:n_tasks; name = "relayout.count") do chunk_index
        count_sparse_relayout_chunk!(
            view(n_entries_per_row_per_chunk, :, chunk_index),
            column_reader_per_thread[threadid()],
            chunks[chunk_index],
        )
        return nothing
    end
    colptr = Vector{Int}(undef, n_destination_columns + 1)
    colptr[1] = 1
    sum!(view(colptr, 2:(n_destination_columns + 1)), n_entries_per_row_per_chunk)
    cumsum!(colptr, colptr)
    copyto!(destination.colptr, colptr)

    max_band_entries = max(1, budget_bytes ÷ (2 * n_tasks * entry_bytes))
    band_first_columns, band_of_column = sparse_relayout_bands(colptr, max_band_entries, 1 << min(band_column_bits, 62))
    n_bands = length(band_first_columns) - 1

    positions = Matrix{Int}(undef, n_bands, n_tasks)
    @inbounds for band_index in 1:n_bands
        band_columns = band_first_columns[band_index]:(band_first_columns[band_index + 1] - 1)
        position = colptr[first(band_columns)]
        for chunk_index in 1:n_tasks
            positions[band_index, chunk_index] = position
            position += sum(view(n_entries_per_row_per_chunk, band_columns, chunk_index))
        end
    end

    buffer_entries = max(1, budget_bytes ÷ (n_tasks * n_bands * entry_bytes))
    parallel_loop_wo_rng(1:n_tasks; name = "relayout.fill_bands") do chunk_index
        fill_sparse_relayout_bands!(
            destination,
            column_reader_per_thread[threadid()],
            chunks[chunk_index],
            band_first_columns,
            band_of_column,
            view(positions, :, chunk_index),
            row_bits,
            buffer_entries,
        )
        return nothing
    end

    max_band_width = maximum(diff(band_first_columns))
    max_entries_in_band = maximum(diff(colptr[band_first_columns]))
    next_band_index = Atomic{Int}(1)
    parallel_loop_wo_rng(1:n_tasks; name = "relayout.sort_bands") do _
        sort_sparse_relayout_bands!(
            destination,
            colptr,
            band_first_columns,
            next_band_index,
            row_bits,
            min(max_entries_in_band, max_band_entries),
            max_band_width,
        )
        return nothing
    end

    return true
end

# Split the relayout destination columns into bands. A band has at most `max_band_entries` entries (unless it is a
# single column), and at most `max_band_columns` columns. Returns the first column of each band (followed by one past
# the last column), and the band of each column.
function sparse_relayout_bands(
    colptr::AbstractVector{Int},
    max_band_entries::Integer,
    max_band_columns::Integer,
)::Tuple{Vector{Int}, Vector{Int32}}
    n_columns = length(colptr) - 1
    band_first_columns = Int[]
    band_of_column = Vector{Int32}(undef, n_columns)
    first_column = 1
    while first_column <= n_columns
        last_column = first_column
        while last_column < n_columns &&
                  last_column - first_column + 1 < max_band_columns &&
                  colptr[last_column + 2] - colptr[first_column] <= max_band_entries
            last_column += 1
        end
        push!(band_first_columns, first_column)
        band_of_column[first_column:last_column] .= length(band_first_columns)
        first_column = last_column + 1
    end
    push!(band_first_columns, n_columns + 1)
    return (band_first_columns, band_of_column)
end

# Pass 1 of a sparse relayout. Count the entries of each `source` row in the `source_columns`, which are read through
# the `column_reader`.
function count_sparse_relayout_chunk!(
    n_entries_per_row::AbstractVector{Int},
    column_reader::ColumnReader,
    source_columns::UnitRange{Int},
)::Nothing
    for source_column in source_columns
        @inbounds for source_row in SparseArrays.nonzeroinds(read_column(column_reader, source_column))
            n_entries_per_row[source_row] += 1
        end
    end
    return nothing
end

# Pass 2 of a sparse relayout. Append the entries of the `source_columns`, which are read through the `column_reader`,
# to their bands in the `destination`, starting at the `positions` of this chunk in each band. Each entry is buffered
# with a packed index of its column in the band and its row, and the buffers are written when full.
function fill_sparse_relayout_bands!(
    destination::SparseMatrixCSC{Td, Id},
    column_reader::ColumnReader,
    source_columns::UnitRange{Int},
    band_first_columns::Vector{Int},
    band_of_column::Vector{Int32},
    positions::AbstractVector{Int},
    row_bits::Int,
    buffer_entries::Int,
)::Nothing where {Td, Id}
    n_bands = length(band_first_columns) - 1
    indices_buffers = Matrix{Id}(undef, buffer_entries, n_bands)
    values_buffers = Matrix{Td}(undef, buffer_entries, n_bands)
    n_buffered_entries = zeros(Int, n_bands)

    for source_column in source_columns
        column = read_column(column_reader, source_column)
        column_rows = SparseArrays.nonzeroinds(column)
        column_values = nonzeros(column)
        packed_row = Id(source_column - 1)
        @inbounds for entry_index in eachindex(column_rows)
            destination_column = column_rows[entry_index]
            band_index = band_of_column[destination_column]
            n_buffered = n_buffered_entries[band_index] + 1
            packed_column = Id(destination_column - band_first_columns[band_index]) << row_bits
            indices_buffers[n_buffered, band_index] = packed_column | packed_row
            values_buffers[n_buffered, band_index] = column_values[entry_index]
            if n_buffered == buffer_entries
                write_sparse_relayout_buffer!(
                    destination,
                    indices_buffers,
                    values_buffers,
                    band_index,
                    n_buffered,
                    positions,
                )
                n_buffered = 0
            end
            n_buffered_entries[band_index] = n_buffered
        end
    end

    for band_index in 1:n_bands
        write_sparse_relayout_buffer!(
            destination,
            indices_buffers,
            values_buffers,
            band_index,
            n_buffered_entries[band_index],
            positions,
        )
    end
    return nothing
end

# Write the first `n_buffered` entries of a band's buffers to the `destination` at the band's position, and advance it.
@inline function write_sparse_relayout_buffer!(
    destination::SparseMatrixCSC,
    indices_buffers::Matrix,
    values_buffers::Matrix,
    band_index::Integer,
    n_buffered::Integer,
    positions::AbstractVector{Int},
)::Nothing
    if n_buffered > 0
        first_buffered = (band_index - 1) * size(indices_buffers, 1) + 1
        copyto!(destination.rowval, positions[band_index], indices_buffers, first_buffered, n_buffered)
        copyto!(destination.nzval, positions[band_index], values_buffers, first_buffered, n_buffered)
        positions[band_index] += n_buffered
    end
    return nothing
end

# Pass 3 of a sparse relayout. Take the next band until there are none left. Sort the entries of each band by their
# column in the band, keeping the order of the rows in each column, and unpack the rows from the packed indices. A band
# of a single column needs no sorting, so it is unpacked in pieces, and may be larger than `max_band_entries`.
function sort_sparse_relayout_bands!(
    destination::SparseMatrixCSC{Td, Id},
    colptr::Vector{Int},
    band_first_columns::Vector{Int},
    next_band_index::Atomic{Int},
    row_bits::Int,
    max_band_entries::Int,
    max_band_width::Int,
)::Nothing where {Td, Id}
    n_bands = length(band_first_columns) - 1
    destination_rowval = destination.rowval
    destination_nzval = destination.nzval
    row_mask = (one(Id) << row_bits) - one(Id)
    packed_indices = Vector{Id}(undef, max_band_entries)
    band_values = Vector{Td}(undef, max_band_entries)
    sorted_rows = Vector{Id}(undef, max_band_entries)
    sorted_values = Vector{Td}(undef, max_band_entries)
    next_positions = Vector{Int}(undef, max_band_width)

    while true
        band_index = atomic_add!(next_band_index, 1)
        if band_index > n_bands
            return nothing
        end
        first_column = band_first_columns[band_index]
        band_width = band_first_columns[band_index + 1] - first_column
        first_position = colptr[first_column]
        n_band_entries = colptr[first_column + band_width] - first_position

        if band_width == 1
            for first_piece_offset in 0:max_band_entries:(n_band_entries - 1)
                n_piece_entries = min(max_band_entries, n_band_entries - first_piece_offset)
                copyto!(packed_indices, 1, destination_rowval, first_position + first_piece_offset, n_piece_entries)
                @inbounds for entry_index in 1:n_piece_entries
                    packed_indices[entry_index] = (packed_indices[entry_index] & row_mask) + one(Id)
                end
                copyto!(destination_rowval, first_position + first_piece_offset, packed_indices, 1, n_piece_entries)
            end

        elseif n_band_entries > 0
            copyto!(packed_indices, 1, destination_rowval, first_position, n_band_entries)
            copyto!(band_values, 1, destination_nzval, first_position, n_band_entries)
            @inbounds for band_column in 1:band_width
                next_positions[band_column] = colptr[first_column + band_column - 1] - first_position + 1
            end
            @inbounds for entry_index in 1:n_band_entries
                packed_index = packed_indices[entry_index]
                band_column = Int(packed_index >> row_bits) + 1
                sorted_index = next_positions[band_column]
                next_positions[band_column] = sorted_index + 1
                sorted_rows[sorted_index] = (packed_index & row_mask) + one(Id)
                sorted_values[sorted_index] = band_values[entry_index]
            end
            copyto!(destination_rowval, first_position, sorted_rows, 1, n_band_entries)
            copyto!(destination_nzval, first_position, sorted_values, 1, n_band_entries)
        end
    end
end

# Relayout the large dense `source` into the `destination` in bands of the `destination` columns, which are the `source`
# rows. Pass 1 leaves each band's region holding the band's `source` rows, and pass 2 transposes each band in memory.
# Returns whether this was done; if not, use `transpose!` instead.
function dense_relayout_by_bands!(destination::DenseMatrix{Td}, source::AbstractMatrix)::Bool where {Td}
    n_destination_rows, n_destination_columns = size(destination)
    @assert size(source) == (n_destination_columns, n_destination_rows)
    # A reference for an element type that isn't a bits type (e.g. `String`).
    element_bytes = Base.elsize(destination)
    data_bytes = length(destination) * element_bytes
    if !is_relayout_by_bands(data_bytes)
        return false
    end

    n_tasks, budget_bytes = relayout_tasks_and_budget(data_bytes)
    column_bytes = n_destination_rows * element_bytes
    band_width = max(1, budget_bytes ÷ (2 * n_tasks * column_bytes))
    n_bands = cld(n_destination_columns, band_width)
    buffer_rows = max(1, budget_bytes ÷ (n_tasks * n_bands * band_width * element_bytes))
    destination_values = vec(destination)

    chunks = relayout_chunks(0:n_destination_rows, n_tasks)
    column_reader_per_thread = new_column_reader_per_thread(source)
    parallel_loop_wo_rng(1:n_tasks; name = "relayout.fill_bands") do chunk_index
        fill_dense_relayout_bands!(
            destination_values,
            column_reader_per_thread[threadid()],
            size(source, 1),
            chunks[chunk_index],
            band_width,
            buffer_rows,
        )
        return nothing
    end

    next_band_index = Atomic{Int}(1)
    parallel_loop_wo_rng(1:n_tasks; name = "relayout.transpose_bands") do _
        transpose_dense_relayout_bands!(
            destination_values,
            n_destination_rows,
            n_destination_columns,
            band_width,
            next_band_index,
        )
        return nothing
    end

    return true
end

# Pass 1 of a dense relayout. Append the band slices of the `destination_rows` (which are `source` columns, read through
# the `column_reader`) to their bands in the `destination_values`. Each band's region ends up holding the band's
# `source` rows, in column-major order. The slices are buffered, and the buffers are written when they hold
# `buffer_rows` rows.
function fill_dense_relayout_bands!(
    destination_values::AbstractVector{Td},
    column_reader::ColumnReader,
    n_destination_columns::Int,
    destination_rows::UnitRange{Int},
    band_width::Int,
    buffer_rows::Int,
)::Nothing where {Td}
    n_destination_rows = length(destination_values) ÷ n_destination_columns
    n_bands = cld(n_destination_columns, band_width)
    buffers = [Vector{Td}(undef, band_width * buffer_rows) for _ in 1:n_bands]
    n_buffered_rows = 0

    for destination_row in destination_rows
        n_buffered_rows += 1
        source_column = read_column(column_reader, destination_row)
        for band_index in 1:n_bands
            first_column = (band_index - 1) * band_width + 1
            last_column = min(first_column + band_width - 1, n_destination_columns)
            n_band_columns = last_column - first_column + 1
            copyto!(
                buffers[band_index],
                (n_buffered_rows - 1) * n_band_columns + 1,
                view(source_column, first_column:last_column),
                1,
                n_band_columns,
            )
        end

        if n_buffered_rows == buffer_rows || destination_row == last(destination_rows)
            first_buffered_row = destination_row - n_buffered_rows + 1
            for band_index in 1:n_bands
                first_column = (band_index - 1) * band_width + 1
                n_band_columns = min(band_width, n_destination_columns - first_column + 1)
                position = (first_column - 1) * n_destination_rows + (first_buffered_row - 1) * n_band_columns + 1
                copyto!(destination_values, position, buffers[band_index], 1, n_band_columns * n_buffered_rows)
            end
            n_buffered_rows = 0
        end
    end
    return nothing
end

# Pass 2 of a dense relayout. Take the next band until there are none left, and transpose it in memory. A band of a
# single column is already in place.
function transpose_dense_relayout_bands!(
    destination_values::AbstractVector{Td},
    n_destination_rows::Int,
    n_destination_columns::Int,
    band_width::Int,
    next_band_index::Atomic{Int},
)::Nothing where {Td}
    n_bands = cld(n_destination_columns, band_width)
    band_values = Vector{Td}(undef, band_width * n_destination_rows)
    transposed_values = Vector{Td}(undef, band_width * n_destination_rows)

    while true
        band_index = atomic_add!(next_band_index, 1)
        if band_index > n_bands
            return nothing
        end
        first_column = (band_index - 1) * band_width + 1
        n_band_columns = min(band_width, n_destination_columns - first_column + 1)
        if n_band_columns > 1
            n_band_values = n_band_columns * n_destination_rows
            position = (first_column - 1) * n_destination_rows + 1
            copyto!(band_values, 1, destination_values, position, n_band_values)
            transpose_values!(
                reshape(view(transposed_values, 1:n_band_values), n_destination_rows, n_band_columns),
                reshape(view(band_values, 1:n_band_values), n_band_columns, n_destination_rows),
            )
            copyto!(destination_values, position, transposed_values, 1, n_band_values)
        end
    end
end

end
