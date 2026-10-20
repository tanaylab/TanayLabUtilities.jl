"""
Parallel loops.
"""
module ParallelLoops

export DebugProgress
export DebugProgressUnknown
export column_scratch
export is_in_parallel_loop
export parallel_foreach_column
export parallel_loop_with_rng
export parallel_loop_wo_rng
export read_column!

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
        progress::Maybe{AbstractProgress} = nothing,
    )::Nothing

Run the `body` in parallel, passing it the iteration `index`. The `policy` is one of:

  - `:greedy_sticky` - `nthreads()` sticky worker tasks spawned via `@threads :static for _ in 1:nthreads()`, each
    pulling the next index position from an `Atomic{Int}` counter. Combines the dynamic load balancing of `:greedy`
    with the threadid stability of `:static`: tasks never migrate, so `threadid()`-indexed scratch is safe inside the
    body. Pair with a heaviest-first `order` (or `weights`) to attack the tail-latency problem - heavy items dispatch
    first, light items fill the tail.
  - `:greedy`, `:dynamic`, `:static` - passed straight to `@threads`. Note that `:greedy` and `:dynamic` create
    *non-sticky* tasks that may migrate across threads at yield points, so `threadid()`-indexed scratch is unsafe
    under those.
  - `:serial` - run the loop on the calling thread (useful for debugging).

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
        @assert !startswith(name, ".") || length(flame_stack_names) > 0  # UNTESTED
        (startswith(name, ".") ? flame_stack_names[end] * name : name, join(flame_stack_names, ";"))  # UNTESTED
    end

    # A loop nested in another parallel loop runs serially, without its own progress.
    is_nested_in_parallel_loop = nthreads() > 1 && !nested && haskey(base_private_storage, :is_in_parallel_loop)
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
    # gets one item from each weight quantile. Under any other policy, the work queue consumes items in the order they
    # appear, so an already-sorted (heaviest-first) `order` directly dispatches the heaviest items earliest. The body
    # always receives `indices[order[k]]` - `view(indices, order)` projects positions to values without allocating.
    if order !== nothing && loop_policy == :static
        round_robin_for_static_chunks!(order, nthreads())
    end
    iteration_indices = order === nothing ? indices : view(indices, order)
    n_index_positions = length(iteration_indices)

    # Per-position progress step. Built after the `:static` round-robin shuffle so `step_per_position[k]` matches the
    # `iteration_indices[k]` actually visited at step `k`. `nothing` when no `weights` were given, in which case the
    # iteration body falls back to `+1` per step (or to `progress_chunk` throttling).
    step_per_position = weights === nothing ? nothing : (order === nothing ? weights : weights[order])

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
                    index_flame_stack = [base_flame_stack]
                    index_private_storage[:flame_stack] = index_flame_stack

                    flame_timed(loop_name) do
                        body(iteration_indices[position])
                        advance_progress(position)
                        return nothing
                    end

                    return nothing
                end

                saved_blas_threads = BLAS.get_num_threads()
                BLAS.set_num_threads(1)
                try
                    if loop_policy == :greedy
                        @threads :greedy for position in 1:n_index_positions
                            iteration_body(position)
                        end

                    elseif loop_policy == :static
                        @threads :static for position in 1:n_index_positions
                            iteration_body(position)
                        end

                    elseif loop_policy == :dynamic
                        @threads :dynamic for position in 1:n_index_positions
                            iteration_body(position)
                        end

                    elseif loop_policy == :greedy_sticky
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
    parallel_foreach_column(
        visit::Function,
        matrix::AbstractMatrix;
        columns::AbstractVector{<:Integer} = 1:size(matrix, 2),
        contiguous::Bool = false,
        name::AbstractString = ".columns",
        policy::Symbol = :greedy_sticky,
        weights::Maybe{AbstractVector{<:Integer}} = nothing,
        progress::Maybe{AbstractProgress} = nothing,
        progress_chunk::Maybe{Integer} = nothing,
    )::Nothing

Run `visit(position, column_index, column)` in parallel for each of the `columns` of the `matrix`. The `position` is the
index of the `column_index` in `columns`, so `visit` can write its results by position. This is [`plain_array`](@ref)
one column at a time: the `column` holds the values of the column in plain storage, so `visit` needs no `plain_array` of
its own. The `column` is valid only during the call to `visit`, and must not be modified.

A dense `column` is a view of a column of plain storage, or a buffer holding the column. A sparse `column` is read
through `nonzeroinds` and `nonzeros`. It is a view of a column of a `SparseMatrixCSC`, or a `SparseVector` holding the
column.

A lazy matrix (which reads its data on demand) is read one column at a time through [`column_scratch`](@ref) and
[`read_column!`](@ref), so all of it is never held in memory. Any other matrix is read once through `plain_array`, which
copies nothing for plain storage, read-only and named wrappers, and strided views. A matrix stored in rows (e.g. a
transpose) is read against the grain, which invokes the [`GLOBAL_INEFFICIENT_ACTION_HANDLER`](@ref). A sparse one is
relayouted to columns first.

Set `contiguous` to copy each column with gaps (of a strided view or of a transpose) into a buffer, for code that needs
contiguous memory.

Buffers are allocated before the loop, one per thread, and dropped after it. They are indexed by `threadid()`, so the
`policy` must keep each task on its thread (`:greedy_sticky`, `:static` or `:serial`). If this is invoked from inside
another parallel loop, it runs serially (see [`parallel_loop_wo_rng`](@ref)) and allocates a single buffer. Since this
allocates inside the other loop, it invokes the [`GLOBAL_INEFFICIENT_ACTION_HANDLER`](@ref).

For a sparse matrix, the `weights` default to the number of stored values in each of the `columns`, unless a
`progress_chunk` is given. The `weights`, the `name`, the `policy`, the `progress` and the `progress_chunk` are passed
to [`parallel_loop_wo_rng`](@ref).

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
    weights::Maybe{AbstractVector{<:Integer}} = nothing,
    progress::Maybe{AbstractProgress} = nothing,
    progress_chunk::Maybe{Integer} = nothing,
)::Nothing
    @assert policy in (:greedy_sticky, :static, :serial)
    if isempty(columns)
        return nothing
    end

    base_matrix = base_array(matrix)
    # A local of its own keeps the type of the default weights concrete.
    weight_per_position = weights
    if weight_per_position === nothing && progress_chunk === nothing && base_matrix isa AbstractSparseMatrixCSC
        column_pointers = SparseArrays.getcolptr(base_matrix)
        weight_per_position =
            [Int(column_pointers[column_index + 1] - column_pointers[column_index]) for column_index in columns]
    end
    is_serial = is_serial_loop(policy)

    scratch = column_scratch(base_matrix)
    if scratch !== nothing
        scratch_per_thread = new_scratch_per_thread(scratch, () -> column_scratch(base_matrix), matrix, is_serial)
        foreach_column_through_scratch(
            visit,
            (thread_scratch, column_index) -> read_column!(thread_scratch, base_matrix, column_index),  # NOLINT
            columns,
            scratch_per_thread;
            name,
            policy,
            weights = weight_per_position,
            progress,
            progress_chunk,
        )
        return nothing
    end

    storage = plain_array(matrix)
    @assert_matrix(storage, Columns)
    if issparse(storage) && major_axis(storage) != Columns
        storage = SparseMatrixCSC(storage)
    end
    foreach_plain_column(
        visit,
        storage,
        matrix,
        columns,
        contiguous,
        is_serial;
        name,
        policy,
        weights = weight_per_position,
        progress,
        progress_chunk,
    )
    return nothing
end

# Whether a parallel loop with the `policy` runs serially, which is also the case inside another parallel loop.
function is_serial_loop(policy::Symbol)::Bool
    return policy == :serial || nthreads() == 1 || is_in_parallel_loop()
end

# Run `visit(position, column_index, column)` for each of the `columns` of the plain `storage` of the `matrix`, where
# the `column` is a view of the storage, or a copy of it in a buffer if it has gaps and must be `contiguous`.
function foreach_plain_column(
    visit::Function,
    storage::AbstractMatrix,
    matrix::AbstractMatrix,
    columns::AbstractVector{<:Integer},
    contiguous::Bool,
    is_serial::Bool;
    name::AbstractString,
    policy::Symbol,
    weights::Maybe{AbstractVector{<:Integer}},
    progress::Maybe{AbstractProgress},
    progress_chunk::Maybe{Integer},
)::Nothing
    if contiguous && !issparse(storage) && !is_contiguous_array(view(storage, :, first(columns)))
        create_buffer = () -> Vector{eltype(storage)}(undef, size(storage, 1))
        buffer_per_thread = new_scratch_per_thread(create_buffer(), create_buffer, matrix, is_serial)
        foreach_column_through_scratch(
            visit,
            (buffer, column_index) -> copyto!(buffer, view(storage, :, column_index)),
            columns,
            buffer_per_thread;
            name,
            policy,
            weights,
            progress,
            progress_chunk,
        )
    else
        parallel_loop_wo_rng(1:length(columns); name, policy, weights, progress, progress_chunk) do position
            column_index = columns[position]
            visit(position, column_index, view(storage, :, column_index))
            return nothing
        end
    end
    return nothing
end

# The scratch of each thread, starting with the `first_scratch`, the rest made by `create_scratch`. A serial loop shares
# the `first_scratch` between all threads. If it is nested inside another parallel loop, this allocation is inside that
# loop, which invokes the `GLOBAL_INEFFICIENT_ACTION_HANDLER`.
function new_scratch_per_thread(
    first_scratch::Any,
    create_scratch::Function,
    matrix::AbstractMatrix,
    is_serial::Bool;
    report::Bool = true,  # TODOX
)::AbstractVector
    if !is_serial
        return [thread_index == 1 ? first_scratch : create_scratch() for thread_index in 1:maxthreadid()]
    end
    if report && is_in_parallel_loop()  # TODOX
        handle_abnormal(GLOBAL_INEFFICIENT_ACTION_HANDLER) do
            return """
                allocating column scratch inside a parallel loop
                for matrix: $(brief(matrix))
                """
        end
    end
    return fill(first_scratch, maxthreadid())
end

# Run `visit(position, column_index, column)` for each of the `columns`, where `read_column(scratch, column_index)`
# returns the column, using the scratch of the current thread.
function foreach_column_through_scratch(
    visit::Function,
    read_column::Function,
    columns::AbstractVector{<:Integer},
    scratch_per_thread::AbstractVector;
    name::AbstractString,
    policy::Symbol,
    weights::Maybe{AbstractVector{<:Integer}},
    progress::Maybe{AbstractProgress},
    progress_chunk::Maybe{Integer},
)::Nothing
    parallel_loop_wo_rng(1:length(columns); name, policy, weights, progress, progress_chunk) do position
        column_index = columns[position]
        visit(position, column_index, read_column(scratch_per_thread[threadid()], column_index))
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

end
