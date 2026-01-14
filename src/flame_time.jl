"""
Generate a coarse-grained flamegraph. If the environment variable `JULIA_FLAME_MEASUREMENTS_FILE` is set, it should
contain the path of a file to append measurements into. Every call to [`flame_timed`](@ref) will append measurement into
this file. This file can't be directly viewed using `flamegraph.pl`. It needs to be converted to a proper flamegraph
file using [`finalize_flamegraph`](@ref).

!!! note

    Measurements are appended into the file, so that running multiple programs automatically creates a unified file for
    a whole computational pipeline. If you want to separately measure each program execution, use a different
    measurements file for each.

This serves as a very lightweight mechanism for visualizing program performance. Every `@logged` function and every
`parallel_loop` is automatically a measured step, so one gets a pretty decent level of detail "for free". It is possible
to supplement this with occasional calls to `flame_timed` to increase the level of detail. Of course one can ignore all
of this and use heavyweight tools such as `Profile` when profiling for performance in anger.
"""
module FlameTime

export flame_timed
export finalize_flamegraph

using ..Types

function __init__()
    path = get(ENV, "JULIA_FLAME_MEASUREMENTS_FILE", nothing)
    if path === nothing
        global FLAME_MEASUREMENTS_FILE = nothing
    else
        global FLAME_MEASUREMENTS_FILE = open(path, "a")  # UNTESTED
        @info "Appending flamegraph measurements into: $(path)"  # UNTESTED
    end
end

FLAME_MEASUREMENTS_LOCK = ReentrantLock()

"""
    flame_timed(body::Function, name::AbstractString)::Any

Add a measurement line to the flamegrah file for the execution of the `body` under the specified `name`. The measurement
is of the elapsed time for the body in nanoseconds. We distinguish between "serial" measurements (in the top-level main
thread) and "parallel" measurements (inside tasks invoked from within a `parallel_loop`).
"""
function flame_timed(body::Function, name::AbstractString)::Any
    if FLAME_MEASUREMENTS_FILE === nothing  # NOLINT
        return body()
    else
        start_ns = time_ns()  # UNTESTED

        private_storage = task_local_storage()  # UNTESTED

        flame_stack = get(private_storage, "flame_stack", nothing)  # UNTESTED
        if flame_stack === nothing  # UNTESTED
            private_storage["flame_stack"] = flame_stack = AbstractString[]  # UNTESTED
        end
        push!(flame_stack, name)  # UNTESTED

        try  # UNTESTED
            return body()  # UNTESTED

        finally
            is_in_parallel = get(private_storage, "is_in_parallel", false)  # UNTESTED

            end_ns = time_ns()  # UNTESTED

            lock(FLAME_MEASUREMENTS_LOCK) do               # UNTESTED
                println(FLAME_MEASUREMENTS_FILE, "$(join(flame_stack, ";")) $(is_in_parallel) $(end_ns - start_ns)")  # NOLINT
                return nothing
            end

            pop!(flame_stack)  # UNTESTED
        end
    end
end

"""
    finalize_flamegraph(; measurements_path::AbstractString, flamegraph_path::AbstractString)::Nothing

Convert the collected measurements in the `measurement_path` into a proper flamegraph file, suitable for use with
`flamegraph.pl` and similar tools. This is needed for two reasons: first, the measurements are inclusive (that is, the
elapsed time of a step includes the elapsed time of all nested steps). Second, measurements of serial steps (which
should add to the total execution time) are mixed with the measurements of parallel steps (which would add up to much
more).

Each `parallel_loop` is therefore measured twice, once as a serial step and once as a parallel step. We scale the
parallel time (and everything nested in it) so it adds up to the serial time. We then convert all the inclusive
measurements to exclusive ones. The result is a flamegraph that shows the contribution of everything (serial and
parallel steps) to the total execution time, under the assumption that the number of iterations in the parallel loops is
large enough that it is the sum of parallel steps that matters.

!!! note

    This will not work if you call [`flame_timed`](@ref) inside an explicit `@threads` loops. You must use a
    `parallel_loop` instead.
"""
function finalize_flamegraph(; measurements_path::AbstractString, flamegraph_path::AbstractString)::Nothing  # UNTESTED
    measurements_dict = Dict{Tuple{AbstractString, Bool}, Int64}()
    for line in eachline(measurements_path)
        fields = split(line)
        @assert length(fields) == 3
        stack = fields[1]
        is_in_parallel = parse(Bool, fields[2])
        elapsed = parse(Int64, fields[3]) + get(measurements_dict, (stack, is_in_parallel), 0)
        measurements_dict[(stack, is_in_parallel)] = elapsed
    end

    flamegraph_dict = Dict{AbstractString, Int64}()

    loop_scale = nothing
    loop_stack = nothing

    previous_stack = nothing
    previous_is_in_parallel = nothing
    previous_elapsed = nothing
    for ((stack, is_in_parallel), elapsed) in sort!(collect(measurements_dict))
        if stack == previous_stack
            @assert is_in_parallel
            @assert !previous_is_in_parallel  # NOJET
            loop_scale = max(previous_elapsed / elapsed, 1.0)  # NOJET
            loop_stack = stack
        elseif is_in_parallel
            @assert startswith(stack, loop_stack)  # NOJET
            @assert length(stack) > length(loop_stack)  # NOJET
            flamegraph_dict[stack] = round(elapsed * loop_scale)  # NOJET
        else
            flamegraph_dict[stack] = elapsed
        end
        previous_stack = stack
        previous_elapsed = elapsed
        previous_is_in_parallel = is_in_parallel
    end

    for (stack, elapsed) in sort!(collect(flamegraph_dict); rev = true)
        semicolon_index = findlast(';', stack)
        if semicolon_index !== nothing
            parent_stack = stack[1:(semicolon_index - 1)]
            if haskey(flamegraph_dict, parent_stack)
                flamegraph_dict[parent_stack] -= elapsed
            end
        end
    end

    open(flamegraph_path, "w") do file
        for (stack, elapsed) in flamegraph_dict
            @assert elapsed >= 0
            if elapsed > 0
                println(file, "$(stack) $(elapsed)")
            end
        end
    end

    return nothing
end

end
