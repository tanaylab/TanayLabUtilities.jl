"""
Generate a coarse-grained flameview. If the environment variable `TLU_FLAME_MEASUREMENTS_FILE` is set, it should
contain the path of a file to append measurements into. Every call to [`flame_timed`](@ref) will collect measurements,
and on normal program exit, the total ("folded") measurements are written into this file. The file can't be directly
viewed using `flameview.py`; it needs to be converted to a proper flameview file using [`finalize_flameview`](@ref).

!!! note

    Measurements are appended into the file, so that running multiple programs automatically creates a unified file for
    a whole computational pipeline. Measurement stacks are prefixed by the program name (`basename(source_path())` or
    `Main`). If you want to separately measure each program execution, use a different measurements file for each. Also,
    there's no locking done, so running programs in parallel with the same measurements file will not end well.

This serves as a very lightweight mechanism for visualizing program performance. Every `@logged` function and every
`parallel_loop` is automatically a measured step, so one gets a pretty decent level of detail "for free". It is possible
to supplement this with occasional calls to `flame_timed` to increase the level of detail. Of course one can ignore all
of this and use more detailed tools such as `Profile` when profiling for performance in anger.
"""
module FlameTime

using Base.Threads
using DictTools

export flame_timed
export finalize_flameview
export warn_about_expensive_operation_in_parallel

using ..Brief
using ..Types

mutable struct RUsage
    ru_utime::Tuple{Int64, Int64}  # User CPU time (sec, usec)
    ru_stime::Tuple{Int64, Int64}  # System CPU time (sec, usec)
    ru_maxrss::Int64               # Maximum resident set size
    ru_ixrss::Int64                # Integral shared memory size
    ru_idrss::Int64                # Integral unshared data size
    ru_isrss::Int64                # Integral unshared stack size
    ru_minflt::Int64               # Page reclaims (soft page faults)
    ru_majflt::Int64               # Page faults (hard page faults)
    ru_nswap::Int64                # Swaps
    ru_inblock::Int64              # Block input operations
    ru_oublock::Int64              # Block output operations
    ru_msgsnd::Int64               # IPC messages sent
    ru_msgrcv::Int64               # IPC messages received
    ru_nsignals::Int64             # Signals received
    ru_nvcsw::Int64                # Voluntary context switches
    ru_nivcsw::Int64               # Involuntary context switches
end

FLAME_MEASUREMENTS_FILE = nothing
FLAME_MEASUREMENTS_DICT = nothing
FLAME_PREFIX = "Main"
FLAME_MEASUREMENTS_LOCK = SpinLock()

function __init__()::Nothing
    path = get(ENV, "TLU_FLAME_MEASUREMENTS_FILE", nothing)
    global FLAME_MEASUREMENTS_DICT
    global FLAME_MEASUREMENTS_FILE
    global FLAME_PREFIX
    if path === nothing
        FLAME_MEASUREMENTS_DICT = nothing
    else
        FLAME_MEASUREMENTS_DICT =  # UNTESTED
            Dict{Tuple{AbstractString, Bool}, Union{SerialMeasurement, ParallelMeasurement}}()
        FLAME_MEASUREMENTS_FILE = path  # NOLINT  # UNTESTED
        FLAME_PREFIX = Base.source_path()  # UNTESTED
        if FLAME_PREFIX === nothing  # UNTESTED
            FLAME_PREFIX = "Main"  # UNTESTED
        else
            FLAME_PREFIX = Base.basename(FLAME_PREFIX)  # UNTESTED
        end
        @info "Will append flameview measurements into: $(path) under: $(FLAME_PREFIX)" _group = :tlu_env  # UNTESTED
        atexit(__end__)  # UNTESTED # NOJET
    end
    return nothing
end

function __end__()::Nothing
    @assert FLAME_MEASUREMENTS_DICT !== nothing
    @assert FLAME_MEASUREMENTS_FILE !== nothing
    open(FLAME_MEASUREMENTS_FILE, "a") do file  # NOJET
        for ((stack, is_in_parallel), measurement) in FLAME_MEASUREMENTS_DICT
            print(file, "$(stack) $(is_in_parallel)")
            if is_in_parallel
                @assert measurement isa ParallelMeasurement
                print(file, " $(measurement.invocations)")
                print(file, " $(measurement.iterations)")
                println(file, " $(measurement.elapsed_ns)")
            else
                @assert measurement isa SerialMeasurement
                print(file, " $(measurement.invocations)")
                print(file, " $(measurement.iterations)")
                print(file, " $(measurement.elapsed_ns)")
                print(file, " $(measurement.gc_ns)")
                print(file, " $(measurement.gc_full_sweeps)")
                print(file, " $(measurement.gc_pauses)")
                print(file, " $(measurement.user_cpu_us)")
                print(file, " $(measurement.system_cpu_us)")
                print(file, " $(measurement.soft_page_faults)")
                print(file, " $(measurement.hard_page_faults)")
                print(file, " $(measurement.read_blocks)")
                print(file, " $(measurement.write_blocks)")
                print(file, " $(measurement.voluntary_context_switches)")
                println(file, " $(measurement.involuntary_context_switches)")
            end
        end
    end
    return nothing
end

@inline function get_flame_stack(private_storage)
    if FLAME_MEASUREMENTS_DICT === nothing
        return nothing
    else
        flame_stack = get(private_storage, :flame_stack, nothing)  # UNTESTED
        if flame_stack === nothing  # UNTESTED
            flame_stack = AbstractString[FLAME_PREFIX]  # UNTESTED
            private_storage[:flame_stack] = flame_stack  # UNTESTED
        end
        return flame_stack  # UNTESTED
    end
end

function get_rusage()  # UNTESTED
    rusage = Ref{RUsage}(RUsage((0, 0), (0, 0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    status = ccall(:getrusage, Cint, (Cint, Ptr{RUsage}), 0, rusage)
    @assert status == 0
    return rusage[]
end

mutable struct SerialMeasurement
    invocations::Int64
    iterations::Int64
    elapsed_ns::Int64
    gc_ns::Int64
    gc_full_sweeps::Int64
    gc_pauses::Int64
    user_cpu_us::Int64
    system_cpu_us::Int64
    soft_page_faults::Int64
    hard_page_faults::Int64
    read_blocks::Int64
    write_blocks::Int64
    voluntary_context_switches::Int64
    involuntary_context_switches::Int64
end

function Base.copy(measurement::SerialMeasurement)::SerialMeasurement  # UNTESTED
    return SerialMeasurement(
        measurement.invocations,
        measurement.iterations,
        measurement.elapsed_ns,
        measurement.gc_ns,
        measurement.gc_full_sweeps,
        measurement.gc_pauses,
        measurement.user_cpu_us,
        measurement.system_cpu_us,
        measurement.soft_page_faults,
        measurement.hard_page_faults,
        measurement.read_blocks,
        measurement.write_blocks,
        measurement.voluntary_context_switches,
        measurement.involuntary_context_switches,
    )
end

function combine_measurements(  # UNTESTED
    left_measurement::SerialMeasurement,
    right_measurement::SerialMeasurement,
)::SerialMeasurement
    return SerialMeasurement(
        left_measurement.invocations + right_measurement.invocations,
        left_measurement.iterations + right_measurement.iterations,
        left_measurement.elapsed_ns + right_measurement.elapsed_ns,
        left_measurement.gc_ns + right_measurement.gc_ns,
        left_measurement.gc_full_sweeps + right_measurement.gc_full_sweeps,
        left_measurement.gc_pauses + right_measurement.gc_pauses,
        left_measurement.user_cpu_us + right_measurement.user_cpu_us,
        left_measurement.system_cpu_us + right_measurement.system_cpu_us,
        left_measurement.soft_page_faults + right_measurement.soft_page_faults,
        left_measurement.hard_page_faults + right_measurement.hard_page_faults,
        left_measurement.read_blocks + right_measurement.read_blocks,
        left_measurement.write_blocks + right_measurement.write_blocks,
        left_measurement.voluntary_context_switches + right_measurement.voluntary_context_switches,
        left_measurement.involuntary_context_switches + right_measurement.involuntary_context_switches,
    )
end

function sort_weight(measurement::SerialMeasurement)::Int64  # UNTESTED
    return measurement.elapsed_ns
end

mutable struct ParallelMeasurement
    invocations::Int64
    iterations::Int64
    elapsed_ns::Int64
    scaled_elapsed_ns::Int64
end

function Base.copy(measurement::ParallelMeasurement)::ParallelMeasurement  # UNTESTED
    return ParallelMeasurement(
        measurement.invocations,
        measurement.iterations,
        measurement.elapsed_ns,
        measurement.scaled_elapsed_ns,
    )
end

function combine_measurements(  # UNTESTED
    left_measurement::ParallelMeasurement,
    right_measurement::ParallelMeasurement,
)::ParallelMeasurement
    return ParallelMeasurement(
        left_measurement.invocations + right_measurement.invocations,
        left_measurement.iterations + right_measurement.iterations,
        left_measurement.elapsed_ns + right_measurement.elapsed_ns,
        left_measurement.scaled_elapsed_ns + right_measurement.scaled_elapsed_ns,
    )
end

function combine_measurements(  # UNTESTED
    left_measurement::SerialMeasurement,
    right_measurement::ParallelMeasurement,
)::ParallelMeasurement
    return combine_measurements(right_measurement, left_measurement)
end

function combine_measurements(  # UNTESTED
    left_measurement::ParallelMeasurement,
    right_measurement::SerialMeasurement,
)::ParallelMeasurement
    return ParallelMeasurement(
        left_measurement.invocations + right_measurement.invocations,
        left_measurement.iterations + right_measurement.iterations,
        left_measurement.elapsed_ns + right_measurement.elapsed_ns,
        left_measurement.scaled_elapsed_ns + right_measurement.elapsed_ns,
    )
end

function sort_weight(measurement::ParallelMeasurement)::Int64  # UNTESTED
    return measurement.scaled_elapsed_ns
end

"""
    flame_timed(body::Function, name::AbstractString)::Any

Add a measurement to the flameview file for the execution of the `body` under the specified `name`. We distinguish
between "serial" measurements (in the top-level main thread) and "parallel" measurements (inside tasks invoked from
within a `parallel_loop`). For parallel code we can only meaningfully measure elapsed time. For serial code (including
complete `parallel_loop`), we measure much more - GC overhead, CPU time, page faults, IO, and context switches.

If `name` starts with `.`, then we take the name of the surrounding `flame_timed` context (which must exist) and append
the `name` to it. This allows easily marking sections inside logged functions (e.g. `flame_timed(".loop")`) in a way that
still makes sense when generating a reversed flame graph.

If `TLU_FLAME_MEASUREMENTS_FILE` was not specified, this just invokes the `body` without (almost) any overhead.

!!! note

    This is automatically invoked for every `@logged` function, which greatly reduces the need to manually annotate code
    sections with an explicit `flame_timed` call, though this is still useful when sections of a function need to be
    measured, or for wrapping calls to expensive standard library functions of interest.
"""
function flame_timed(body::Function, name::AbstractString; iterations::Integer = 0)::Any
    if FLAME_MEASUREMENTS_DICT === nothing
        return body()
    end

    start_private_storage = task_local_storage()  # UNTESTED
    start_is_in_parallel = get(start_private_storage, :is_in_parallel_loop, false)  # UNTESTED
    start_flame_stack = get_flame_stack(start_private_storage)  # UNTESTED
    @assert start_flame_stack !== nothing  # UNTESTED

    if startswith(name, ".")  # UNTESTED
        @assert length(start_flame_stack) > 0  # UNTESTED
        name = start_flame_stack[end] * name  # UNTESTED
    end

    start_elapsed_ns = time_ns()  # UNTESTED
    if !start_is_in_parallel  # UNTESTED
        gc_num = Base.gc_num()  # UNTESTED
        start_gc_ns = gc_num.total_time  # UNTESTED
        start_gc_full_sweeps = gc_num.full_sweep  # UNTESTED
        start_gc_pauses = gc_num.pause  # UNTESTED
        start_rusage = get_rusage()  # UNTESTED
        start_system_cpu_us = start_rusage.ru_stime[1] * 1000000 + start_rusage.ru_stime[2]  # UNTESTED
        start_user_cpu_us = start_rusage.ru_utime[1] * 1000000 + start_rusage.ru_utime[2]  # UNTESTED
        start_soft_page_faults = start_rusage.ru_minflt  # UNTESTED
        start_hard_page_faults = start_rusage.ru_majflt  # UNTESTED
        start_read_blocks = start_rusage.ru_inblock  # UNTESTED
        start_write_blocks = start_rusage.ru_oublock  # UNTESTED
        start_voluntary_context_switches = start_rusage.ru_nvcsw  # UNTESTED
        start_involuntary_context_switches = start_rusage.ru_nivcsw  # UNTESTED
    end

    push!(start_flame_stack, name)  # UNTESTED
    key = (join(start_flame_stack, ";"), start_is_in_parallel)  # UNTESTED

    try  # UNTESTED
        return body()  # UNTESTED

    finally
        elapsed_ns = time_ns() - start_elapsed_ns  # UNTESTED
        if start_is_in_parallel  # UNTESTED
            @assert iterations >= 0  # UNTESTED
            new_measurement = ParallelMeasurement(1, iterations, elapsed_ns, 0)  # UNTESTED
        else
            gc_ns = Base.gc_num().total_time - start_gc_ns  # UNTESTED
            gc_full_sweeps = Base.gc_num().full_sweep - start_gc_full_sweeps  # UNTESTED
            gc_pauses = Base.gc_num().pause - start_gc_pauses  # UNTESTED
            end_rusage = get_rusage()  # UNTESTED
            system_cpu_us = end_rusage.ru_stime[1] * 1000000 + end_rusage.ru_stime[2] - start_system_cpu_us  # UNTESTED
            user_cpu_us = end_rusage.ru_utime[1] * 1000000 + end_rusage.ru_utime[2] - start_user_cpu_us  # UNTESTED
            soft_page_faults = end_rusage.ru_minflt - start_soft_page_faults  # UNTESTED
            hard_page_faults = end_rusage.ru_majflt - start_hard_page_faults  # UNTESTED
            read_blocks = end_rusage.ru_inblock - start_read_blocks  # UNTESTED
            write_blocks = end_rusage.ru_oublock - start_write_blocks  # UNTESTED
            voluntary_context_switches = end_rusage.ru_nvcsw - start_voluntary_context_switches  # UNTESTED
            involuntary_context_switches = end_rusage.ru_nivcsw - start_involuntary_context_switches  # UNTESTED
            new_measurement = SerialMeasurement(  # UNTESTED
                1,
                iterations,
                elapsed_ns,
                gc_ns,
                gc_full_sweeps,
                gc_pauses,
                user_cpu_us,
                system_cpu_us,
                soft_page_faults,
                hard_page_faults,
                read_blocks,
                write_blocks,
                voluntary_context_switches,
                involuntary_context_switches,
            )
        end

        lock(FLAME_MEASUREMENTS_LOCK) do  # UNTESTED
            update!(
                FLAME_MEASUREMENTS_DICT,
                key,
                old_measurement -> combine_measurements(old_measurement, new_measurement),
                new_measurement,
            )
            return nothing
        end
        @assert pop!(start_flame_stack) == name  # UNTESTED
    end
end

"""
    finalize_flameview(;
        measurements_path::AbstractString,
        flameview_path::AbstractString,
        reversed::Bool = false,
    )::Nothing

Convert the collected measurements in the `measurement_path` into a proper `flameview` file, suitable for use with This
is needed for several reasons:

  - The measurements are inclusive (that is, the elapsed time of a step includes the elapsed time of all nested steps).
  - Measurements of serial steps (which should add to the total execution time) are mixed with the measurements of
    parallel steps (which would add up to much more).
  - We collect much more than (elapsed) time, so we need to generate a nive HTML annotation for each line to show the
    extra data.

In addition, a reversed flamegraph (showing callers instead of callees) doesn't work well when just reversing the stacks
of the flameview file, because the annotations won't be properly computed. Specify `reversed` to generate a proper
reversed flameview file.

Each `parallel_loop` is therefore measured twice, once as a serial step and once as a parallel step. We scale the
parallel time (and everything nested in it) so it adds up to the serial time. We then convert all the inclusive
measurements to exclusive ones. The result is a flameview that shows the contribution of everything (serial and parallel
steps) to the total execution time, under the assumption that the number of iterations in the parallel loops is large
enough that it is the sum of parallel steps that matters.

!!! note

    This will not work if you call [`flame_timed`](@ref) inside an explicit `@threads` loops. You must use a
    `parallel_loop` instead.

Parallel names are suffixed with `_[1]` to allow using different colors for serial and parallel code. This only works
with `flameview`; in general the generated file only works with `flameview.py`. If you want to feed it into a
`flamegraph` tool, you will need to pass it through `sed 's/ #.*//;s/_\\[1\\]/_[j]/'` - that is, get rid of the (very
useful) annotations attached to each node, and switch to a `_[j]` suffix (or any other suffix you choose that works with
`flamegraph.pl`, or even completely get rid of the suffix.)
"""
function finalize_flameview(;  # UNTESTED
    measurements_path::AbstractString,
    flameview_path::AbstractString,
    reversed::Bool = false,
)::Nothing
    raw_measurements_dict = Dict{Tuple{AbstractString, Bool}, Union{SerialMeasurement, ParallelMeasurement}}()
    for line in eachline(measurements_path)
        fields = split(line)
        stack = fields[1]
        is_in_parallel = parse(Bool, fields[2])
        if is_in_parallel
            @assert length(fields) == 5
            invocations = parse(Int64, fields[3])
            iterations = parse(Int64, fields[4])
            elapsed_ns = parse(Int64, fields[5])
            new_measurement = ParallelMeasurement(invocations, iterations, elapsed_ns, 0)
        else
            @assert length(fields) == 16
            invocations = parse(Int64, fields[3])
            iterations = parse(Int64, fields[4])
            elapsed_ns = parse(Int64, fields[5])
            gc_ns = parse(Int64, fields[6])
            gc_full_sweeps = parse(Int64, fields[7])
            gc_pauses = parse(Int64, fields[8])
            user_cpu_us = parse(Int64, fields[9])
            system_cpu_us = parse(Int64, fields[10])
            soft_page_faults = parse(Int64, fields[11])
            hard_page_faults = parse(Int64, fields[12])
            read_blocks = parse(Int64, fields[13])
            write_blocks = parse(Int64, fields[14])
            voluntary_context_switches = parse(Int64, fields[15])
            involuntary_context_switches = parse(Int64, fields[16])
            new_measurement = SerialMeasurement(
                invocations,
                iterations,
                elapsed_ns,
                gc_ns,
                gc_full_sweeps,
                gc_pauses,
                user_cpu_us,
                system_cpu_us,
                soft_page_faults,
                hard_page_faults,
                read_blocks,
                write_blocks,
                voluntary_context_switches,
                involuntary_context_switches,
            )
        end
        update!(
            raw_measurements_dict,
            (stack, is_in_parallel),
            old_measurement -> combine_measurements(old_measurement, new_measurement),
            new_measurement,
        )
    end

    folded_measurements_dict = Dict{AbstractString, Union{SerialMeasurement, ParallelMeasurement}}()
    self_measurements_dict = Dict{AbstractString, Union{SerialMeasurement, ParallelMeasurement}}()

    loop_scale = nothing
    loop_stack = nothing

    previous_stack = nothing
    previous_is_in_parallel = nothing
    previous_measurement = nothing

    sorted = sort!(collect(raw_measurements_dict); by = pair -> pair[1])
    for ((stack, is_in_parallel), measurement) in sorted
        if stack == previous_stack
            @assert is_in_parallel
            @assert previous_is_in_parallel === false
            @assert previous_measurement isa SerialMeasurement
            @assert measurement isa ParallelMeasurement
            @assert previous_measurement.iterations == -measurement.invocations
            loop_scale = min(previous_measurement.elapsed_ns / measurement.elapsed_ns, 1.0)  # NOJET
            loop_stack = stack
        else
            if is_in_parallel
                @assert measurement isa ParallelMeasurement
                @assert loop_stack !== nothing
                @assert startswith(stack, loop_stack)  # NOJET
                @assert length(stack) > length(loop_stack)  # NOJET
                measurement.scaled_elapsed_ns = Int64(round(measurement.elapsed_ns * loop_scale))  # NOJET
            else
                @assert measurement isa SerialMeasurement
            end
            folded_measurements_dict[stack] = measurement
            self_measurements_dict[stack] = copy(measurement)
        end

        previous_stack = stack
        previous_is_in_parallel = is_in_parallel
        previous_measurement = measurement
    end

    saved = [(stack, copy(measurement)) for (stack, measurement) in self_measurements_dict]
    for (stack, measurement) in saved
        semicolon_index = findlast(';', stack)
        if semicolon_index !== nothing
            parent_stack = stack[1:(semicolon_index - 1)]
            parent_measurement = get(self_measurements_dict, parent_stack, nothing)
            if parent_measurement isa ParallelMeasurement
                @assert measurement isa ParallelMeasurement
                parent_measurement.elapsed_ns -= measurement.elapsed_ns
                parent_measurement.scaled_elapsed_ns -= measurement.scaled_elapsed_ns
            elseif parent_measurement isa SerialMeasurement
                if measurement isa ParallelMeasurement
                    parent_measurement.elapsed_ns -= measurement.scaled_elapsed_ns
                else
                    @assert measurement isa SerialMeasurement
                    parent_measurement.elapsed_ns -= measurement.elapsed_ns
                    parent_measurement.gc_ns -= measurement.gc_ns
                    parent_measurement.gc_full_sweeps -= measurement.gc_full_sweeps
                    parent_measurement.gc_pauses -= measurement.gc_pauses
                    parent_measurement.user_cpu_us -= measurement.user_cpu_us
                    parent_measurement.system_cpu_us -= measurement.system_cpu_us
                    parent_measurement.soft_page_faults -= measurement.soft_page_faults
                    parent_measurement.hard_page_faults -= measurement.hard_page_faults
                    parent_measurement.read_blocks -= measurement.read_blocks
                    parent_measurement.write_blocks -= measurement.write_blocks
                    parent_measurement.voluntary_context_switches -= measurement.voluntary_context_switches
                    parent_measurement.involuntary_context_switches -= measurement.involuntary_context_switches
                end
            end
        end
    end

    if reversed
        reversed_self_measurements_dict = Dict{AbstractString, Union{SerialMeasurement, ParallelMeasurement}}()
        for (stack, self_measurement) in self_measurements_dict
            folded_measurement = folded_measurements_dict[stack]  # NOLINT
            reversed_names = reverse!(split(stack, ";"))
            reversed_stack = join(reversed_names, ";")
            reversed_self_measurements_dict[reversed_stack] = self_measurement
        end

        reversed_folded_measurements_dict = Dict{AbstractString, Union{SerialMeasurement, ParallelMeasurement}}()
        reversed_sorted_reversed_stacks = sort!(collect(keys(reversed_self_measurements_dict)); rev = true)
        for reversed_stack in reversed_sorted_reversed_stacks
            self_measurement = reversed_self_measurements_dict[reversed_stack]
            if self_measurement isa SerialMeasurement
                empty_measurement = SerialMeasurement(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            else
                empty_measurement = ParallelMeasurement(0, 0, 0, 0)
            end
            while true
                update!(
                    reversed_folded_measurements_dict,
                    reversed_stack,
                    old_measurement -> combine_measurements(old_measurement, self_measurement),
                    self_measurement,
                )
                semicolon_index = findlast(';', reversed_stack)
                if semicolon_index === nothing
                    break
                end
                reversed_stack = reversed_stack[1:(semicolon_index - 1)]
                update!(
                    reversed_self_measurements_dict,
                    reversed_stack,
                    old_measurement -> combine_measurements(old_measurement, empty_measurement),
                    empty_measurement,
                )
            end
        end

        self_measurements_dict = reversed_self_measurements_dict
        folded_measurements_dict = reversed_folded_measurements_dict
    end

    for (stack, folded_measurement) in folded_measurements_dict
        self_measurement = self_measurements_dict[stack]
        if self_measurement.elapsed_ns != folded_measurement.elapsed_ns
            semicolon_index = findlast(';', stack)
            if semicolon_index === nothing
                semicolon_index = 0
            end
            self_stack = stack * ";" * stack[(semicolon_index + 1):end] * ".self"  # NOJET
            self_measurements_dict[self_stack] = folded_measurements_dict[self_stack] = copy(self_measurement)
            self_measurement.elapsed_ns = 0
            if self_measurement isa ParallelMeasurement
                self_measurement.scaled_elapsed_ns = 0
            end
        end
    end

    open(flameview_path, "w") do file
        for (stack, self_measurement) in self_measurements_dict
            folded_measurement = folded_measurements_dict[stack]
            if self_measurement isa SerialMeasurement
                text = measurement_text("Calls"; samples = folded_measurement.invocations, units = "executions")
                if self_measurement.iterations != 0
                    text *=
                        "&#10;" *
                        measurement_text("Loops"; samples = abs(folded_measurement.iterations), units = "iterations")
                end
                text *=
                    "&#10;" *
                    measurement_text(
                        "Elapsed (serial)";
                        samples = folded_measurement.elapsed_ns,
                        scale = 1e9,
                        units = "seconds",
                        base = folded_measurement.elapsed_ns,
                    ) *
                    "&#10;" *
                    measurement_text(
                        "- User";
                        samples = folded_measurement.user_cpu_us * 1000,
                        scale = 1e9,
                        units = "seconds",
                        base = folded_measurement.elapsed_ns,
                    ) *
                    "&#10;" *
                    measurement_text(
                        "- System";
                        samples = folded_measurement.system_cpu_us * 1000,
                        scale = 1e9,
                        units = "seconds",
                        base = folded_measurement.elapsed_ns,
                    ) *
                    "&#10;Garbage collection:&#10;" *
                    "&#10;" *
                    measurement_text(
                        "- GC time";
                        samples = folded_measurement.gc_ns,
                        scale = 1e9,
                        units = "seconds",
                        base = folded_measurement.elapsed_ns,
                    ) *
                    "&#10;" *
                    measurement_text(
                        "- GC sweeps";
                        samples = folded_measurement.gc_full_sweeps,
                        units = "full sweeps",
                    ) *
                    "&#10;" *
                    measurement_text("- GC pauses"; samples = folded_measurement.gc_pauses, units = "pauses") *
                    "&#10;CPU:&#10;" *
                    measurement_text(
                        "- Total";
                        samples = (folded_measurement.user_cpu_us + folded_measurement.system_cpu_us) * 1000,
                        scale = 1e9,
                        units = "seconds",
                        base = folded_measurement.elapsed_ns,
                    ) *
                    "&#10;Page faults:&#10;" *
                    measurement_text("- Hard"; samples = folded_measurement.hard_page_faults, units = "blocks") *
                    "&#10;" *
                    measurement_text("- Soft"; samples = folded_measurement.soft_page_faults, units = "blocks") *
                    "&#10;Context switched:&#10;" *
                    measurement_text(
                        "- Involuntary";
                        samples = folded_measurement.involuntary_context_switches,
                        units = "switches",
                    ) *
                    "&#10;" *
                    measurement_text(
                        "- Voluntary";
                        samples = folded_measurement.voluntary_context_switches,
                        units = "switches",
                    )
                elapsed_ns = self_measurement.elapsed_ns
                if self_measurement.iterations < 0
                    if !endswith(stack, ".self")
                        elapsed_ns = 0
                    end
                    stack = stack * "_[1]"
                end
                println(file, "$(stack) $(elapsed_ns) # $(text)")
            else
                @assert self_measurement isa ParallelMeasurement
                text = measurement_text("Calls"; samples = folded_measurement.invocations, units = "executions")
                if self_measurement.iterations != 0
                    text *=
                        "&#10;" *
                        measurement_text("Loops"; samples = abs(folded_measurement.iterations), units = "iterations")
                end
                text *=
                    "&#10;" * measurement_text(
                        "Elapsed (scaled serial):";
                        samples = folded_measurement.scaled_elapsed_ns,
                        scale = 1e9,
                        units = "seconds",
                    )
                println(file, "$(stack)_[1] $(self_measurement.scaled_elapsed_ns) # $(text)")
            end
        end
    end

    return nothing
end

function measurement_text(  # UNTESTED
    name::AbstractString;
    samples::Int64,
    scale::Maybe{Real} = nothing,
    units::AbstractString,
    base::Maybe{Int64} = nothing,
)::AbstractString
    texts = [name, ": "]
    if scale === nothing
        push!(texts, delimited_number(samples))
    else
        push!(texts, delimited_number(samples / scale))
    end
    push!(texts, " ", units)
    if base !== nothing
        push!(texts, " (", replace(percent(samples, base), "<" => "&lt;", ">" => "&gt;"), ")")
    end
    return join(texts)
end

DID_WARN_ABOUT_EXPENSIVE_OPERATIONS_IN_PARALLEL = Set{Tuple{AbstractString, AbstractString}}()

function warn_about_expensive_operation_in_parallel(name::AbstractString)::Nothing  # UNTESTED
    private_storage = task_local_storage()
    is_in_parallel = get(private_storage, :is_in_parallel, false)
    if is_in_parallel
        flame_stack = get(private_storage, :flame_stack, nothing)
        if flame_stack === nothing
            flame_stack = "Main"
        else
            flame_stack = join(flame_stack, ";")
        end

        key = (flame_stack, name)
        lock(FLAME_MEASUREMENTS_LOCK) do
            if !(key in DID_WARN_ABOUT_EXPENSIVE_OPERATIONS_IN_PARALLEL)
                push!(DID_WARN_ABOUT_EXPENSIVE_OPERATIONS_IN_PARALLEL, key)
                @warn "Expensive operation: $(name) called by: $(flame_stack)"
            end
        end
    end
    return nothing
end

end
