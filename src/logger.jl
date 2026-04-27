"""
Setup a global logger the way we like it.
"""
module Logger

export @logged
export setup_logger

using ..Brief
using ..FlameTime
using ..Types
using Base.Threads
using Dates
using Distributed
using ExprTools
using Logging

"""
    setup_logger(
        io::IO = stderr;
        [level::LogLevel = Info,
        show_time::Bool = true,
        show_module::Bool = true,
        show_location::Bool = false]
    )::Nothing

Setup a global logger that will print into `io`.

By default, this will only print warnings. Note that increasing the log level will apply to **everything**. An
alternative is to set up the environment variable `JULIA_DEBUG` to a comma-separated list of modules you wish to see the
debug messages of.

If `show_time`, each message will be prefixed with a `yyyy-dd-mm HH:MM:SS.sss` timestamp prefix.

If `show_module`, each message will be prefixed with the name of the module emitting the message.

If `show_location`, each message will be prefixed with the file name and the line number emitting the message.

!!! note

    When multi-processing is used, a `P<id>:` process index is added to the log entries. When multi-threading is used, a
    `T<id>:` thread index is added to the log entries, as well as a `K<id>:` task index. To generate the latter, this
    stores a unique `:task_id` index in the `task_local_storage`. This is important since a task may migrate between
    threads.
"""
function setup_logger(
    io::IO = stderr;
    level::LogLevel = Info,
    show_time::Bool = true,
    show_module::Bool = true,
    show_location::Bool = false,
)::Nothing
    global_logger(
        ConsoleLogger(io, level; meta_formatter = (args...) -> metafmt(show_time, show_module, show_location, args...)),
    )
    return nothing
end

"""
    @logged function something(...)
        return ...
    end

    @logged(group) function something(...)
        return ...
    end

Automatically log (in `Debug` level) every invocation to the function. This will also log the values of the arguments.
Emits a second log entry when the function returns, with the result (if any). If `group` is specified, the log messages
will be under that `_group`. Otherwise the log messages are generated under the invoking module's name.

```jldoctest
@logged function bar()::Nothing
    return nothing
end

@logged function foo(positional; named = 1 + 2)
    bar()
    return positional + named
end

using Test
using Logging
logger = TestLogger(; min_level = Logging.Debug)
with_logger(logger) do
    return foo(1; named = 2)
end
print(join(["\$(record.level) : \$(record.message)" for record in logger.logs], "\n"))

# output

Debug : foo {
Debug : - positional: 1
Debug : - named: 2
Debug : bar {
Debug : bar return }
Debug : foo return: 3 }
```
"""
macro logged(group, definition)
    return do_logged(group, __source__, __module__, definition)
end

macro logged(definition)
    return do_logged(nothing, __source__, __module__, definition)
end

function do_logged(debug_group, _source_, _module_, definition)
    while definition.head === :macrocall
        definition = macroexpand(_module_, definition)  # UNTESTED
    end

    inner_definition = ExprTools.splitdef(definition)
    outer_definition = copy(inner_definition)

    function_name = get(inner_definition, :name, nothing)
    if function_name === nothing
        error("@logged requires a named function")  # UNTESTED
    end
    @assert function_name isa Symbol
    function_module = _module_
    function_file = string(_source_.file)
    function_line = _source_.line
    full_name = "$(function_module).$(function_name)"

    has_result = get(inner_definition, :rtype, :Any) != :Nothing
    arg_names = [parse_arg(arg) for arg in get(outer_definition, :args, [])]
    inner_definition[:name] = Symbol(function_name, :_logged)
    function_name = string(function_name)
    if startswith(full_name, "TanayLabUtilities.") || contains(full_name, ".TanayLabUtilities.")
        outer_definition[:body] = Expr(  # UNTESTED
            :call,
            :(GenericLogging.logged_wrapper(
                $debug_group,
                $function_module,
                $function_file,
                $function_line,
                $function_name,
                $full_name,
                $arg_names,
                $has_result,
                $(ExprTools.combinedef(inner_definition)),
            )),
            pass_args(false, get(outer_definition, :args, []))...,
            pass_args(true, get(outer_definition, :kwargs, []))...,
        )
    else
        outer_definition[:body] = Expr(
            :call,
            :(TanayLabUtilities.Logger.logged_wrapper(
                $debug_group,
                $function_module,
                $function_file,
                $function_line,
                $function_name,
                $full_name,
                $arg_names,
                $has_result,
                $(ExprTools.combinedef(inner_definition)),
            )),
            pass_args(false, get(outer_definition, :args, []))...,
            pass_args(true, get(outer_definition, :kwargs, []))...,
        )
    end

    return esc(ExprTools.combinedef(outer_definition))
end

function parse_arg(arg::Symbol)::AbstractString
    return split(string(arg), "::"; limit = 2)[1]
end

function parse_arg(arg::Expr)::AbstractString
    return parse_arg(arg.args[1])
end

function logged_wrapper(  # UNTESTED
    _group::Maybe{Symbol},
    _module::Module,
    _file::AbstractString,
    _line::Integer,
    function_name::AbstractString,
    full_name::AbstractString,
    arg_names::AbstractVector{<:AbstractString},
    has_result::Bool,
    inner_function,
)
    return (args...; kwargs...) -> (
        @debug "$(full_name) {" _module = _module _file = _file _line = _line _group = _group;
        for (arg_name, arg_value) in zip(arg_names, args)
            @debug "- $(arg_name): $(brief(arg_value))" _module = _module _file = _file _line = _line _group = _group
        end;
        for (arg_name, arg_value) in kwargs
            @debug "- $(arg_name): $(brief(arg_value))" _module = _module _file = _file _line = _line _group = _group
        end;
        result = flame_timed(function_name) do
            return inner_function(args...; kwargs...)
        end;
        if has_result
            @debug "$(full_name) return: $(brief(result)) }" _module = _module _file = _file _line = _line _group =
                _group
        else
            @debug "$(full_name) return }" _module = _module _file = _file _line = _line _group = _group
        end;
        result
    )
end

NEXT_TASK_ID = Atomic{Int}(1)

function metafmt(
    show_time::Bool,
    show_module::Bool,
    show_location::Bool,
    level::LogLevel,
    _module,
    ::Any,
    ::Any,
    file::AbstractString,
    line::Maybe{Integer},
)::Tuple{Symbol, AbstractString, AbstractString}
    @nospecialize
    color = Logging.default_logcolor(level)
    prefix_parts = []
    if show_time
        push!(prefix_parts, Dates.format(now(), "yyyy-mm-dd HH:MM:SS.sss"))
    end
    if nprocs() > 1
        push!(prefix_parts, "P$(myid())")  # UNTESTED
    end
    if nthreads() > 1
        push!(prefix_parts, "T$(threadid())")
        task_id = get!(task_local_storage(), :task_id) do
            return atomic_add!(NEXT_TASK_ID, 1)
        end
        push!(prefix_parts, "K$(task_id)")
    end
    push!(prefix_parts, string(level == Warn ? "Warning" : string(level)))
    if show_module
        push!(prefix_parts, string(_module))
    end
    if show_location
        if line === nothing  # UNTESTED
            push!(prefix_parts, "$(file)")  # UNTESTED
        else
            push!(prefix_parts, "$(file):$(line)")  # UNTESTED
        end
    end
    prefix = join(prefix_parts, ": ") * ":"
    return color, prefix, ""
end

function pass_args(is_named::Bool, args)::Vector{Union{Expr, Symbol}}
    return [pass_arg(is_named, arg) for arg in args]  # NOJET
end

function pass_arg(is_named::Bool, arg::Symbol)::Union{Expr, Symbol}
    arg = Symbol(parse_arg(arg))
    if is_named
        return Expr(:kw, arg, arg)
    else
        return arg
    end
end

function pass_arg(is_named::Bool, arg::Expr)::Union{Expr, Symbol}
    return pass_arg(is_named, arg.args[1])
end

end  # module
