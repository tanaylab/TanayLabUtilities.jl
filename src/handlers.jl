"""
ma
Functions for handling abnormal conditions.
"""
module Handlers

export AbnormalHandler
export ErrorHandler
export IgnoreHandler
export WarnHandler
export handle_abnormal

"""
The action to take when encountering an "abnormal" (but recoverable) operation.

Valid values are:

`IgnoreHandler` - ignore the issue and perform the recovery operation.

`WarnHandler` - emit a warning using `@warn` and perform the recovery operation.

`ErrorHandler` - abort the program with an error message.
"""
@enum AbnormalHandler IgnoreHandler WarnHandler ErrorHandler

"""
    handle_abnormal(message::Function, handler::AbnormalHandler)::Nothing
    handle_abnormal(handler::AbnormalHandler, message::AbstractString)::Nothing

Call this when encountering some abnormal, but recoverable, condition. Follow it by the recovery code
(`handle_abnormal(abnormal_handler, "message"); recovery...` or
`handle_abnormal(abnormal_handler) do return "...message..." end; recovery...`).

This will `error` if the `handler` is `ErrorHandler`, and abort the program with the `message`. If it is `WarnHandler`,
it will just `@warn` and return. If it is `IgnoreHandler` it will just return.

If `message` is a function, it should return the actual message to `error` with.

```jldoctest
handle_abnormal(IgnoreHandler, "message")

# output

```

```jldoctest
handle_abnormal(IgnoreHandler) do
    @assert false
end

# output

```

```jldoctest
handle_abnormal(ErrorHandler, "message")

# output

ERROR: message
```

```jldoctest
handle_abnormal(ErrorHandler) do
    return "message"
end

# output

ERROR: message
```

```jldoctest; filter = r"@ TanayLabUtilities.Handlers.*"
handle_abnormal(WarnHandler, "message")

# output

┌ Warning: message
└ @ TanayLabUtilities.Handlers
```

```jldoctest; filter = r"@ TanayLabUtilities.Handlers.*"
handle_abnormal(WarnHandler) do
    return "message"
end

# output

┌ Warning: message
└ @ TanayLabUtilities.Handlers
```
"""
function handle_abnormal(handler::AbnormalHandler, message::AbstractString)::Nothing
    if handler == IgnoreHandler
        return nothing
    end

    if handler == WarnHandler
        @warn message
        return nothing
    end

    @assert handler == ErrorHandler
    error(message)

    @assert false
end

function handle_abnormal(message::Function, handler::AbnormalHandler)::Nothing
    if handler == IgnoreHandler
        return nothing
    end

    if handler == WarnHandler
        @warn message()
        return nothing
    end

    @assert handler == ErrorHandler
    error(message())

    @assert false
end

end  # module
