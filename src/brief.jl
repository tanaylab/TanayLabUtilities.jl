"""
Functions for generating a brief description of arbitrary data. Yes, this is what the builtin `summary` is expected to
do. However, `summary` doesn't summarize arrays that well. As we don't want to override it for types we don't own, we
default `brief` to call `summary` and can then override it for whatever types we feel is necessary. For your own types,
provide an implementation for `summary` as that is the "right thing to do".
"""
module Brief

export MAX_BRIEF_STRING
export brief
export percent

using Distributed

"""
    brief(value::Any)::String

Provide a brief description of a `value`. This is basically `summary` but modified for specific types (in particular,
vectors and matrices) to give "better" results.

```jldoctest
brief(1.0)

# output

"1.0"
```

```jldoctest
brief(true)

# output

"true"
```

```jldoctest
brief(:foo)

# output

":foo"
```

```jldoctest
brief(nothing)

# output

"nothing"
```

```jldoctest
brief(missing)

# output

"missing"
```

```jldoctest
brief(undef)

# output

"undef"
```

```jldoctest
brief(("foo", :bar))

# output

"(\\"foo\\", :bar)"
```

```jldoctest
brief("foo")

# output

"\\"foo\\""
```

```jldoctest
brief("foo "^10)

# output

"\\"foo foo foo foo ...\\" (40)"
```

```jldoctest
@enum Foo Bar Baz

brief(Bar)

# output

"Foo::Bar"
```

```jldoctest
struct Foo end

brief(Foo())

# output

"Foo"
```

```jldoctest
brief([true, false])

# output

"2 x Bool (Dense; 50% true)"
```
"""
function brief(value::Any)::String
    try
        return "$(eltype(value)) x $(join(string.(size(value)), " x ")) $(nameof(typeof(value)))"
    catch
        try
            return "$(eltype(value)) x $(length(value)) $(nameof(typeof(value)))"
        catch
            return summary(value)
        end
    end
end

function brief(value::Tuple)::String
    return "(" * join([brief(entry) for entry in value], ", ") * ")"
end

function brief(value::Union{Real, Type, Nothing, Missing})::String
    return "$(value)"
end

function brief(value::Enum)::String
    return "$(nameof(typeof(value)))::$(value)"
end

function brief(::UndefInitializer)::String
    return "undef"
end

"""
The maximal length of strings we show as-is in [`brief`](@ref). We only show a prefix of longer strings (followed by
their length).
"""
MAX_BRIEF_STRING::Integer = 16

function brief(value::AbstractString)::String
    if length(value) <= MAX_BRIEF_STRING
        return "\"$(value)\""
    else
        return "\"$(value[1:MAX_BRIEF_STRING])...\" ($(length(value)))"
    end
end

function brief(value::Symbol)::String
    return ":$(value)"
end

"""
    percent(used::Real, out_of::Real)::String

Format a fraction of `used` amount `out_of` some total, as an integer percent value. Very small fractions are denoted as
`<1%` and very large fractions are denoted as `>99%`. We use this to show the percent of `true` values in masks, and the
percent of non-zero entries in sparse arrays.

```jldoctest
percent(0, 0)

# output

"NA%"
```

```jldoctest
percent(0, 1000)

# output

"0%"
```

```jldoctest
percent(9, 1000)

# output

"<1%"
```

```jldoctest
percent(10, 1000)

# output

"1%"
```

```jldoctest
percent(11, 1000)

# output

"1%"
```

```jldoctest
percent(990, 1000)

# output

"99%"
```

```jldoctest
percent(991, 1000)

# output

">99%"
```

```jldoctest
percent(1000, 1000)

# output

"100%"
```
"""
function percent(used::Real, out_of::Real)::String
    @assert 0 <= used <= out_of

    if out_of == 0
        return "NA%"
    end

    if used == 0
        return "0%"
    end

    if used == out_of
        return "100%"
    end

    float_percent = 100.0 * Float64(used) / Float64(out_of)

    if float_percent < 1
        return "<1%"

    elseif float_percent > 99
        return ">99%"

    else
        int_percent = round(Int64, float_percent)
        return "$(int_percent)%"
    end
end

end  # module
