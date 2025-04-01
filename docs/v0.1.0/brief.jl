"""
Functions for generating a brief description of arbitrary data. Yes, this is what the builtin `summary` is expected to
do. However, `summary` doesn't summarize arrays that well. As we don't want to override it for types we don't own, we
default `brief` to call `summary` and can then override it for whatever types we feel is necessary. For your own types,
provide an implementation for `summary` as that is the "right thing to do".
"""
module Brief

export brief
export percent

using Distributed
#using LinearAlgebra
#using NamedArrays
#using SparseArrays

"""
    brief(value::Any)::String

Provide a brief description of a `value`. This is basically `summary` but modified for specific types (in particular,
vectors and matrices) to give "better" results.

```jldoctest
using TanayLabUtilities.Brief

brief(1.0)

# output

"1.0 (Float64)"
```

```jldoctest
using TanayLabUtilities.Brief

brief(true)

# output

"true"
```

```jldoctest
using TanayLabUtilities.Brief

brief(:foo)

# output

":foo"
```

```jldoctest
using TanayLabUtilities.Brief

brief(nothing)

# output

"nothing"
```

```jldoctest
using TanayLabUtilities.Brief

brief(missing)

# output

"missing"
```

```jldoctest
using TanayLabUtilities.Brief

brief(undef)

# output

"undef"
```

```jldoctest
using TanayLabUtilities.Brief

brief(("foo", :bar))

# output

"(\"foo\", :bar)"
```

```jldoctest
using TanayLabUtilities.Brief

brief("foo")

# output

"\"foo\""

```jldoctest
using TanayLabUtilities.Brief

brief("foo " ^ 10)

# output

"\"foo foo foo foo ...\"" (40)
```
"""
function brief(value::Any)::String
    return summary(value)
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

function brief(value::Real)::String
    return "$(value) ($(typeof(value)))"
end

function brief(value::Tuple)::String
    return "(" * join([brief(entry) for entry in value], ", ") * ")"
end

function brief(value::Union{Bool, Type, Nothing, Missing})::String
    return "$(value)"
end

function brief(::UndefInitializer)::String
    return "undef"
end

function brief(value::AbstractString)::String
    if length(value) < 16
        return "\"$(value)\""
    else
        return "\"$(value[1:16])...\" ($(length(value)))"
    end
end

function brief(value::Symbol)::String
    return ":$(value)"
end

function brief(array::AbstractArray)::String
    if eltype(array) == Bool
        n_trues = sum(array)  # NOJET
        suffix = " ($(n_trues) true, $(percent(n_trues, length(array))))"
    else
        suffix = ""
    end
    return "$(eltype(value)) x $(join(string.(size(value)), " x ")) $(nameof(typeof(value)))$(suffix)"
end

"""
    percent(used::Real, out_of::Real)::String

Format a fraction of `used` amount `out_of` some total, as an integer percent value. Very small fractions are denoted as
`<1%` and very large fractions are denoted as `>99%`. We use this to show the percent of `true` values in masks, and the
percent of non-zero entries in sparse arrays.

```jldoctest
using TanayLabUtilities.Brief

percent(0, 0)

# output

"NA%"
```

```jldoctest
using TanayLabUtilities.Brief

percent(0, 1000)

# output

"0%"
```

```jldoctest
using TanayLabUtilities.Brief

percent(9, 1000)

# output

"<1%"
```

```jldoctest
using TanayLabUtilities.Brief

percent(10, 1000)

# output

"1%"
```

```jldoctest
using TanayLabUtilities.Brief

percent(11, 1000)

# output

"1%"
```

```jldoctest
using TanayLabUtilities.Brief

percent(990, 1000)

# output

"99%"
```

```jldoctest
using TanayLabUtilities.Brief

percent(991, 1000)

# output

">99%"
```

```jldoctest
using TanayLabUtilities.Brief

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
