"""
Functions for generating a brief description of arbitrary data. Yes, this is what the builtin `summary` is expected to
do. However, `summary` doesn't summarize arrays that well. As we don't want to override it for types we don't own, we
default `brief` to call `summary` and can then override it for whatever types we feel is necessary. For your own types,
provide an implementation for `summary` as that is the "right thing to do".
"""
module Brief

export MAX_BRIEF_STRING
export brief
export delimited_number
export percent

using Distributed

"""
    brief(value::Any)::AbstractString

Provide a brief description of a `value`. This is basically `summary` but modified for specific types (in particular,
vectors and matrices) to give "better" results.

```jldoctest
@assert brief(1.0) == "1.0"
@assert brief(true) == "true"
@assert brief(:foo) == ":foo"
@assert brief(nothing) == "nothing"
@assert brief(missing) == "missing"
@assert brief(undef) == "undef"
@assert brief(("foo", :bar)) == "(\\"foo\\", :bar)"
@assert brief("foo" => :bar) == "\\"foo\\" => :bar"
@assert brief("foo") == "\\"foo\\""
@assert brief("foo "^10) == "\\"foo foo foo foo ...\\" (40)"
@assert brief([true, false]) == "2 x Bool (Dense; 1 (50%) true)"
@assert brief(Int64) == "Int64"
@assert brief(String) == "Str"
@assert brief(AbstractString) == "Str"

@enum Foo Bar Baz
@assert brief(Bar) == "Foo::Bar"

struct Vaz end
@assert brief(Vaz()) == summary(Vaz())

@assert brief(Set([1])) == "1 x Int64 (Set)"

@assert brief(rand(5)) == "5 x Float64 (Dense)"
@assert brief(rand(3, 4)) == "3 x 4 x Float64 in Columns (Dense)"

@assert brief(read_only_array(rand(5))) == "5 x Float64 (ReadOnly, Dense)"
@assert brief(PermutedDimsArray(rand(3, 4), (2, 1))) == "4 x 3 x Float64 in Rows (Permute, Dense)"
@assert brief(PermutedDimsArray(rand(3, 4), (1, 2))) == "3 x 4 x Float64 in Columns (!Permute, Dense)"

using SparseArrays

@assert brief(SparseVector([0.0, 1.0])) == "2 x Float64 (Sparse 1 (50%) [Int64])"
@assert brief(SparseMatrixCSC([0.0 1.0 2.0; 3.0 4.0 0.0])) == "2 x 3 x Float64 in Columns (Sparse 4 (67%) [Int64])"

using NamedArrays

@assert brief(NamedArray(rand(2))) == "2 x Float64 (Named, Dense)"
@assert brief(NamedArray(SparseVector([0.0, 1.0]))) == "2 x Float64 (Named, Sparse 1 (50%) [Int64])"

using LinearAlgebra

@assert brief(transpose(rand(2))) == "2 x Float64 (Transpose, Dense)"
@assert brief(adjoint(rand(2))) == "2 x Float64 (Adjoint, Dense)"

@assert brief(Dict(["a" => 1])) == "1 x Str => Int64 (Dict)"

# output

```
"""
function brief(value::Any)::AbstractString
    try
        return "$(join(string.(size(value)), " x ")) x $(brief(eltype(value))) ($(brief(typeof(value))))"  # NOJET
    catch
        try
            return "$(length(value)) x $(brief(eltype(value))) ($(brief(typeof(value))))"
        catch
            return summary(value)
        end
    end
end

function brief(value::Type)::AbstractString
    if value <: AbstractString
        return "Str"
    else
        try
            value = nameof(value)
        catch  # UNTESTED
        end
        return "$(value)"
    end
end

function brief(value::Tuple)::AbstractString
    return "(" * join([brief(entry) for entry in value], ", ") * ")"
end

function brief(value::Pair)::AbstractString
    return "$(brief(value[1])) => $(brief(value[2]))"
end

function brief(value::Union{Real, Type, Nothing, Missing})::AbstractString
    return "$(value)"
end

function brief(value::Enum)::AbstractString
    return "$(nameof(typeof(value)))::$(value)"
end

function brief(::UndefInitializer)::AbstractString
    return "undef"
end

function brief(value::AbstractDict)::AbstractString
    return "$(length(value)) x $(brief(keytype(value))) => $(brief(valtype(value))) ($(brief(typeof(value))))"
end

"""
The maximal length of strings we show as-is in [`brief`](@ref). We only show a prefix of longer strings (followed by
their length).
"""
MAX_BRIEF_STRING::Integer = 16

function brief(value::AbstractString)::AbstractString
    if length(value) <= MAX_BRIEF_STRING
        return "\"$(value)\""
    else
        return "\"$(value[1:MAX_BRIEF_STRING])...\" ($(length(value)))"
    end
end

function brief(value::Symbol)::AbstractString
    return ":$(value)"
end

"""
    percent(used::Real, out_of::Real)::AbstractString

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
function percent(used::Real, out_of::Real)::AbstractString
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

    if 0 < float_percent < 1
        return "<1%"

    elseif 99 < float_percent < 100
        return ">99%"

    else
        int_percent = round(Int64, float_percent)
        return "$(int_percent)%"
    end
end

const LEFT_REGEX = r"(?<=\d)(?=(?:\d{3})+(?!\d))"
const RIGHT_REGEX = r"(\d{3})(?=\d)"

"""
    delimited_number(number::Number; decimal=".", delim=",")::AbstractString

Convert a number to a string using a delimiter for every 3 digits. No, `Format` doesn't do that - it doesn't add
delimiters to the decimal fraction.

```jldoctest
println(delimited_number(123))
println(delimited_number(1.0))
println(delimited_number(1234567))
println(delimited_number(1.2345678))
println(delimited_number(1.234567e9))

# output

123
1.0
1,234,567
1.234,567,8
1.234,567e9
```
"""
function delimited_number(number::Number; decimal = ".", delim = ",")::AbstractString
    raw = string(number)
    parts = split(raw, "e")
    if length(parts) == 2
        suffix = parts[2]
        raw = parts[1]
    else
        suffix = nothing
    end
    parts = split(raw, ".")
    left_part = parts[1]
    left_part = replace(left_part, LEFT_REGEX => delim)
    if length(parts) == 1
        delimited = left_part
    elseif length(parts) == 2
        right_part = parts[2]
        right_part = replace(right_part, RIGHT_REGEX => SubstitutionString("\\1" * delim))  # NOJET
        delimited = left_part * decimal * right_part  # NOJET
    else
        @assert false
    end
    if suffix === nothing
        return delimited  # NOJET
    else
        return delimited * "e" * suffix  # NOJET
    end
end

end  # module
