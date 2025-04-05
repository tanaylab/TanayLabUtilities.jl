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
    brief(value::Any)::AbstractString

Provide a brief description of a `value`. This is basically `summary` but modified for specific types (in particular,
vectors and matrices) to give "better" results.

```jldoctest
using Test

@test brief(1.0) == "1.0"
@test brief(true) == "true"
@test brief(:foo) == ":foo"
@test brief(nothing) == "nothing"
@test brief(missing) == "missing"
@test brief(undef) == "undef"
@test brief(("foo", :bar)) == "(\\"foo\\", :bar)"
@test brief("foo" => :bar) == "\\"foo\\" => :bar"
@test brief("foo") == "\\"foo\\""
@test brief("foo "^10) == "\\"foo foo foo foo ...\\" (40)"
@test brief([true, false]) == "2 x Bool (Dense; 50% true)"
@test brief(Int64) == "Int64"
@test brief(String) == "Str"
@test brief(AbstractString) == "Str"

@enum Foo Bar Baz
@test brief(Bar) == "Foo::Bar"

struct Vaz end
@test brief(Vaz()) == summary(Vaz())

@test brief(Set([1])) == "1 x Int64 (Set)"

@test brief(rand(5)) == "5 x Float64 (Dense)"
@test brief(rand(3, 4)) == "3 x 4 x Float64 in Columns (Dense)"

@test brief(read_only_array(rand(5))) == "5 x Float64 (ReadOnly, Dense)"
@test brief(PermutedDimsArray(rand(3, 4), (2, 1))) == "4 x 3 x Float64 in Rows (Permute, Dense)"
@test brief(PermutedDimsArray(rand(3, 4), (1, 2))) == "3 x 4 x Float64 in Columns (!Permute, Dense)"

using SparseArrays

@test brief(SparseVector([0.0, 1.0])) == "2 x Float64 (Sparse Int64 50%)"
@test brief(SparseMatrixCSC([0.0 1.0 2.0; 3.0 4.0 0.0])) == "2 x 3 x Float64 in Columns (Sparse Int64 67%)"

using NamedArrays

@test brief(NamedArray(rand(2))) == "2 x Float64 (Named, Dense)"
@test brief(NamedArray(SparseVector([0.0, 1.0]))) == "2 x Float64 (Named, Sparse Int64 50%)"

using LinearAlgebra

@test brief(transpose(rand(2))) == "2 x Float64 (Transpose, Dense)"
@test brief(adjoint(rand(2))) == "2 x Float64 (Adjoint, Dense)"

@test brief(Dict(["a" => 1])) == "1 x Str => Int64 (Dict)"

println("OK")

# output

OK
```
"""
function brief(value::Any)::AbstractString
    try
        return "$(join(string.(size(value)), " x ")) x $(brief(eltype(value))) ($(brief(typeof(value))))"
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
        return "$(nameof(value))"
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
