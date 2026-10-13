"""
Functions for finding duplicate values.
"""
module Duplicates

export indices_of_duplicates

"""
    indices_of_duplicates(
        values::AbstractVector{T};
        empty_value::Any = nothing,
    )::AbstractDict{T, <:AbstractVector{<:Integer}} where {T}

Return the indices of the duplicate `values`. This is a `Dict` from each value which appears more than once, to the
indices of its appearances, in increasing order. If the values are all unique, this is empty.

The `empty_value` (by default, `nothing`) stands for no data, so it is not included in the result, even if it appears
more than once (e.g., an empty string). The values are compared to it using `isequal`.

```jldoctest
using TanayLabUtilities.Duplicates

sort(collect(indices_of_duplicates(["a", "b", "a", "c", "b", "a"])))

# output

2-element Vector{Pair{String, Vector{Int64}}}:
 "a" => [1, 3, 6]
 "b" => [2, 5]
```

```jldoctest
using TanayLabUtilities.Duplicates

indices_of_duplicates(["a", "", "b", "", "a"]; empty_value = "")

# output

Dict{String, Vector{Int64}} with 1 entry:
  "a" => [1, 5]
```

```jldoctest
using TanayLabUtilities.Duplicates

isempty(indices_of_duplicates([1, 2, 3]))

# output

true
```
"""
function indices_of_duplicates(
    values::AbstractVector{T};
    empty_value::Any = nothing,
)::AbstractDict{T, <:AbstractVector{<:Integer}} where {T}
    indices_of_value = Dict{T, Vector{Int}}()
    for index in eachindex(values)
        value = values[index]
        if !isequal(value, empty_value)
            push!(get!(Vector{Int}, indices_of_value, value), index)
        end
    end
    filter!(entry -> length(entry.second) > 1, indices_of_value)
    return indices_of_value
end

end  # module
