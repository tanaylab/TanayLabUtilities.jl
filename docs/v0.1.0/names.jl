"""
Functions for generating names.
"""
module Names

export unique_name

using Distributed

UNIQUE_NAME_PREFIXES = Dict{AbstractString, Int64}()
UNIQUE_NAME_LOCK = ReentrantLock()

"""
    unique_name(prefix::AbstractString, separator::AbstractString = "#")::AbstractString

Using short, human-readable unique names for things is a great help when debugging. Normally one has to choose between
using a human-provided short non-unique name, and an opaque object identifier, or a combination thereof. This function
replaces the opaque object identifier with a short counter, which gives names that are both unique and short.

That is, this will return a unique name starting with the `prefix` and followed by the `separator` (by default, `#`),
the process index (if using multiple processes), and an index (how many times this name was used in the process). For
example, `unique_name("foo")` will return `foo` for the first usage, `foo#2` for the 2nd, etc. If using multiple
processes, it will return `foo`, `foo#1.2`, etc.

That is, for code where the names are unique (e.g., a simple script or Jupyter notebook), this doesn't mess up the
names. It only appends a suffix to the names if it is needed to disambiguate between multiple uses of the same name.

To help with tests, if the `prefix` contains `!`, we return it as-is, accepting it may not be unique.

# Example

```jldoctest
using TanayLabUtilities.Names

unique_name("foo")

# output

"foo"
```

# Second usage

```jldoctest
using TanayLabUtilities.Names

unique_name("foo")

# output

"foo#2"
```

# Force not unique

```jldoctest
using TanayLabUtilities.Names

unique_name("foo!")

# output

"foo!"
```

# Second usage

```jldoctest
using TanayLabUtilities.Names

unique_name("foo!")

# output

"foo!"
```
"""
function unique_name(prefix::AbstractString, separator::AbstractString = "#")::AbstractString
    if contains(prefix, '!')
        return String(prefix)
    end

    counter = lock(UNIQUE_NAME_LOCK) do
        global UNIQUE_NAME_PREFIXES
        counter = get(UNIQUE_NAME_PREFIXES, prefix, 0)
        counter += 1
        UNIQUE_NAME_PREFIXES[prefix] = counter
        return counter
    end

    if counter == 1
        return prefix
    elseif nprocs() > 1
        return "$(prefix)$(separator)$(myid()).$(counter)"  # UNTESTED
    else
        return "$(prefix)$(separator)$(counter)"
    end
end

end  # module
