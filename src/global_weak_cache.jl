"""
Cache big expensive objects (that consume a lot of OS resources).
"""
module GlobalWeakCache

export get_through_global_weak_cache

using Base.Threads

GLOBAL_LOCK = ReentrantLock()

GLOBAL_CACHE = Dict{Any, WeakRef}()

"""
    get_through_global_weak_cache(getter::Function, key::Any)::Any

Given some unique `key`, fetch the associated object from the global cache (if it is there) or invoke the `getter` to
obtain it (and cache it for future reference). As a convenience, the `getter` is passed the `key`.

There's a singleton `GLOBAL_CACHE` and all accesses are protected by a single `GLOBAL_LOCK`, so while this is
thread-safe, it does not support parallelism. It is meant for heavy operations such as opening files etc., not for
memoizing functions results for performance.

The cache holds weak references to the objects, so they are let go if there remain no active references to them (outside
the cache).

```jldoctest
using Test

first = get_through_global_weak_cache(:test) do _
    return [1]
end
@test first == [1]
@test first === get_through_global_weak_cache(:test) do _
    @assert false
end

first = nothing
gc()
second = get_through_global_weak_cache(:test) do _
    return [2]
end
@assert second == [2]

println("OK")

# output

OK
"""
function get_through_global_weak_cache(getter::Function, key::Any)::Any
    return lock(GLOBAL_LOCK) do
        weak_ref = get(GLOBAL_CACHE, key, nothing)
        if weak_ref === nothing || weak_ref.value === nothing
            @debug "Get $(key)"
            value = getter(key)
            GLOBAL_CACHE[key] = WeakRef(value)
        else
            @debug "Reuse $(key)"
            value = weak_ref.value
        end
        return value
    end
end

end
