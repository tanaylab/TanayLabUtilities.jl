"""
Cache big expensive objects (that consume a lot of OS resources).
"""
module GlobalWeakCache

export get_through_global_weak_cache

using Base.Threads

GLOBAL_LOCK = ReentrantLock()

GLOBAL_CACHE = Dict{Tuple{AbstractString, Any}, Tuple{Float64, WeakRef}}()

"""
    get_through_global_weak_cache(getter::Function, path::AbstractString, key::Any)::Any

Get some object associated with a `path` and identified by the `key from the global cache (if it is there) or invoke the
`getter` to obtain it (and cache it for future reference). As a convenience, the `getter` is passed the `path` and
`key`. If `purge` is set, then any existing value is ignored and is replaced by the result of the `getter`. If the
`mtime` of the file does not match the value when the object was cached, it is likewise ignored and `getter` is called
to fetch a fresh value.

There's a singleton `GLOBAL_CACHE` and all accesses are protected by a single `GLOBAL_LOCK`, so while this is
thread-safe, it does not support parallelism. It is meant for "expensive" OS operations such as `mmap`, not for
memoizing functions results for performance.

The cache holds weak references to the objects, so they are let go if there remain no active references to them (outside
the cache).

!!! note

    Use absolute paths to ensure correct results.

```jldoctest
using Test

first = get_through_global_weak_cache("fake path", :test) do _
    return [1]
end
@test first == [1]
@test first === get_through_global_weak_cache("fake path", :test) do _
    @assert false
end

second = get_through_global_weak_cache("fake path", :test; purge = true) do _
    return [2]
end
@assert second == [2]

first = second = nothing
gc()
third = get_through_global_weak_cache("fake path", :test) do _
    return [3]
end
@assert third == [3]

println("OK")

# output

OK
"""
function get_through_global_weak_cache(getter::Function, path::AbstractString, key::Any; purge::Bool = false)::Any
    return lock(GLOBAL_LOCK) do
        current_mtime = stat(path).mtime
        cached_mtime = nothing
        cached_weak_ref = nothing

        if !purge
            cached = get(GLOBAL_CACHE, (path, key), nothing)
            if cached !== nothing
                cached_mtime, cached_weak_ref = cached
            end
        end

        if cached_weak_ref === nothing || cached_weak_ref.value === nothing || current_mtime != cached_mtime
            @debug "Get $(key) for $(path)"
            value = getter(key)
            GLOBAL_CACHE[(path, key)] = (current_mtime, WeakRef(value))
        else
            @debug "Reuse $(key) for $(path)"
            value = cached_weak_ref.value
        end

        return value
    end
end

end
