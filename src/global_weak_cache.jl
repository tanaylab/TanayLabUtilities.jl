"""
Cache big expensive objects (that consume a lot of OS resources).
"""
module GlobalWeakCache

export get_through_global_weak_cache

using Base.Threads

GLOBAL_LOCK = ReentrantLock()

GLOBAL_CACHE = Dict{Tuple{AbstractString, Any}, Tuple{Float64, UInt, WeakRef}}()

"""
    get_through_global_weak_cache(getter::Function, path::AbstractString, key::Any)::Any

Get some object associated with a `path` and identified by the `key from the global cache (if it is there) or invoke the
`getter` to obtain it (and cache it for future reference). As a convenience, the `getter` is passed the `path` and
`key`. If `purge` is set, then any existing value is ignored and is replaced by the result of the `getter`. The file's
inode is always compared, so a cached entry is invalidated if the file at `path` has been replaced (for example, by
`rm` followed by re-creation). By default, the `mtime` of the file is **not** checked: callers are assumed to be the
sole writers of the file, and their own writes (which bump the `mtime`) must not throw away a cached object that is
still in use. If the caller instead wants to treat external `mtime` changes as invalidation (for example, because the
file may be rewritten by an unrelated process), set `check_mtime = true`.

There's a singleton `GLOBAL_CACHE` and all accesses are protected by a single `GLOBAL_LOCK`, so while this is
thread-safe, it does not support parallelism. It is meant for "expensive" OS operations such as `mmap`, not for
memoizing functions results for performance.

The cache holds weak references to the objects, so they are let go if there remain no active references to them (outside
the cache).

!!! note

    Use absolute paths to ensure correct results.

```jldoctest
first = get_through_global_weak_cache("fake path", :test) do _
    return [1]
end
@assert first == [1]
@assert first === get_through_global_weak_cache("fake path", :test) do _
    @assert false;
end

second = get_through_global_weak_cache("fake path", :test; purge = true) do _
    return [2]
end
@assert second == [2];

first = second = nothing
gc()
third = get_through_global_weak_cache("fake path", :test) do _
    return [3]
end
@assert third == [3];

# output

"""
function get_through_global_weak_cache(
    getter::Function,
    path::AbstractString,
    key::Any;
    purge::Bool = false,
    check_mtime::Bool = false,
)::Any
    return lock(GLOBAL_LOCK) do
        file_stat = stat(path)
        current_mtime = file_stat.mtime
        current_inode = file_stat.inode
        cached_mtime = nothing
        cached_inode = nothing
        cached_weak_ref = nothing

        if !purge
            cached = get(GLOBAL_CACHE, (path, key), nothing)
            if cached !== nothing
                cached_mtime, cached_inode, cached_weak_ref = cached
            end
        end

        is_invalid =
            cached_weak_ref === nothing ||
            cached_weak_ref.value === nothing ||
            current_inode != cached_inode ||
            (check_mtime && current_mtime != cached_mtime)

        if is_invalid
            @debug "Get $(key) for $(path)"
            value = getter(key)
        else
            @debug "Reuse $(key) for $(path)"
            value = cached_weak_ref.value
        end
        GLOBAL_CACHE[(path, key)] = (current_mtime, current_inode, WeakRef(value))

        return value
    end
end

end
