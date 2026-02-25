"""
Wrap `ispath` with a cache such that repeated calls do not go to the OS. This greatly speeds up
operations when working with NFS data.

!!! note

    The code here is thread-safe.
"""
module CachedIsPath

export cached_ispath
export empty_ispath_cache!

using Base.Threads
using ConcurrentUtils

using ..Brief
using ..FlameTime
using ..Types

"""
How long to hold on to `ispath` results before going back to the OS and asking for an updated result.
This can be controlled by setting the `TLU_IS_PATH_CACHE_TIMEOUT_NS` environment variable.

By default, this is set to a negative value, meaning we cache everything forever. This is fastest and works as long as
the code using [`cached_ispath`](@ref) also invokes [`empty_ispath_cache!`](@ref), for example the `Daf` file system
`FilesDaf` format does this. If the code is designed to deal with external modifications to the file system, it should
contain additional calls to [`cached_ispath`](@ref) or avoid using [`cached_ispath`](@ref) in the first place.

If this is zero then [`cached_ispath`](@ref) ignores the cache and simply calls `ispath`. This is slow (especially when
accessing network disks) but is "safest" as we will always return the correct answer regardless of anything. This is
provided as a failsafe method for disabling the mechanism.

Otherwise this can be set to the number of nanoseconds to hold cache entries for. A low value (a few seconds) will
significantly improve performance while still allowing multiple processes to (eventually) react to external changes
to the file system. This is provided for completeness.
"""
IS_PATH_CACHE_TIMEOUT_NS = -1

IS_PATH_CACHE_LOCK = ReadWriteLock()

struct IsPathCacheEntry
    at_ns::UInt64
    file_names::AbstractSet{<:AbstractString}
end

IS_PATH_CACHE_DICT = Dict{Tuple{AbstractString, AbstractString}, IsPathCacheEntry}()

function __init__()::Nothing
    timeout = get(ENV, "TLU_IS_PATH_CACHE_TIMEOUT_NS", nothing)
    if timeout !== nothing
        global IS_PATH_CACHE_TIMEOUT_NS  # UNTESTED
        IS_PATH_CACHE_TIMEOUT_NS = parse(Int64, timeout)  # UNTESTED
    end
    if IS_PATH_CACHE_TIMEOUT_NS > 0
        @info "Will cache ispath data for $(delimited_number(IS_PATH_CACHE_TIMEOUT_NS)) nanoseconds" _group = :tlu_env  # UNTESTED
    elseif IS_PATH_CACHE_TIMEOUT_NS < 0
        @info "Will cache ispath data forever" _group = :tlu_env
    end
    return nothing
end

"""
    cached_ispath(path::AbstractString)::Bool

Same as `ispath`, except that if less than [`IS_PATH_CACHE_TIMEOUT_NS`](@ref) has passed since the last time
`cached_ispath` was called for the same path, we reuse the result. If [`IS_PATH_CACHE_TIMEOUT_NS`](@ref) is zero, this
ignores the cache and simply calls `ispath`. If [`IS_PATH_CACHE_TIMEOUT_NS`](@ref) is negative, we always return the
cached result.

!!! note

    Do not pass this a path that ends with a `/`, or the empty string. We actually cache the list of files in the
    directory, since this gives much faster results when checking for multiple files in the same directory. However this
    requires being more careful with [`empty_ispath_cache!`](@ref).
"""
function cached_ispath(path::AbstractString)::Bool  # UNTESTED
    if IS_PATH_CACHE_TIMEOUT_NS == 0
        return ispath(path)
    end

    return flame_timed("cached_ispath") do
        file_name = basename(path)
        @assert file_name != ""

        dir_name = dirname(path)
        if dir_name == ""
            dir_name = "."
        end

        if isabspath(path)
            cwd = "/"
        else
            cwd = pwd()
        end

        for is_write in (false, true)
            result = flame_timed(is_write ? "cached_ispath.write" : "cached_ispath.read") do
                (is_write ? lock : lock_read)(IS_PATH_CACHE_LOCK) do
                    cached = get(IS_PATH_CACHE_DICT, (cwd, dir_name), nothing)

                    now_ns = time_ns()
                    if cached !== nothing &&
                       (IS_PATH_CACHE_TIMEOUT_NS < 0 || now_ns - cached.at_ns <= IS_PATH_CACHE_TIMEOUT_NS)
                        return file_name in cached.file_names
                    end

                    if is_write
                        cached = IsPathCacheEntry(now_ns, Set(readdir(dir_name)))
                        IS_PATH_CACHE_DICT[(cwd, dir_name)] = cached
                        return file_name in cached.file_names
                    end

                    return nothing
                end
            end

            if result !== nothing
                return result
            end
        end

        @assert false
    end
end

"""
    empty_ispath_cache!(path::Maybe{AbstractString} = nothing)::Nothing

Empty the `cached_ispath` cache. If a path is specified, only clears the cached entry for this path (and every other
file in the same directory).
"""
function empty_ispath_cache!(path::Maybe{AbstractString} = nothing)::Nothing  # UNTESTED
    lock(IS_PATH_CACHE_LOCK.lock) do
        if path === nothing
            empty!(IS_PATH_CACHE_DICT)
        else
            if isabspath(path)
                cwd = "/"
            else
                cwd = pwd()
            end

            dir_name = dirname(path)
            if dir_name == ""
                dir_name = "."
            end

            delete!(IS_PATH_CACHE_DICT, (cwd, dir_name))
        end
    end
    return nothing
end

end
