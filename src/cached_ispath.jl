"""
Speed up `stat` calls, avoiding going to the OS. This greatly speeds up operations when working with NFS data. It
assumes that all relevant filesystem modification operations are done through the API here. External modifications can
be supported by specifying a timeout, so that the actual filesystem will be queried eventually for fresh results.
"""
module CachedIsPath

using ConcurrentUtils
using Dates
using DataStructures

export cached_ispath
export report_modified!

using ..FlameTime
using ..MostlyReadWriteLocks
using ..Types

struct DirectoryFiles
    file_names::Set{String}
    fetched_at::DateTime
end

"""
How long to hold on to `ispath` results before going back to the OS and asking for an updated result.
This can be controlled by setting the `TLU_IS_PATH_CACHE_TIMEOUT` environment variable.

By default, this is set to a negative value, meaning we cache everything forever. This is fastest and works as long as
the code using [`cached_ispath`](@ref) also invokes [`report_modified!](@ref) (for example, the `Daf` file system
`FilesDaf` format does this). If the code is designed to deal with external modifications to the file system, it should
contain additional calls to [`report_modified!](@ref) to force refreshing the cache.

If this is zero then [`cached_ispath`](@ref) ignores the cache and simply calls `ispath`. This is slow (especially when
accessing network disks) but is "safest" as we will always return the correct answer regardless of anything. This is
provided as a failsafe method for disabling the mechanism.

Otherwise this can be set to the number of seconds to hold cache entries for. A low value (a few seconds) will
significantly improve performance while still allowing multiple processes to (eventually) react to external changes to
the file system. This is provided for completeness.
"""
IS_PATH_CACHE_TIMEOUT = -1

IS_PATH_CACHE_LOCK = MostlyReadWriteLock()

IS_PATH_CACHE_DICT = SortedDict{AbstractString, DirectoryFiles}()

function __init__()::Nothing
    timeout = get(ENV, "TLU_IS_PATH_CACHE_TIMEOUT", nothing)
    if timeout !== nothing
        global IS_PATH_CACHE_TIMEOUT  # UNTESTED
        IS_PATH_CACHE_TIMEOUT = parse(Float64, timeout)  # UNTESTED
    end
    if IS_PATH_CACHE_TIMEOUT > 0
        @info "Will cache ispath data for $(IS_PATH_CACHE_TIMEOUT) seconds"  # UNTESTED
    elseif IS_PATH_CACHE_TIMEOUT < 0
        @info "Will cache ispath data forever"
    end
    return nothing
end

function normalize_path(path::AbstractString)::AbstractString  # UNTESTED
    if path == "/"
        return path
    else
        return abspath(rstrip(path, '/'))
    end
end

function get_if_cached_and_fresh(path::String)::Maybe{DirectoryFiles}  # UNTESTED
    @assert IS_PATH_CACHE_TIMEOUT != 0
    directory_files = get(IS_PATH_CACHE_DICT, path, nothing)
    if directory_files !== nothing &&
       (IS_PATH_CACHE_TIMEOUT < 0 || (now() - directory_files.fetched_at).value / 1000.0 < IS_PATH_CACHE_TIMEOUT)
        return directory_files
    else
        return nothing
    end
end

"""
    report_modified!(path::AbstractString)::Nothing

Report that the specified file or directory was modified in some way, so delete all relevant cached data. This includes
the list of files in the containing directory; if a directory is modified, all the cached data data for all its
sub-directories is deleted as well.
"""
function report_modified!(path::AbstractString)::Nothing  # UNTESTED
    path = normalize_path(path)
    prefix = path * "/"
    directory = dirname(path)
    to_delete = AbstractString[directory]

    @debug "report_modified! $(path)"

    lock(IS_PATH_CACHE_LOCK) do
        token = searchsortedfirst(IS_PATH_CACHE_DICT, path)
        while token !== pastendsemitoken(IS_PATH_CACHE_DICT)
            key = deref_key((IS_PATH_CACHE_DICT, token))
            if key != path && !startswith(key, prefix)
                break
            end
            push!(to_delete, key)
            token = advance((IS_PATH_CACHE_DICT, token))
        end

        for key in to_delete
            delete!(IS_PATH_CACHE_DICT, key)
            @debug "- delete! $(key)"
        end
    end

    return nothing
end

function cached_readdir(directory::AbstractString)::Set{<:String}  # UNTESTED
    @assert IS_PATH_CACHE_TIMEOUT != 0

    return flame_timed("cached_readdir") do
        cached = lock_read(IS_PATH_CACHE_LOCK) do
            return get_if_cached_and_fresh(directory)
        end

        if cached !== nothing
            return cached.file_names
        end

        fetched_at = now()
        file_names = Set(readdir(directory))

        return lock(IS_PATH_CACHE_LOCK) do
            cached = get_if_cached_and_fresh(directory)
            if cached === nothing
                IS_PATH_CACHE_DICT[directory] = cached = DirectoryFiles(file_names, fetched_at)
                @debug "cached_readdir $(directory)"
            end
            return cached.file_names
        end
    end
end

"""
    cached_ispath(path::AbstractString)::Bool

Same as `ispath`, except that if less than [`IS_PATH_CACHE_TIMEOUT`](@ref) has passed since the last time
`cached_ispath` was called for the same path, we reuse the result. If [`IS_PATH_CACHE_TIMEOUT`](@ref) is zero, this
ignores the cache and simply calls `ispath`. If [`IS_PATH_CACHE_TIMEOUT`](@ref) is negative, we always return the cached
result.
"""
function cached_ispath(path::AbstractString)::Bool  # UNTESTED
    if IS_PATH_CACHE_TIMEOUT == 0
        return ispath(path)
    else
        path = normalize_path(path)
        directory = dirname(path)
        file_name = basename(path)
        directory_files = cached_readdir(directory)
        result = file_name in directory_files
        @debug "cached_ispath $(result) $(path)"
        return result
    end
end

end # module
