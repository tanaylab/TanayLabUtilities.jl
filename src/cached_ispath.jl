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

struct DirectoryRedirect
    real_directory::AbstractString
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

IS_PATH_CACHE_DICT = SortedDict{AbstractString, Union{DirectoryFiles, DirectoryRedirect}}()

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

function normalize_path(path::AbstractString)::AbstractString
    if path == "/"
        return path
    else
        return abspath(rstrip(path, '/'))
    end
end

function resolve_realpath(path::AbstractString)::AbstractString
    current = path
    while true
        try
            resolved = realpath(current)
            if current === path
                return resolved
            end
            return resolved * SubString(path, ncodeunits(current) + 1)  # UNTESTED
        catch
            current = dirname(current)  # UNTESTED
        end
    end
end

function get_if_cached_and_fresh(path::AbstractString)::Maybe{DirectoryFiles}
    @assert IS_PATH_CACHE_TIMEOUT != 0
    entry = get(IS_PATH_CACHE_DICT, path, nothing)
    while entry isa DirectoryRedirect
        if IS_PATH_CACHE_TIMEOUT > 0 && (now() - entry.fetched_at).value / 1000.0 >= IS_PATH_CACHE_TIMEOUT
            return nothing
        end
        entry = get(IS_PATH_CACHE_DICT, entry.real_directory, nothing)
    end
    if entry isa DirectoryFiles &&
       (IS_PATH_CACHE_TIMEOUT < 0 || (now() - entry.fetched_at).value / 1000.0 < IS_PATH_CACHE_TIMEOUT)
        return entry
    else
        return nothing
    end
end

function collect_subtree!(to_delete::Vector{AbstractString}, path::AbstractString)::Nothing
    prefix = path * "/"
    token = searchsortedfirst(IS_PATH_CACHE_DICT, prefix)
    while token !== pastendsemitoken(IS_PATH_CACHE_DICT)
        key = deref_key((IS_PATH_CACHE_DICT, token))  # UNTESTED
        if !startswith(key, prefix)  # UNTESTED
            break  # UNTESTED
        end
        push!(to_delete, key)  # UNTESTED
        token = advance((IS_PATH_CACHE_DICT, token))  # UNTESTED
    end
    return nothing
end

"""
    report_modified!(path::AbstractString)::Nothing

Report that the specified file or directory was modified in some way, so delete all relevant cached data. This includes
the list of files in the containing directory; if a directory is modified, all the cached data data for all its
sub-directories is deleted as well.
"""
function report_modified!(path::AbstractString)::Nothing
    path = normalize_path(path)
    directory = dirname(path)
    to_delete = AbstractString[directory, path]

    @debug "report_modified! $(path)"

    lock(IS_PATH_CACHE_LOCK) do
        index = 0
        while index < length(to_delete)
            index += 1
            key = to_delete[index]
            collect_subtree!(to_delete, key)
            entry = get(IS_PATH_CACHE_DICT, key, nothing)
            if entry isa DirectoryRedirect
                push!(to_delete, entry.real_directory)  # UNTESTED
                collect_subtree!(to_delete, entry.real_directory)  # UNTESTED
            end
        end

        for key in to_delete
            delete!(IS_PATH_CACHE_DICT, key)
            @debug "- delete! $(key)"
        end
    end

    return nothing
end

function cached_readdir(directory::AbstractString)::Set{<:String}
    @assert IS_PATH_CACHE_TIMEOUT != 0

    return flame_timed("cached_readdir") do
        readdir_target = nothing
        cached = lock_read(IS_PATH_CACHE_LOCK) do
            readdir_target = directory
            entry = get(IS_PATH_CACHE_DICT, readdir_target, nothing)
            while entry isa DirectoryRedirect
                if IS_PATH_CACHE_TIMEOUT > 0 && (now() - entry.fetched_at).value / 1000.0 >= IS_PATH_CACHE_TIMEOUT  # UNTESTED
                    return nothing  # UNTESTED
                end
                readdir_target = entry.real_directory  # UNTESTED
                entry = get(IS_PATH_CACHE_DICT, readdir_target, nothing)  # UNTESTED
            end
            if entry isa DirectoryFiles
                if IS_PATH_CACHE_TIMEOUT < 0 || (now() - entry.fetched_at).value / 1000.0 < IS_PATH_CACHE_TIMEOUT  # UNTESTED
                    return entry  # UNTESTED
                end
            end
            return nothing
        end

        if cached !== nothing
            return cached.file_names  # UNTESTED
        end

        fetched_at = now()
        real_directory = resolve_realpath(readdir_target)
        file_names = Set(readdir(real_directory))

        lock(IS_PATH_CACHE_LOCK) do
            if real_directory != directory
                IS_PATH_CACHE_DICT[directory] = DirectoryRedirect(real_directory, fetched_at)
                @debug "cached_readdir redirect $(directory) -> $(real_directory)"

                if readdir_target != directory
                    IS_PATH_CACHE_DICT[readdir_target] = DirectoryRedirect(real_directory, fetched_at)  # UNTESTED
                    @debug "cached_readdir redirect $(readdir_target) -> $(real_directory)"  # UNTESTED
                end
            end

            IS_PATH_CACHE_DICT[readdir_target] = cached = DirectoryFiles(file_names, fetched_at)
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
function cached_ispath(path::AbstractString)::Bool
    if IS_PATH_CACHE_TIMEOUT == 0
        return ispath(path)  # UNTESTED
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
