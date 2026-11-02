"""
Memory-map files with small pages, and keep track of the mappings.

A page fault reads at least its whole page. Linux may back a mapping with 2 MB transparent huge pages, so an access of a
few bytes would read 2 MB. This defeats the point of memory-mapping a large file, which is to read only the bytes that
are used. So every mapping made here turns off transparent huge pages.

Every mapping is also recorded in a process-wide registry, which maps its addresses back to its file and offset (see
[`mapped_file_position`](@ref)). The entry is removed when the mapped array is garbage collected.
"""
module MemoryMappings

export mapped_file_position
export mmap_with_small_pages
export remap_with_small_pages!

using DataStructures

using ..Types

import Mmap

# Platform-specific MAP_ANON flag value for `mmap(2)` (Darwin uses 0x1000; Linux and other BSDs use 0x20).
@static if Sys.isapple()
    const MAP_ANON_FLAG = Cint(0x1000)
else
    const MAP_ANON_FLAG = Cint(0x20)
end
const MAP_PRIVATE_FLAG = Cint(0x02)
const MAP_SHARED_FLAG = Cint(0x01)
const MAP_FIXED_FLAG = Cint(0x10)
const PROT_NONE_FLAG = Cint(0x00)
const PROT_READ_FLAG = Cint(0x01)
const PROT_WRITE_FLAG = Cint(0x02)
const MADV_NOHUGEPAGE = Cint(15)
const F_GETFL = Cint(3)
const O_ACCMODE = Cint(3)
const O_RDONLY = Cint(0)

# A range of mapped addresses, starting at its key in `MAPPINGS`. It is mutable so that each mapping has its own
# identity. A finalizer removes only its own entry, never a later mapping at the same address.
mutable struct MappedRange
    n_bytes::UInt
    path::String
    file_offset::UInt64
end

const MAPPINGS = SortedDict{UInt, MappedRange}()

# A spin lock disables finalizers while it is held, so a finalizer can't run inside the critical section on that thread.
# It never switches tasks, so a finalizer may take it.
const MAPPINGS_LOCK = Threads.SpinLock()

"""
    mmap_with_small_pages(
        path::AbstractString,
        ::Type{ArrayT},
        dims::Maybe{Union{Integer, Tuple{Vararg{Integer}}}} = nothing,
        offset::Integer = 0;
        mode::AbstractString = "r",
        reserved_bytes::Maybe{Integer} = nothing,
    )::ArrayT where {T, ArrayT <: Array{T}}

    mmap_with_small_pages(
        io::IOStream,
        ::Type{ArrayT},
        dims::Maybe{Union{Integer, Tuple{Vararg{Integer}}}} = nothing,
        offset::Integer = 0;
        reserved_bytes::Maybe{Integer} = nothing,
    )::ArrayT where {T, ArrayT <: Array{T}}

Memory-map an array of `dims` elements from a file, starting `offset` bytes into it. If `dims` is `nothing`, the array
is a vector of the rest of the file. The file is given by its `path`, and opened in the `mode` (`"r"` or `"r+"`), or as
an open `io`. The `io` form avoids opening the file again, when mapping several parts of an open file. The mapping never
grows the file, and turns off transparent huge pages. It is recorded for [`mapped_file_position`](@ref) until the array
is garbage collected.

If `reserved_bytes` is given, the array is a `Vector{UInt8}` of `reserved_bytes` of address space. The first `dims`
bytes of the file are mapped at its start, and the rest can't be accessed until [`remap_with_small_pages!`](@ref) maps
more of the file there. The `offset` must be zero.
"""
function mmap_with_small_pages(
    path::AbstractString,
    ::Type{ArrayT},
    dims::Maybe{Union{Integer, Tuple{Vararg{Integer}}}} = nothing,
    offset::Integer = 0;
    mode::AbstractString = "r",
    reserved_bytes::Maybe{Integer} = nothing,
)::ArrayT where {T, ArrayT <: Array{T}}
    @assert mode in ("r", "r+")
    return open(path, mode) do io
        return mmap_file(io, String(path), ArrayT, dims, offset, reserved_bytes)
    end
end

function mmap_with_small_pages(
    io::IOStream,
    ::Type{ArrayT},
    dims::Maybe{Union{Integer, Tuple{Vararg{Integer}}}} = nothing,
    offset::Integer = 0;
    reserved_bytes::Maybe{Integer} = nothing,
)::ArrayT where {T, ArrayT <: Array{T}}
    return mmap_file(io, io_path(io), ArrayT, dims, offset, reserved_bytes)
end

function io_path(io::IOStream)::String
    @static if Sys.islinux()
        return readlink("/proc/self/fd/$(file_descriptor(io))")
    else
        return chop(io.name; head = 6, tail = 1)  # `open` names the stream `<file path>`. # UNTESTED
    end
end

function file_descriptor(io::IOStream)::Cint
    return Base.cconvert(Cint, fd(io))
end

# The protection that the access mode of the descriptor allows. A stream made by `fdio` is `iswritable` even when its
# descriptor is read-only, so this asks the descriptor.
function mapping_protection(io::IOStream)::Cint
    access_mode = ccall(:fcntl, Cint, (Cint, Cint, Cint...), file_descriptor(io), F_GETFL)
    if access_mode == -1
        error("fcntl F_GETFL failed: $(Base.Libc.strerror())")  # UNTESTED
    end
    return access_mode & O_ACCMODE == O_RDONLY ? PROT_READ_FLAG : PROT_READ_FLAG | PROT_WRITE_FLAG
end

function mmap_file(
    io::IOStream,
    path::String,
    ::Type{ArrayT},
    maybe_dims::Maybe{Union{Integer, Tuple{Vararg{Integer}}}},
    offset::Integer,
    reserved_bytes::Maybe{Integer},
)::ArrayT where {T, ArrayT <: Array{T}}
    if maybe_dims === nothing
        @assert ArrayT <: Vector
        dims = div(filesize(io) - offset, sizeof(T))
    else
        dims = maybe_dims
    end
    n_bytes = prod(dims) * sizeof(T)
    if offset + n_bytes > filesize(io)
        error("mapping $(n_bytes) bytes at offset $(offset) is past the end of the file: $(path)")
    end
    protection = mapping_protection(io)

    if reserved_bytes !== nothing
        @assert ArrayT == Vector{UInt8}
        @assert offset == 0
        @assert n_bytes <= reserved_bytes
        base = reserve_addresses(reserved_bytes)
        if n_bytes > 0
            map_file_at(base, io, n_bytes, protection)
        end
        disable_huge_pages(base, reserved_bytes)
        return track_mapping(Vector{UInt8}, base, reserved_bytes, reserved_bytes, path, UInt64(0))
    end

    if n_bytes == 0
        return ArrayT(undef, dims)
    end
    page_offset = offset % Mmap.PAGESIZE  # NOLINT
    mapped_bytes = n_bytes + page_offset
    base = ccall(
        :mmap,
        Ptr{Cvoid},
        (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Int64),
        Ptr{Cvoid}(0),
        Csize_t(mapped_bytes),
        protection,
        MAP_SHARED_FLAG,
        file_descriptor(io),
        Int64(offset - page_offset),
    )
    if reinterpret(Int, base) == -1
        error("mmap of $(n_bytes) bytes at offset $(offset) failed: $(Base.Libc.strerror()): $(path)")  # UNTESTED
    end
    disable_huge_pages(base, mapped_bytes)
    return track_mapping(ArrayT, base, mapped_bytes, dims, path, UInt64(offset - page_offset))
end

# Reserve `n_bytes` of address space as an anonymous `PROT_NONE` mapping. It takes no RAM and no file space. On failure
# (typically because reservations which are no longer used, but not yet collected, fill the address space), collect them
# and try once more.
function reserve_addresses(n_bytes::Integer)::Ptr{Cvoid}
    base = try_reserve_addresses(n_bytes)
    if reinterpret(Int, base) == -1
        GC.gc()  # UNTESTED
        base = try_reserve_addresses(n_bytes)  # UNTESTED
        if reinterpret(Int, base) == -1  # UNTESTED
            error("mmap reservation of $(n_bytes) bytes failed: $(Base.Libc.strerror())")  # UNTESTED
        end
    end
    return base
end

function try_reserve_addresses(n_bytes::Integer)::Ptr{Cvoid}
    return ccall(
        :mmap,
        Ptr{Cvoid},
        (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Int64),
        Ptr{Cvoid}(0),
        Csize_t(n_bytes),
        PROT_NONE_FLAG,
        MAP_PRIVATE_FLAG | MAP_ANON_FLAG,
        Cint(-1),
        Int64(0),
    )
end

# Map the first `n_bytes` of the file at `base`, replacing whatever was mapped there.
function map_file_at(base::Ptr{Cvoid}, io::IOStream, n_bytes::Integer, protection::Cint)::Nothing
    mapped = ccall(
        :mmap,
        Ptr{Cvoid},
        (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Int64),
        base,
        Csize_t(n_bytes),
        protection,
        MAP_SHARED_FLAG | MAP_FIXED_FLAG,
        file_descriptor(io),
        Int64(0),
    )
    if mapped != base
        error("mmap of $(n_bytes) bytes at $(base) failed: $(Base.Libc.strerror())")  # UNTESTED
    end
    return nothing
end

function disable_huge_pages(base::Ptr{Cvoid}, n_bytes::Integer)::Nothing
    @static if Sys.islinux()
        result = ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), base, Csize_t(n_bytes), MADV_NOHUGEPAGE)
        if result != 0
            error("madvise of $(n_bytes) bytes at $(base) failed: $(Base.Libc.strerror())")  # UNTESTED
        end
    end
    return nothing
end

# Wrap the `mapped_bytes` at `base` as an array, which starts at the end of the page offset, and record the mapping. The
# array's finalizer removes the entry, then unmaps, so no other mapping can reuse the addresses while they are recorded.
function track_mapping(
    ::Type{ArrayT},
    base::Ptr{Cvoid},
    mapped_bytes::Integer,
    dims::Union{Integer, Tuple{Vararg{Integer}}},
    path::String,
    base_file_offset::UInt64,
)::ArrayT where {T, ArrayT <: Array{T}}
    page_offset = mapped_bytes - prod(dims) * sizeof(T)
    array = unsafe_wrap(ArrayT, Ptr{T}(base + page_offset), dims; own = false)
    start = UInt(base)
    mapped_range = MappedRange(UInt(mapped_bytes), path, base_file_offset)
    lock(MAPPINGS_LOCK) do
        MAPPINGS[start] = mapped_range
        return nothing
    end
    finalizer(array) do _
        lock(MAPPINGS_LOCK) do
            if get(MAPPINGS, start, nothing) === mapped_range
                delete!(MAPPINGS, start)
            end
            return nothing
        end
        ccall(:munmap, Cint, (Ptr{Cvoid}, Csize_t), Ptr{Cvoid}(start), Csize_t(mapped_bytes))
        return nothing
    end
    return array
end

"""
    remap_with_small_pages!(array::Vector{UInt8}, io::IOStream, n_bytes::Integer)::Nothing

Map the first `n_bytes` of the file `io` at the start of the `array`, which [`mmap_with_small_pages`](@ref) returned
with `reserved_bytes`. This makes the array access more of a grown file, or less of a shrunk one. The address of the
array doesn't change.
"""
function remap_with_small_pages!(array::Vector{UInt8}, io::IOStream, n_bytes::Integer)::Nothing
    @assert n_bytes <= length(array)
    if n_bytes > 0
        map_file_at(Ptr{Cvoid}(pointer(array)), io, n_bytes, mapping_protection(io))
    end
    return nothing
end

"""
    mapped_file_position(address::Union{Ptr, Integer})::Maybe{Tuple{String, UInt64}}

Return the path of the file mapped at the `address`, and the offset in the file of the byte there. Returns `nothing` if
no mapping made by [`mmap_with_small_pages`](@ref) contains the `address`.
"""
function mapped_file_position(address::Union{Ptr, Integer})::Maybe{Tuple{String, UInt64}}
    address = UInt(address)
    return lock(MAPPINGS_LOCK) do
        token = searchsortedlast(MAPPINGS, address)
        if token == beforestartsemitoken(MAPPINGS)
            return nothing
        end
        start, mapped_range = deref((MAPPINGS, token))
        if address >= start + mapped_range.n_bytes
            return nothing
        end
        return (mapped_range.path, mapped_range.file_offset + UInt64(address - start))
    end
end

end  # module
