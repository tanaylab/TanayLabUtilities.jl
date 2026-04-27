"""
(Very partially) mitigate old Linux kernels memory mapping issues.

Linux kernels prior to 5.7 serialize page faults from multiple threads of the same process. Which means that if we
memory map a file, and different threads access different pages, triggering page faults, all the IO related code becomes
serial, which butchers performance.

A very partial and unsatisfactory workaround is to have the file in a RAM disk, and also pre-populate the page table
when mapping the file. There's no I/O and serially pre-populating the page table is faster than doing this in the middle
of the parallel code.

Of course this requires enough RAM so that having the files in a RAM disk (e.g. `/dev/shm`) is practical, and also
requires manually copying the files back from/to a real disk if they need to be persistent, so this workaround only
works when the stars are in alignment.

However, if you are stuck with an ancient Linux kernel, this can dramatically increase performance of multi-threaded
code that uses memory mapping (specifically, `Metacells` code). Given 5.7 was released in 2020 and will be end-of-life
in 2029, and large institute IT departments sometimes use ancient kernel versions until the last possible moment, we
should probably be able to delete this module in 2030. Sigh.
"""
module OldLinuxMmap

export mmap_populate_if_old_linux_ramdisk
export mmap_with_small_pages

import Mmap

IS_OLD_LINUX_KERNEL = false

function __init__()::Nothing
    global IS_OLD_LINUX_KERNEL
    @static if Sys.islinux()  # FLAKY TESTED
        version_string = read("/proc/version", String)  # FLAKY TESTED
        parts = split(version_string, " ")  # FLAKY TESTED
        parts = tuple(parse.(Int, split(parts[3], ".")[1:2])...)  # FLAKY TESTED
        IS_OLD_LINUX_KERNEL = parts <= (5, 7)  # FLAKY TESTED
    else
        IS_OLD_LINUX_KERNEL = false  # FLAKY TESTED
    end

    if IS_OLD_LINUX_KERNEL
        @info "Old linux kernel, will pre-populate mmap into RAM disk: $(version_string)"  # FLAKY TESTED # NOJET
    else
        @debug "New linux kernel, will not pre-populate mmap into RAM disk: $(version_string)" _group = :tlu_env  # FLAKY TESTED # NOJET
    end

    return nothing
end

function is_in_linux_ramdisk(directory::AbstractString)::Bool
    @static if Sys.islinux()
        directory = string(directory)
        buffer = zeros(UInt8, 512)  # For struct statfs
        status = ccall(:statfs, Cint, (Cstring, Ptr{UInt8}), directory, buffer)
        if status != 0
            return false
        end
        f_type = reinterpret(UInt64, buffer[1:8])[1]
        TMPFS_MAGIC = 0x01021994
        RAMFS_MAGIC = 0x858458f6
        return f_type in (TMPFS_MAGIC, RAMFS_MAGIC)
    else
        return false
    end
end

function disable_thp(ptr::Ptr{Cvoid}, size::Integer)::Nothing  # FLAKY TESTED
    if size > 0
        MADV_NOHUGEPAGE = 15
        ret = ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), ptr, Csize_t(size), Cint(MADV_NOHUGEPAGE))
        if ret != 0
            error("madvise size: $(size) failed: $(Base.Libc.strerror())")
        end
    end
    return nothing
end

"""
    mmap_with_small_pages(
        io::IO,
        ::Type{ArrayT},
        dims::Union{Integer, Tuple{Vararg{Integer}}},
        offset::Integer;
        grow::Bool = true,
        shared::Bool = true,
    )::ArrayT where {T, ArrayT <: Array{T}}

Call `Mmap.mmap` with the given arguments and, on Linux, immediately disable Transparent Huge Pages
on the returned region via `madvise(MADV_NOHUGEPAGE)`. Use this in place of `Mmap.mmap` when you
rely on demand paging to load only the specific bytes an access actually touches: without disabling
THP, Linux can promote consecutive 4 KiB pages into 2 MiB huge pages, turning a single-byte access
into a 2 MiB fault and defeating the lazy-load benefit of memory mapping a large file.
"""
function mmap_with_small_pages(  # FLAKY TESTED
    io::IO,
    ::Type{ArrayT},
    dims::Union{Integer, Tuple{Vararg{Integer}}},
    offset::Integer;
    grow::Bool = true,
    shared::Bool = true,
)::ArrayT where {T, ArrayT <: Array{T}}
    array = Mmap.mmap(io, ArrayT, dims, offset; grow, shared)
    @static if Sys.islinux()
        disable_thp(Ptr{Cvoid}(pointer(array)), length(array) * sizeof(T))
    end
    return array
end

"""
    mmap_populate_if_old_linux_ramdisk(
        path::AbstractString,
        ::Type{ArrayT},
        size::Union{Integer, Tuple{<:Integer, <:Integer}},
        mode::AbstractString
    )::ArrayT where {ArrayT <: Array}

Similar to `mmap` but pre-populate the page table if in Linux, the kernel is older than 5.7, and the `file` is in a RAM
disk. The `mode` must be either `r` or `r+`. Mapping is always done with `grow = false`.
"""
function mmap_populate_if_old_linux_ramdisk(  # UNTESTED
    path::AbstractString,
    ::Type{ArrayT},
    size::Union{Integer, Tuple{<:Integer, <:Integer}},
    mode::AbstractString,
)::ArrayT where {T, ArrayT <: Array{T}}
    @assert mode in ("r", "r+")

    bytes_size = 0

    @static if Sys.islinux()
        if size isa Tuple
            total_size = size[1] * size[2]
        else
            total_size = size
        end
        bytes_size = total_size * sizeof(T)

        if bytes_size > 0 && IS_OLD_LINUX_KERNEL && is_in_linux_ramdisk(dirname(path))
            return open(path, mode) do file  # NOJET
                MAP_POPULATE = Cint(0x08000)
                MAP_SHARED = Cint(0x01)
                PROT_READ = Cint(0x01)
                PROT_WRITE = Cint(0x02)
                prot = mode == "r+" ? PROT_READ | PROT_WRITE : PROT_READ
                flags = MAP_SHARED | MAP_POPULATE
                ptr = ccall(  # NOJET
                    :mmap,
                    Ptr{Cvoid},
                    (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Csize_t),
                    Ptr{Cvoid}(0),
                    Csize_t(bytes_size),
                    Cint(prot),
                    Cint(flags),
                    Cint(reinterpret(Int32, fd(file))),
                    Csize_t(0),
                )
                if ptr == Ptr{Cvoid}(-1)
                    error("mmap path: $(path) mode: $(mode) size: $(bytes_size) failed: $(Base.Libc.strerror())")
                end
                array = unsafe_wrap(Array, Ptr{T}(ptr), size; own = false)
                let captured_ptr = ptr, captured_size = bytes_size
                    finalizer(array) do _
                        return ccall(
                            :munmap,
                            Cint,
                            (Ptr{Cvoid}, Csize_t),
                            Ptr{Cvoid}(captured_ptr),
                            Csize_t(captured_size),
                        )
                    end
                end
                disable_thp(ptr, bytes_size)
                step = div(4096, sizeof(T))
                sum = 0.0
                for index in 1:step:total_size
                    sum += @inbounds array[index]
                end
                @assert !isnan(sum)
                return array
            end
        end
    end

    return open(path, mode) do file  # NOJET
        return mmap_with_small_pages(file, ArrayT, size, 0; grow = false)  # NOJET
    end
end

end  # module
