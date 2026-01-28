"""
Enhanced (reentrant) read-write locks.

These add functionality on top of "any" simple read-write locks, such as `ConcurrentUtils.ReadWriteLock` or
[`MostlyReadWriteLock`](@ref). Specifically, they allow querying the status of the lock.

We also provide aliases `lock_write` and `unlock_write` to `lock` and `unlock` for clearer code. These will work for
"any" read-write lock, basic or extended.

The locking and unlocking functions are extended to generate `@debug` log messages for the operation(s). If a named
`what` parameter is given to the function(s), it (or its return value if it is a function) is included in the message.
Enabling debugging for `ExtendedReadWriteLocks` will therefore log each and every operation, which is a blunt tool, but
useful in desperate cases to debug multi-threaded locking behavior.

!!! note

    There's no standard abstract type for "any" read-write lock. Defining a new one doesn't help since
    `ConcurrentUtils.ReadWriteLock` won't use it. Sigh.

```jldoctest
using Base.Threads
using ConcurrentUtils

read_write_lock = ExtendedReadWriteLock()

@assert !has_write_lock(read_write_lock)
@assert !has_read_lock(read_write_lock)

lock_write(read_write_lock; what = "top_write") do
    @assert has_write_lock(read_write_lock)
    @assert has_read_lock(read_write_lock)

    lock_write(read_write_lock; what = "nested_write") do
        @assert has_write_lock(read_write_lock)
        @assert has_read_lock(read_write_lock)
    end

    lock_read(read_write_lock; what = "nested_read") do
        @assert has_write_lock(read_write_lock)
        @assert has_read_lock(read_write_lock)
    end
end

@assert !has_write_lock(read_write_lock)
@assert !has_read_lock(read_write_lock)

# output

```

```jldoctest
using Base.Threads
using ConcurrentUtils

read_write_lock = ExtendedReadWriteLock()

@assert !has_write_lock(read_write_lock)
@assert !has_read_lock(read_write_lock)

lock_read(read_write_lock; what = () -> "top_read") do
    @assert !has_write_lock(read_write_lock)
    @assert has_read_lock(read_write_lock)

    lock_read(read_write_lock; what = () -> "nested_read") do
        @assert !has_write_lock(read_write_lock)
        @assert has_read_lock(read_write_lock)
    end
end

@assert !has_write_lock(read_write_lock)
@assert !has_read_lock(read_write_lock)

# output

```

```jldoctest; filter = r" 0x[a-f0-9]*"
using Base.Threads
using ConcurrentUtils

read_write_lock = ExtendedReadWriteLock()

@assert !has_write_lock(read_write_lock)
@assert !has_read_lock(read_write_lock)

lock_read(read_write_lock; what = "top_read") do
    @assert !has_write_lock(read_write_lock)
    @assert has_read_lock(read_write_lock)

    lock_write(read_write_lock; what = () -> "nested_write") do
        @assert false
    end
end

# output

ERROR: trying to obtain write lock for: nested_write
while holding read lock:
```
"""
module ExtendedReadWriteLocks

export ExtendedReadWriteLock
export has_read_lock
export has_write_lock
export lock_write
export trylock_write
export unlock_write

using Base.Threads
using ConcurrentUtils

using ..Types

import Base.AbstractLock

using ..MostlyReadWriteLocks

"""
    struct ExtendedReadWriteLock{BasicReadWriteLock} <: AbstractLock ... end
    where {ReadWriteLockType <: AbstractLock}

    ExtendedReadWriteLock(basic_read_write_lock::ReadWriteLockType)::ExtendedReadWriteLock{ReadWriteLockType}
    where {ReadWriteLockType <: AbstractLock}

    ExtendedReadWriteLock()::ExtendedReadWriteLock{MostlyReadWriteLock}

A read-write lock that supports extended operations (querying lock status). The `basic_read_write_lock` provides the
core functionality. It is wrapped to support the additional operations.
"""
struct ExtendedReadWriteLock{BasicReadWriteLock} <: AbstractLock
    lock::BasicReadWriteLock
end

function ExtendedReadWriteLock()::ExtendedReadWriteLock{MostlyReadWriteLock}
    return ExtendedReadWriteLock(MostlyReadWriteLock())
end

function Base.lock(
    read_write_lock::ExtendedReadWriteLock;
    what::Maybe{Union{Function, AbstractString}} = nothing,
)::Nothing
    private_storage = task_local_storage()
    lock_id = objectid(read_write_lock.lock)
    write_key = Symbol((lock_id, true))

    write_depth = get(private_storage, write_key, nothing)
    if write_depth !== nothing
        write_depth[1] += 1
        @debug "WLOCKED $(UInt64(lock_id)) $(write_depth[1])$(what_suffix(what))) {{{"
    else
        read_key = Symbol((lock_id, false))
        if haskey(private_storage, read_key)
            error("""
                trying to obtain write lock for:$(what_suffix(what))
                while holding read lock: $(UInt64(lock_id))
                """)
        end
        private_storage[write_key] = [1]
        @debug "WLOCK $(UInt64(lock_id)) 1$(what_suffix(what)) {{{"
        lock(read_write_lock.lock)
        @debug "WLOCKED $(UInt64(lock_id)) 1$(what_suffix(what))"
    end

    return nothing
end

function Base.unlock(
    read_write_lock::ExtendedReadWriteLock;
    what::Maybe{Union{Function, AbstractString}} = nothing,
)::Nothing
    private_storage = task_local_storage()
    lock_id = objectid(read_write_lock.lock)
    write_key = Symbol((lock_id, true))
    @assert haskey(private_storage, write_key)

    write_depth = private_storage[write_key]
    if write_depth[1] > 1
        write_depth[1] -= 1
        @debug "WUNLOCKED $(UInt64(lock_id)) $(write_depth[1])$(what_suffix(what)) }}}"
    else
        read_key = Symbol((lock_id, false))
        @assert !haskey(private_storage, read_key)
        @assert write_depth[1] == 1
        delete!(private_storage, write_key)
        unlock(read_write_lock.lock)
        @debug "WUNLOCKED $(UInt64(lock_id)) 0$(what_suffix(what)) }}}"
    end
    return nothing
end

"""
    has_write_lock(read_write_lock::ExtendedReadWriteLock)::Bool

Return whether the current task has the write lock.
"""
function has_write_lock(read_write_lock::ExtendedReadWriteLock)::Bool
    private_storage = task_local_storage()
    lock_id = objectid(read_write_lock.lock)
    write_key = Symbol((lock_id, true))
    return haskey(private_storage, write_key)
end

function ConcurrentUtils.lock_read(  # NOLINT
    read_write_lock::ExtendedReadWriteLock;
    what::Maybe{Union{Function, AbstractString}} = nothing,
)::Nothing
    private_storage = task_local_storage()
    lock_id = objectid(read_write_lock.lock)
    write_key = Symbol((lock_id, true))
    read_key = Symbol((lock_id, false))

    read_depth = get(private_storage, read_key, nothing)
    if read_depth !== nothing
        read_depth[1] += 1
        @debug "RLOCKED $(UInt64(lock_id)) $(read_depth[1])$(what_suffix(what)) {{{"
    else
        private_storage[read_key] = [1]
        if !haskey(private_storage, write_key)
            @debug "RLOCK $(UInt64(lock_id)) 1$(what_suffix(what)) {{{"
            lock_read(read_write_lock.lock)
        end
        @debug "RLOCKED $(UInt64(lock_id)) 1$(what_suffix(what))"
    end

    return nothing
end

function ConcurrentUtils.unlock_read(  # NOLINT
    read_write_lock::ExtendedReadWriteLock;
    what::Maybe{Union{AbstractString, Function}} = nothing,
)::Nothing
    private_storage = task_local_storage()
    lock_id = objectid(read_write_lock.lock)
    write_key = Symbol((lock_id, true))
    read_key = Symbol((lock_id, false))
    @assert haskey(private_storage, read_key)

    read_depth = private_storage[read_key]
    if read_depth[1] > 1
        read_depth[1] -= 1
        @debug "RUNLOCKED $(UInt64(lock_id)) $(read_depth[1])$(what_suffix(what)) }}}"
    else
        @assert read_depth[1] == 1
        delete!(private_storage, read_key)
        if !haskey(private_storage, write_key)
            unlock_read(read_write_lock.lock)
        end
        @debug "RUNLOCKED $(UInt64(lock_id)) 0 $(what_suffix(what)) }}}"
    end

    return nothing
end

"""
    has_read_lock(read_write_lock::ExtendedReadWriteLock; read_only::Bool = false)::Bool

Return whether the current task has a read lock or the write lock. If `read_only` is set, then this will only return
whether the current task as a read lock.
"""
function has_read_lock(read_write_lock::ExtendedReadWriteLock; read_only::Bool = false)::Bool
    private_storage = task_local_storage()
    lock_id = objectid(read_write_lock.lock)
    return haskey(private_storage, Symbol((lock_id, false))) ||
           (!read_only && haskey(private_storage, Symbol((lock_id, true))))
end

function Base.lock(
    action::Function,
    read_write_lock::ExtendedReadWriteLock;
    what::Maybe{Union{Function, AbstractString}} = nothing,
)::Any
    lock_write(read_write_lock; what)
    try
        return action()
    finally
        unlock_write(read_write_lock; what)
    end
end

function ConcurrentUtils.lock_read(  # NOLINT
    action::Function,
    read_write_lock::ExtendedReadWriteLock;
    what::Maybe{Union{Function, AbstractString}} = nothing,
)::Any
    lock_read(read_write_lock; what)
    try
        return action()
    finally
        unlock_read(read_write_lock; what)
    end
end

"""
    const lock_write = Base.lock

Alias for `Base.lock` for read-write locks to clarify the intent.
"""
const lock_write = Base.lock

"""
    const trylock_write = Base.lock

Alias for `Base.trylock` for read-write locks to clarify the intent.
"""
const trylock_write = Base.trylock

"""
    const unlock_write = Base.lock

Alias for `Base.unlock` for read-write locks to clarify the intent.
"""
const unlock_write = Base.unlock

function what_suffix(what::Function)::AbstractString
    return what_suffix(what())
end

function what_suffix(::Nothing)::AbstractString  # UNTESTED
    return ""
end

function what_suffix(what::AbstractString)::AbstractString
    return " " * what
end

end
