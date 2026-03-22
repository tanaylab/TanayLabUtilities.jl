"""
Read-oriented (reentrant) read-write locks.
"""
module MostlyReadWriteLocks

export MostlyReadWriteLock

using Base.Threads
using ConcurrentUtils

import Base.AbstractLock  # NOLINT

using ..FlameTime
using ..Types

"""
    mutable struct MostlyReadWriteLock <: AbstractLock ... end

Drop-in replacement for `ConcurrentUtils.ReadWriteLock` which is optimized to be more efficient for many fast readers
and rare slow writers. That is, code assumes that read locks are common and are held for a short time (in particular,
spinning waiting for readers to release their lock is sensible) while write locks are uncommon and held for a long time
(that is, waiting for a writer to release its lock via a proper mutex and a context switch is sensible).
"""
mutable struct MostlyReadWriteLock <: AbstractLock
    readers::Atomic{Int}
    writer_active::Atomic{Bool}
    writers_waiting::Atomic{Int}
    write_mutex::ReentrantLock

    MostlyReadWriteLock() = new(Atomic{Int}(0), Atomic{Bool}(false), Atomic{Int}(0), ReentrantLock())
end

const MAX_SPINS = 100
const PAUSE_CYCLES = 10

@inline function cpu_pause()::Nothing  # UNTESTED
    # x86 PAUSE instruction equivalent
    ccall(:jl_cpu_pause, Cvoid, ())
    return nothing
end

function Base.lock(lock::MostlyReadWriteLock)::Nothing
    atomic_add!(lock.writers_waiting, 1)

    flame_timed("ReentrantLock.lock") do
        return Base.lock(lock.write_mutex)  # NOLINT
    end

    atomic_sub!(lock.writers_waiting, 1)
    lock.writer_active[] = true

    spins = 0

    while lock.readers[] > 0
        if spins < MAX_SPINS  # UNTESTED
            spins += 1  # UNTESTED
            for _ in 1:PAUSE_CYCLES  # UNTESTED
                cpu_pause()  # UNTESTED
            end
            GC.safepoint()  # UNTESTED
        else
            yield()  # UNTESTED
            spins = 0  # UNTESTED
        end
    end

    return nothing
end

function Base.unlock(lock::MostlyReadWriteLock)::Nothing
    lock.writer_active[] = false
    flame_timed("ReentrantLock.unlock") do
        return Base.unlock(lock.write_mutex)
    end
    return nothing
end

function Base.trylock(lock::MostlyReadWriteLock)::Bool  # UNTESTED
    if lock.readers[] > 0
        return false
    end

    if !trylock(lock.write_mutex)
        return false
    end

    if lock.readers[] > 0
        Base.unlock(lock.write_mutex)
        return false
    end

    lock.writer_active[] = true
    return true
end

function Base.lock(action::Function, lock::MostlyReadWriteLock)::Any
    Base.lock(lock)  # NOLINT
    try
        return action()
    finally
        Base.unlock(lock)
    end
end

function ConcurrentUtils.lock_read(lock::MostlyReadWriteLock)::Nothing
    while lock.writers_waiting[] > 0 || lock.writer_active[]
        yield()  # UNTESTED
    end

    atomic_add!(lock.readers, 1)

    while lock.writer_active[]
        atomic_sub!(lock.readers, 1)  # UNTESTED

        while lock.writer_active[] || lock.writers_waiting[] > 0  # UNTESTED
            yield()  # UNTESTED
        end

        atomic_add!(lock.readers, 1)  # UNTESTED
    end

    return nothing
end

function ConcurrentUtils.lock_read(action::Function, lock::MostlyReadWriteLock)::Any
    lock_read(lock)
    try
        return action()
    finally
        unlock_read(lock)
    end
end

function ConcurrentUtils.trylock_read(lock::MostlyReadWriteLock)::Bool  # UNTESTED
    if lock.writer_active[] || lock.writers_waiting[] > 0
        return false
    end

    atomic_add!(lock.readers, 1)

    if lock.writer_active[]
        atomic_sub!(lock.readers, 1)
        return false
    end

    return true
end

function ConcurrentUtils.trylock_read(action::Function, lock::MostlyReadWriteLock)::Maybe{Some{Any}}  # NOLINT # UNTESTED
    if trylock_read(lock)
        try
            return Some(action())
        finally
            read_unlock(lock)  # NOLINT # NOJET
        end
    else
        return nothing
    end
end

function ConcurrentUtils.unlock_read(lock::MostlyReadWriteLock)::Nothing
    atomic_sub!(lock.readers, 1)
    return nothing
end

end
