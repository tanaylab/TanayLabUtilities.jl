"""
Read-only vectors and matrices, by ab/using `SparseArrays.ReadOnly`. We need this because we want access to shared data
to be copy-free for efficiency (in particular, matrices can be several GBs). However, it would be very easy for the user
code to access vector or matrix data and accidentally modify it in-place, thereby corrupting it and causing all sort of
hard-to-debug hilarity. Julia in its infinite wisdom takes the view that "everything is mutable" so has no builtin
notion of "read-only view of an array", probably due to the general weakness of its type system (which might be a price
paid for efficient multiple dispatch?). Luckily the `SparseArraysReadOnly` package happens to implement something along these
lines, which we shamelessly ab/use for our purposes and re-export as `ReadOnlyArray`.

!!! note

    The read-only array functions below are restricted to dealing with normal (dense) arrays, `SparseArrays`,
    `NamedArrays`, `PermutedDimsArray`, and `LinearAlgebra` arrays (specifically, `Transpose` and `Adjoint`), as these
    are the types we actually use. YMMV if using more exotic matrix types. In theory you could extend the implementation
    to cover such types as well.

When using "wrapper" array types, we try to push the "read-only-ness" as far down the stack of wrappers as possible.
Thus applying this to a `NamedArray`, `PermutedDimsArray`, `Transpose`, `Adjoint` will wrap the `parent` of the array
rather than the array itself. However `SparseArrays` have chosen to implement `ReadOnly` as a wrapper around
`SparseArray` rather than as a wrapper for the internal data, presumably because it would have required them to
separately wrap each of the internal arrays. IMVHO it would have been better if they went that way anyway and in general
would have allowed for a wider range of array types for the internal arrays, as that would have allowed all sort of
interesting features (e.g., a sparse array whose structure is fixed but only the non-zero values are mutable, a
zero-copy sparse array whose data is a slice of a larger array, etc.). In general Julia's type hierarchy for arrays has
some rough spots we need to deal with, partially because of the type system limitations, and partially because of some
debatable decisions made when defining the core array types and conventions.
"""
module ReadOnlyArrays

export mutable_array
export is_read_only_array
export read_only_array
export ReadOnlyArray

using LinearAlgebra
using NamedArrays
using SparseArrays

"""
Just a rename (and re-export) of the internal ``ReadOnlyArray``.
"""
const ReadOnlyArray = SparseArrays.ReadOnly

"""
    read_only_array(array::AbstractArray):AbstractArray

An immutable view of an `array`, using `SparseArrays.ReadOnly`. If the array is already immutable, it is returned as-is.

```jldoctest
# Base

base = [0 1 2; 3 4 0]
@assert !is_read_only_array(base)

read_only = read_only_array(base)
@assert is_read_only_array(read_only)
@assert read_only_array(read_only) === read_only

# Named

using NamedArrays

named = NamedArray(base)
@assert !is_read_only_array(named)

named_read_only = read_only_array(named)
@assert named_read_only isa NamedArray
@assert is_read_only_array(named_read_only)
@assert read_only_array(named_read_only) === named_read_only

# Permuted

permuted = PermutedDimsArray(base, (2, 1))
@assert !is_read_only_array(permuted)

permuted_read_only = read_only_array(permuted)
@assert permuted_read_only isa PermutedDimsArray
@assert is_read_only_array(permuted_read_only)
@assert read_only_array(permuted_read_only) === permuted_read_only

unpermuted = PermutedDimsArray(base, (1, 2))
@assert !is_read_only_array(unpermuted)

unpermuted_read_only = read_only_array(unpermuted)
@assert unpermuted_read_only isa PermutedDimsArray
@assert is_read_only_array(unpermuted_read_only)
@assert read_only_array(unpermuted_read_only) === unpermuted_read_only

# LinearAlgebra

using LinearAlgebra

transposed = transpose(base)
@assert !is_read_only_array(transposed)

transposed_read_only = read_only_array(transposed)
@assert transposed_read_only isa Transpose
@assert is_read_only_array(transposed_read_only)
@assert read_only_array(transposed_read_only) === transposed_read_only

adjointed = adjoint(base)
@assert !is_read_only_array(adjointed)

adjointed_read_only = read_only_array(adjointed)
@assert adjointed_read_only isa Adjoint
@assert is_read_only_array(adjointed_read_only)
@assert read_only_array(adjointed_read_only) === adjointed_read_only

# Sparse

using SparseArrays

sparse = SparseMatrixCSC(base)
@assert !is_read_only_array(sparse)

sparse_read_only = read_only_array(sparse)
@assert issparse(sparse_read_only)
@assert is_read_only_array(sparse_read_only)
@assert is_read_only_array(sparse_read_only)
@assert read_only_array(sparse_read_only) === sparse_read_only

# output
```
"""
function read_only_array(array::AbstractArray)::AbstractArray
    return SparseArrays.ReadOnly(array)
end

function read_only_array(array::ReadOnlyArray)::AbstractArray
    return array
end

function read_only_array(array::PermutedDimsArray{T, 2, P, IP, A})::PermutedDimsArray where {T, P, IP, A}
    parent_array = parent(array)
    read_only_parent_array = read_only_array(parent_array)
    if read_only_parent_array === parent_array
        return array
    else
        return PermutedDimsArray(read_only_parent_array, P)
    end
end

function read_only_array(array::Transpose)::Transpose
    parent_array = parent(array)
    read_only_parent_array = read_only_array(parent_array)
    if read_only_parent_array === parent_array
        return array
    else
        return Transpose(read_only_parent_array)
    end
end

function read_only_array(array::Adjoint)::Adjoint
    parent_array = parent(array)
    read_only_parent_array = read_only_array(parent_array)
    if read_only_parent_array === parent_array
        return array
    else
        return Adjoint(read_only_parent_array)
    end
end

function read_only_array(array::NamedArray)::NamedArray
    parent_array = array.array
    read_only_parent_array = read_only_array(parent_array)
    if read_only_parent_array === parent_array
        return array
    else
        return NamedArray(read_only_parent_array, array.dicts, array.dimnames)
    end
end

"""
    is_read_only_array(array::AbstractArray)::Bool

Return whether an `array` is immutable. See [`ReadOnlyArray`](@ref)
"""
function is_read_only_array(array::AbstractArray)::Bool
    return mutable_array(array) !== array
end

"""
    mutable_array(array::AbstractArray)::AbstractArray

Grant mutable access to an array, even if it is a read-only array. If the array is already mutable, it is returned
as-is.

!!! note

    **This should be used with great care** because code depends on read-only arrays not changing their values.

```jldoctest
base = [0 1 2; 3 4 0]

@assert mutable_array(base) === base

read_only = read_only_array(base)
@assert mutable_array(read_only) === base

# Named

using NamedArrays

named = NamedArray(base)
@assert !is_read_only_array(named)

@assert mutable_array(named) === named
@assert mutable_array(read_only_array(named)).array === named.array

# Permuted

permuted = PermutedDimsArray(base, (2, 1))

@assert mutable_array(permuted) === permuted
@assert mutable_array(read_only_array(permuted)) === permuted

unpermuted = PermutedDimsArray(base, (1, 2))
@assert !is_read_only_array(unpermuted)

@assert mutable_array(unpermuted) === unpermuted
@assert mutable_array(read_only_array(unpermuted)) === unpermuted

# LinearAlgebra

using LinearAlgebra

transposed = transpose(base)
@assert !is_read_only_array(transposed)

@assert mutable_array(transposed) === transposed
@assert mutable_array(read_only_array(transposed)) === transposed

adjointed = adjoint(base)
@assert !is_read_only_array(adjointed)

@assert mutable_array(adjointed) === adjointed
@assert mutable_array(read_only_array(adjointed)) === adjointed

# Sparse

using SparseArrays

sparse = SparseMatrixCSC(base)

@assert mutable_array(sparse) === sparse
@assert mutable_array(read_only_array(sparse)) === sparse

# output
```
"""
function mutable_array(array::AbstractArray)::AbstractArray
    return array
end

function mutable_array(array::PermutedDimsArray{T, 2, P, IP, A})::PermutedDimsArray where {T, P, IP, A}
    parent_array = parent(array)
    mutable_parent_array = mutable_array(parent_array)
    if mutable_parent_array === parent_array
        return array
    else
        return PermutedDimsArray(mutable_parent_array, P)
    end
end

function mutable_array(array::Transpose)::Transpose
    parent_array = parent(array)
    mutable_parent_array = mutable_array(parent_array)
    if mutable_parent_array === parent_array
        return array
    else
        return Transpose(mutable_parent_array)
    end
end

function mutable_array(array::Adjoint)::Adjoint
    parent_array = parent(array)
    mutable_parent_array = mutable_array(parent_array)
    if mutable_parent_array === parent_array
        return array
    else
        return Adjoint(mutable_parent_array)
    end
end

function mutable_array(array::SparseArrays.ReadOnly)::AbstractArray
    return parent(array)
end

function mutable_array(array::NamedArray)::NamedArray
    parent_array = parent(array)
    mutable_parent_array = mutable_array(parent_array)
    if mutable_parent_array === parent_array
        return array
    else
        return NamedArray(mutable_parent_array, array.dicts, array.dimnames)
    end
end

end # module
