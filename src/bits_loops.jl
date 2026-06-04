"""
Efficient iterations on bits.
"""
module BitsLoops

export @foreach_true_index
export @foreach_true_index_position

"""
    @foreach_true_index bits index begin
        ...use index...
    end

Iterate on all indices for which `bits` are `true` and invoke `body(index)` for each. This efficiently skips 64-but
chunks so is reasonably efficient for sparse bit vectors. It also does this with no allocations (unlike using
`findall`).
"""
macro foreach_true_index(bits, index_var, body)
    return quote
        let chunks = $(esc(bits)).chunks, n_bits = length($(esc(bits)))
            @inbounds for chunk_index in eachindex(chunks)
                word = chunks[chunk_index]
                if chunk_index == length(chunks)
                    remainder = n_bits & 63
                    if remainder != 0
                        word &= (one(UInt64) << remainder) - one(UInt64)
                    end
                end
                base_index = (chunk_index - 1) * 64
                while word != 0
                    first_true_offset = trailing_zeros(word)
                    $(esc(index_var)) = base_index + first_true_offset + 1
                    # Clear the current bit before running the body so `continue` (and `break`) inside the body
                    # behave naturally - if the body skipped this, `continue` would re-iterate on the same bit.
                    word &= word - one(UInt64)
                    $(esc(body))
                end
            end
        end
    end
end

"""
    @foreach_true_index_position bits index position begin
        ...use index and position...
    end

Similar to [`@foreach_true_index`](@ref) but calls `body` with both the index of the entry and its position (the number
of true values before it).
"""
macro foreach_true_index_position(bits, index_var, position_var, body)
    return quote
        let chunks = $(esc(bits)).chunks, n_bits = length($(esc(bits)))
            $(esc(position_var)) = 0
            @inbounds for chunk_index in eachindex(chunks)
                word = chunks[chunk_index]
                if chunk_index == length(chunks)
                    remainder = n_bits & 63
                    if remainder != 0
                        word &= (one(UInt64) << remainder) - one(UInt64)
                    end
                end
                base_index = (chunk_index - 1) * 64
                while word != 0
                    first_true_offset = trailing_zeros(word)
                    $(esc(index_var)) = base_index + first_true_offset + 1
                    $(esc(position_var)) += 1
                    # Clear the current bit before running the body so `continue` (and `break`) inside the body
                    # behave naturally - if the body skipped this, `continue` would re-iterate on the same bit.
                    word &= word - one(UInt64)
                    $(esc(body))
                end
            end
        end
    end
end

end
