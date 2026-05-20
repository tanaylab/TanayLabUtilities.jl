"""
A version of `cor` that returns 0.0 instead of NaN for undefined correlations.
"""
module ZeroCorrelation

export zero_cor_between_matrices_columns
export zero_cor_between_matrix_columns
export zero_cor_between_vector_and_matrix_columns
export zero_cor_between_vectors

using LinearAlgebra
using StatsBase

using ..FlameTime
using ..Types

"""
    zero_cor_between_matrix_columns(
        matrix::AbstractMatrix{<:Real};
        result::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
        scratch_matrix::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
        scratch_column::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    )::AbstractMatrix{<:AbstractFloat}

Same as `cor`, except that undefined cases return 0.0 instead of NaN. In addition, unlike `cor`, this doesn't set
entries to one when correlating matrices and a constant vector is correlated with itself. It properly sets all
correlations with such vectors to zero, even with themselves. This does zero allocations if the
optional parameters are provided.

```jldoctest
m = [1 2 3 4; 1 3 2 5];
zero_cor_between_matrix_columns(m)

# output

4×4 Matrix{Float64}:
 0.0   0.0   0.0   0.0
 0.0   1.0  -1.0   1.0
 0.0  -1.0   1.0  -1.0
 0.0   1.0  -1.0   1.0
```
"""
function zero_cor_between_matrix_columns(
    matrix::AbstractMatrix{<:Real};
    result::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
    scratch_matrix::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
    scratch_row::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
)::AbstractMatrix{<:AbstractFloat}
    return flame_timed("zero_cor_between_matrix_columns") do
        n_rows, n_columns = size(matrix)

        if result === nothing
            result = Matrix{Float64}(undef, n_columns, n_columns)
        else
            @assert size(result) == (n_columns, n_columns)  # UNTESTED
        end

        if scratch_matrix === nothing
            scratch_matrix = Matrix{Float64}(undef, n_rows, n_columns)
        else
            @assert size(scratch_matrix) == size(matrix)  # UNTESTED
        end

        if scratch_row === nothing
            scratch_row = Vector{Float64}(undef, n_columns)
        else
            @assert length(scratch_row) == n_columns  # UNTESTED
        end

        normalized = scratch_matrix
        norms = scratch_row

        for column_index in 1:n_columns
            @views column = matrix[:, column_index]
            @views normalized_column = normalized[:, column_index]
            mean_column = sum(column) / n_rows
            normalized_column .= column .- mean_column
            column_norm = norm(normalized_column)
            if column_norm == 0 || minimum(column) == maximum(column)
                normalized_column .= 0
                norms[column_index] = 1
            else
                norms[column_index] = column_norm
            end
        end

        normalized ./= transpose(norms)
        mul!(result, transpose(normalized), normalized)

        for column_index in 1:n_columns
            if norms[column_index] == 1 && normalized[1, column_index] == 0
                result[:, column_index] .= 0
                result[column_index, :] .= 0
            end
        end

        return result
    end
end

"""
    zero_cor_between_matrices_columns(
        left_matrix::AbstractMatrix{<:Real},
        right_matrix::AbstractMatrix{<:Real};
        result::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
        left_scratch_matrix::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
        right_scratch_matrix::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
        left_scratch_row::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
        right_scratch_row::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    )::AbstractMatrix{<:AbstractFloat}

Same as `cor`, except that undefined cases return 0.0 instead of NaN. This does zero allocations if the optional
parameters are provided.

```jldoctest
m = [1 2 3 4; 1 3 2 5];
n = [1 2 3; 1 3 2];
zero_cor_between_matrices_columns(m, n)

# output

4×3 Matrix{Float64}:
 0.0   0.0   0.0
 0.0   1.0  -1.0
 0.0  -1.0   1.0
 0.0   1.0  -1.0
```
"""
function zero_cor_between_matrices_columns(
    left_matrix::AbstractMatrix{<:Real},
    right_matrix::AbstractMatrix{<:Real};
    result::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
    left_scratch_matrix::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
    right_scratch_matrix::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
    left_scratch_row::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    right_scratch_row::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
)::AbstractMatrix{<:AbstractFloat}
    return flame_timed("zero_cor_between_matrices_columns") do
        n_rows, n_left_columns = size(left_matrix)
        n_right_columns = size(right_matrix, 2)
        @assert size(right_matrix, 1) == n_rows

        if result === nothing
            result = Matrix{Float64}(undef, n_left_columns, n_right_columns)
        else
            @assert size(result) == (n_left_columns, n_right_columns)  # UNTESTED
        end

        if left_scratch_matrix === nothing
            left_scratch_matrix = Matrix{Float64}(undef, n_rows, n_left_columns)
        else
            @assert size(left_scratch_matrix) == size(left_matrix)  # UNTESTED
        end

        if right_scratch_matrix === nothing
            right_scratch_matrix = Matrix{Float64}(undef, n_rows, n_right_columns)
        else
            @assert size(right_scratch_matrix) == size(right_matrix)  # UNTESTED
        end

        if left_scratch_row === nothing
            left_scratch_row = Vector{Float64}(undef, n_left_columns)
        else
            @assert length(left_scratch_row) == n_left_columns  # UNTESTED
        end

        if right_scratch_row === nothing
            right_scratch_row = Vector{Float64}(undef, n_right_columns)
        else
            @assert length(right_scratch_row) == n_right_columns  # UNTESTED
        end

        left_normalized = left_scratch_matrix
        right_normalized = right_scratch_matrix
        left_column_norms = left_scratch_row
        right_column_norms = right_scratch_row

        for column_index in 1:n_left_columns
            @views left_column = left_matrix[:, column_index]
            @views left_normalized_column = left_normalized[:, column_index]
            mean_column = sum(left_column) / n_rows
            left_normalized_column .= left_column .- mean_column
            left_norm = norm(left_normalized_column)
            if left_norm == 0 || minimum(left_column) == maximum(left_column)
                left_normalized_column .= 0
                left_column_norms[column_index] = 1
            else
                left_column_norms[column_index] = left_norm
            end
        end
        left_normalized ./= transpose(left_column_norms)

        for column_index in 1:n_right_columns
            @views right_column = right_matrix[:, column_index]
            @views right_normalized_column = right_normalized[:, column_index]
            mean_column = sum(right_column) / n_rows
            right_normalized_column .= right_column .- mean_column
            right_norm = norm(right_normalized_column)
            if right_norm == 0 || minimum(right_column) == maximum(right_column)
                right_normalized_column .= 0
                right_column_norms[column_index] = 1
            else
                right_column_norms[column_index] = right_norm
            end
        end
        right_normalized ./= transpose(right_column_norms)

        mul!(result, transpose(left_normalized), right_normalized)

        for column_index in 1:n_right_columns
            for row_index in 1:n_left_columns
                if isnan(result[row_index, column_index])
                    result[row_index, column_index] = 0  # UNTESTED
                end
            end
        end

        return result
    end
end

"""
    zero_cor_between_vectors(left_vector::AbstractVector, right_vector::AbstractVector)::AbstractFloat

Same as `cor`, except that undefined cases return 0.0 instead of NaN.

```jldoctest
using StatsBase

println(cor([1.0, 1.0], [1.0, 2.0]))
println(cor([1.0, 2.0], [1.0, 2.0]))

# output

NaN
1.0
```

```jldoctest
println(zero_cor_between_vectors([1.0, 1.0], [1.0, 2.0]))
println(zero_cor_between_vectors([1.0, 2.0], [1.0, 2.0]))

# output

0.0
1.0
```
"""
function zero_cor_between_vectors(
    left_vector::AbstractVector{<:Real},
    right_vector::AbstractVector{<:Real},
)::AbstractFloat
    return flame_timed("zero_cor_between_vectors") do
        if minimum(left_vector) == maximum(left_vector) || minimum(right_vector) == maximum(right_vector)
            return 0.0
        else
            return cor(left_vector, right_vector)
        end
    end
end

"""
    zero_cor_between_vector_and_matrix_columns(
        left_vector::AbstractVector{<:Real},
        right_matrix::AbstractMatrix{<:Real};
        result::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
        result_scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
        left_scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
        right_scratch::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
    )::AbstractVector{<:AbstractFloat}

Similar to `cor`, except that undefined cases return 0.0 instead of NaN, and returns a vector. This does zero
allocations if the optional parameters are provided.

```jldoctest
using StatsBase

println(cor.(Ref([1.0, 1.0]), eachcol([1.0 2.0; 2.0 1.0])))
println(cor.(Ref([1.0, 2.0]), eachcol([1.0 2.0; 2.0 1.0])))

# output

[NaN, NaN]
[1.0, -1.0]
```

```jldoctest
println(zero_cor_between_vector_and_matrix_columns([1.0, 1.0], [1.0 2.0; 2.0 1.0]))
println(zero_cor_between_vector_and_matrix_columns([1.0, 2.0], [1.0 2.0; 2.0 1.0]))

# output

[0.0, 0.0]
[0.9999999999999998, -0.9999999999999998]
```
"""
function zero_cor_between_vector_and_matrix_columns(
    left_vector::AbstractVector{<:Real},
    right_matrix::AbstractMatrix{<:Real};
    result::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    result_scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    left_scratch::Maybe{AbstractVector{<:AbstractFloat}} = nothing,
    right_scratch::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
)::AbstractVector{<:AbstractFloat}
    return flame_timed("zero_cor_between_vector_and_matrix_columns") do
        n_rows, n_columns = size(right_matrix)
        @assert length(left_vector) == n_rows

        if result === nothing
            result = Vector{Float64}(undef, n_columns)
        else
            @assert length(result) == n_columns  # UNTESTED
        end

        if result_scratch === nothing
            result_scratch = Vector{Float64}(undef, n_columns)
        else
            @assert length(result_scratch) == n_columns  # UNTESTED
        end

        if left_scratch === nothing
            left_scratch = Vector{Float64}(undef, n_rows)
        else
            @assert length(left_scratch) == n_rows  # UNTESTED
        end

        if right_scratch === nothing
            right_scratch = Matrix{Float64}(undef, n_rows, n_columns)
        else
            @assert size(right_scratch) == (n_rows, n_columns)  # UNTESTED
        end

        left_normalized = left_scratch
        right_normalized = right_scratch
        right_column_norms = result_scratch

        mean_left = sum(left_vector) / n_rows
        left_normalized .= left_vector .- mean_left
        left_norm = norm(left_normalized)
        if left_norm == 0 || minimum(left_vector) == maximum(left_vector)
            result .= 0
            return result
        end
        left_normalized ./= left_norm

        for column_index in 1:n_columns
            @views right_column = right_matrix[:, column_index]
            @views right_normalized_column = right_normalized[:, column_index]
            mean_column = sum(right_column) / n_rows
            right_normalized_column .= right_column .- mean_column
            right_norm = norm(right_normalized_column)
            if right_norm == 0 || minimum(right_column) == maximum(right_column)
                right_normalized_column .= 0  # UNTESTED
                right_column_norms[column_index] = 1  # UNTESTED
            else
                right_column_norms[column_index] = right_norm
            end
        end
        right_normalized ./= transpose(right_column_norms)

        mul!(result, transpose(right_normalized), left_normalized)

        for column_index in 1:n_columns
            if isnan(result[column_index])
                result[column_index] = 0  # UNTESTED
            end
        end

        return result
    end
end

end
