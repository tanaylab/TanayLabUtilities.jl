"""
A version of `cor` that returns 0.0 instead of NaN for undefined correlations.
"""
module ZeroCorrelation

export zero_cor_between_matrices_columns
export zero_cor_between_matrix_columns
export zero_cor_between_vector_and_matrix_columns
export zero_cor_between_vectors

using StatsBase

using ..FlameTime

"""
    zero_cor_between_matrix_columns(
        matrix::AbstractMatrix,
    )::AbstractMatrix

Same as `cor`, except that undefined cases return 0.0 instead of NaN. In addition, unlike `cor`, this doesn't set
entries to one when correlating matrices and a constant vector is correlated with itself. It properly sets all
correlations with such vectors to zero, even with themselves.

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
function zero_cor_between_matrix_columns(matrix::AbstractMatrix)::AbstractMatrix
    return zero_cor_between_matrices_columns(matrix, matrix)
end

"""
    zero_cor_between_matrices_columns(
        left_matrix::AbstractMatrix,
        right_matrix::AbstractMatrix
    )::AbstractMatrix

Same as `cor`, except that undefined cases return 0.0 instead of NaN. In addition, unlike `cor`, this doesn't set
entries to one when correlating matrices and a constant vector is correlated with itself. It properly sets all
correlations with such vectors to zero, even with themselves.
"""
function zero_cor_between_matrices_columns(left_matrix::AbstractMatrix, right_matrix::AbstractMatrix)::AbstractMatrix
    return flame_timed("zero_cor_between_matrices_columns") do
        min_per_left_column = vec(minimum(left_matrix; dims = 1))
        max_per_left_column = vec(maximum(left_matrix; dims = 1))
        is_zero_per_left_column = min_per_left_column .== max_per_left_column

        min_per_right_column = vec(minimum(right_matrix; dims = 1))
        max_per_right_column = vec(maximum(right_matrix; dims = 1))
        is_zero_per_right_column = min_per_right_column .== max_per_right_column

        correlation_per_left_column_per_right_column = cor(left_matrix, right_matrix)
        correlation_per_left_column_per_right_column[is_zero_per_left_column, :] .= 0
        correlation_per_left_column_per_right_column[:, is_zero_per_right_column] .= 0
        @assert !any(isnan.(correlation_per_left_column_per_right_column))
        return correlation_per_left_column_per_right_column
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
function zero_cor_between_vectors(left_vector::AbstractVector, right_vector::AbstractVector)::AbstractFloat
    return flame_timed("zero_cor_between_vectors") do
        if minimum(left_vector) == maximum(left_vector) || minimum(right_vector) == maximum(right_vector)
            return 0.0
        else
            return cor(left_vector, right_vector)
        end
    end
end

"""
    zero_cor_between_vector_and_matrix_columns(left_vector::AbstractVector, right_matrix::AbstractMatrix)::AbstractVector

Similar to `cor`, except that undefined cases return 0.0 instead of NaN, and returns a vector.

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
[1.0, -1.0]
```
"""
function zero_cor_between_vector_and_matrix_columns(
    left_vector::AbstractVector,
    right_matrix::AbstractMatrix,
)::AbstractVector
    return flame_timed("zero_cor_between_vector_and_matrix_columns") do
        if minimum(left_vector) == maximum(left_vector)
            return fill(0.0, size(right_matrix, 2))
        else
            min_per_right_column = vec(minimum(right_matrix; dims = 1))
            max_per_right_column = vec(maximum(right_matrix; dims = 1))
            is_zero_per_right_column = min_per_right_column .== max_per_right_column

            correlation_per_right_column = vec(cor.(Ref(left_vector), eachcol(right_matrix)))  # NOJET
            correlation_per_right_column[is_zero_per_right_column] .= 0
            return correlation_per_right_column
        end
    end
end

end
