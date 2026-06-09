"""
Chi-squared test for 2x2 contingency tables derived from two count vectors.
"""
module ChiSquared

export chi_squared

using LoopVectorization
using SpecialFunctions

using ..FlameTime
using ..MatrixLayouts
using ..Types

"""
    chi_squared(
        first_column::AbstractVector{<:Real},
        second_column::AbstractVector{<:Real};
        yates::Bool = true,
    )::Matrix{Float64}

    chi_squared(
        matrix::AbstractMatrix{<:Real};
        yates::Bool = true,
    )::Matrix{Float64}

Compute chi-squared statistics and p-values for 2x2 contingency tables. The two input vectors (or the two columns of
the input matrix) contain non-negative counts. Each entry defines one 2x2 table together with the column sums:

|              | First          | Second         |
|:------------ |:-------------- |:-------------- |
| Current entry| `a`            | `c`            |
| Other entries| `sum1 - a`     | `sum2 - c`     |

If `yates` is `true` (the default), Yates' continuity correction is applied. The inputs must not contain `NaN` values.
The function works with both dense and sparse inputs.

Returns an `n x 3` matrix where the first column contains the chi-squared statistics, the second column contains
the p-values (for chi-squared with 1 degree of freedom, computed as `erfc(sqrt(chi2 / 2))`), and the third column
contains the FDR-corrected p-values (Benjamini-Hochberg procedure).

```jldoctest
result = chi_squared([10, 30, 50], [20, 40, 60])
@assert size(result) == (3, 3)
@views @assert all(result[:, 1] .>= 0)
@views @assert all(0 .<= result[:, 2] .<= 1)
@views @assert all(0 .<= result[:, 3] .<= 1)
@views @assert all(result[:, 3] .>= result[:, 2] .- 1e-12)

# output

```

```jldoctest
result_yates = chi_squared([10, 30], [20, 40]; yates = true)
result_no_yates = chi_squared([10, 30], [20, 40]; yates = false)
@views @assert all(result_yates[:, 1] .<= result_no_yates[:, 1] .+ 1e-12)
@views @assert all(result_yates[:, 2] .>= result_no_yates[:, 2] .- 1e-12)
@views @assert all(result_yates[:, 3] .>= result_yates[:, 2] .- 1e-12)

# output

```

```jldoctest
result = chi_squared([10 20; 30 40; 50 60])
@assert size(result) == (3, 3)

# output

```
"""
function chi_squared(
    first_column::AbstractVector{<:Real},
    second_column::AbstractVector{<:Real};
    yates::Bool = true,
)::Matrix{Float64}
    return flame_timed("chi_squared") do
        n_rows = length(first_column)
        @assert length(second_column) == n_rows "vectors must have the same length"
        @assert n_rows >= 1 "vectors must have at least 1 entry"
        @assert all(first_column .>= 0) "first vector must contain non-negative counts"
        @assert all(second_column .>= 0) "second vector must contain non-negative counts"

        first_values = LoopVectorization.check_args(first_column) ? first_column : Vector{Float64}(first_column)
        second_values = LoopVectorization.check_args(second_column) ? second_column : Vector{Float64}(second_column)

        first_column_sum = Float64(sum(first_values))
        second_column_sum = Float64(sum(second_values))
        n_total = first_column_sum + second_column_sum
        half_n_total = n_total / 2.0

        result = Matrix{Float64}(undef, n_rows, 3)
        @views result_chi_squared = result[:, 1]
        @views result_p_values = result[:, 2]
        @views result_fdr = result[:, 3]

        @check_turbo_vector(first_values)
        @check_turbo_vector(second_values)
        @check_turbo_vector(result_chi_squared)

        if yates
            @turbo for row_index in 1:n_rows
                first_count = first_values[row_index]
                second_count = second_values[row_index]
                first_remainder = first_column_sum - first_count
                second_remainder = second_column_sum - second_count
                row_total = first_count + second_count
                remainder_total = first_remainder + second_remainder
                denominator = first_column_sum * second_column_sum * row_total * remainder_total
                difference = first_count * second_remainder - second_count * first_remainder
                corrected = max(abs(difference) - half_n_total, 0.0)
                result_chi_squared[row_index] =
                    ifelse(denominator == 0.0, 0.0, corrected * corrected * n_total / denominator)
            end
        else
            @turbo for row_index in 1:n_rows
                first_count = first_values[row_index]
                second_count = second_values[row_index]
                first_remainder = first_column_sum - first_count
                second_remainder = second_column_sum - second_count
                row_total = first_count + second_count
                remainder_total = first_remainder + second_remainder
                denominator = first_column_sum * second_column_sum * row_total * remainder_total
                difference = first_count * second_remainder - second_count * first_remainder
                result_chi_squared[row_index] =
                    ifelse(denominator == 0.0, 0.0, difference * difference * n_total / denominator)
            end
        end

        @inbounds for row_index in 1:n_rows
            result_p_values[row_index] = erfc(sqrt(result_chi_squared[row_index] / 2.0))
        end

        order = sortperm(result_p_values)
        sorted_fdr = Vector{Float64}(undef, n_rows)
        n_rows_float = Float64(n_rows)

        @inbounds for rank in 1:n_rows
            sorted_fdr[rank] = result_p_values[order[rank]]
        end

        @check_turbo_vector(sorted_fdr)
        @turbo for rank in 1:n_rows
            sorted_fdr[rank] = min(sorted_fdr[rank] * n_rows_float / rank, 1.0)
        end

        @inbounds for rank in (n_rows - 1):-1:1
            sorted_fdr[rank] = min(sorted_fdr[rank], sorted_fdr[rank + 1])
        end

        @inbounds for rank in 1:n_rows
            result_fdr[order[rank]] = sorted_fdr[rank]
        end

        return result
    end
end

function chi_squared(matrix::AbstractMatrix{<:Real}; yates::Bool = true)::Matrix{Float64}
    n_columns = size(matrix, 2)
    @assert n_columns == 2 "matrix must have exactly 2 columns, not $(n_columns)"
    @views return chi_squared(matrix[:, 1], matrix[:, 2]; yates)
end

end  # module
