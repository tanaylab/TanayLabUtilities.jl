"""
Incremental Pearson correlations, one per series, between a fixed set of per-point values and a mutable set of per-point
values, where the points are partitioned into consecutive groups. Replacing the variable values of a single group
cheaply updates every per-series correlation. A per-point `is_active` mask filters which points contribute to the
correlations - inactive points are excluded from every sum (both the fixed totals and the variable/product sums) and
from the effective `n` used in the correlation. The mask can be supplied at build, and a new mask for a single group
can be passed on a `replace_group!` / `mean_correlation_if_group_replaced` call.
"""
module GroupedCorrelations

export GroupedSeriesCorrelations
export correlation_per_series!
export mean_correlation
export mean_correlation_if_group_replaced
export replace_group!

using LoopVectorization

using ..FlameTime
using ..MatrixLayouts
using ..Types

"""
Incremental Pearson correlations between a fixed and a variable value per point, for each of several series, where the
points are partitioned into `n_groups` consecutive groups. Values rows are points and columns are series. Points are
laid out such that the points in each group are in consecutive rows. A per-point `is_active` mask filters which points
contribute - inactive points are excluded from every sum and from the effective `n` used in the correlation. The per-
group active counts are tracked so that mask changes during `replace_group!` cheaply update both the fixed-side totals
(via a transition delta) and the effective `n`.
"""
struct GroupedSeriesCorrelations{F <: AbstractFloat}
    n_points::Int
    n_series::Int
    n_groups::Int
    first_point_index_per_group::Vector{Int}
    fixed_per_point_per_series::Matrix{F}
    is_active_per_point::Vector{Bool}
    n_active_per_group::Vector{Int}
    sum_fixed_per_series::Vector{Float64}
    sum_squared_fixed_per_series::Vector{Float64}
    sum_variable_per_series::Vector{Float64}
    sum_squared_variable_per_series::Vector{Float64}
    sum_product_per_series::Vector{Float64}
    sum_variable_per_series_per_group::Matrix{Float32}
    sum_squared_variable_per_series_per_group::Matrix{Float32}
    sum_product_per_series_per_group::Matrix{Float32}
end

"""
    GroupedSeriesCorrelations(
        fixed_per_point_per_series::AbstractMatrix{F},
        variable_per_point_per_series::AbstractMatrix{<:Real},
        n_points_per_group::AbstractVector{<:Integer};
        is_active_per_point::Maybe{AbstractVector{Bool}} = nothing,
    )::GroupedSeriesCorrelations{F} where {F <: AbstractFloat}

Build the incremental correlations from the initial fixed and variable values. The points must already be ordered group
by group, with `n_points_per_group` points belonging to the groups (so `sum(n_points_per_group) == n_points`). The fixed
values are retained (the caller must not mutate them afterwards); the variable values are consumed into sums and
discarded. The optional `is_active_per_point` mask (default `nothing` = all points active) filters which points
contribute to the correlations.
"""
function GroupedSeriesCorrelations(
    fixed_per_point_per_series::AbstractMatrix{F},
    variable_per_point_per_series::AbstractMatrix{<:Real},
    n_points_per_group::AbstractVector{<:Integer};
    is_active_per_point::Maybe{AbstractVector{Bool}} = nothing,
)::GroupedSeriesCorrelations{F} where {F <: AbstractFloat}
    return flame_timed("GroupedSeriesCorrelations") do
        n_points, n_series = size(fixed_per_point_per_series)
        @assert_matrix(variable_per_point_per_series, n_points, n_series)
        @check_turbo_matrix(fixed_per_point_per_series)
        @check_turbo_matrix(variable_per_point_per_series)
        n_groups = length(n_points_per_group)
        @assert sum(n_points_per_group) == n_points

        first_point_index_per_group = Vector{Int}(undef, n_groups + 1)
        first_point_index_per_group[1] = 1
        for group_index in 1:n_groups
            first_point_index_per_group[group_index + 1] =
                first_point_index_per_group[group_index] + n_points_per_group[group_index]
        end

        # Default-active mask, or copy the supplied one.
        if is_active_per_point === nothing
            is_active = fill(true, n_points)
        else
            @assert_vector(is_active_per_point, n_points)
            is_active = Vector{Bool}(is_active_per_point)
        end
        @check_turbo_vector(is_active)

        # Per-group active counts.
        n_active_per_group = Vector{Int}(undef, n_groups)
        for group_index in 1:n_groups
            first_point_index = first_point_index_per_group[group_index]
            n_group_points = first_point_index_per_group[group_index + 1] - first_point_index
            n_active_per_group[group_index] = count_active_in_group(is_active, first_point_index, n_group_points)
        end

        # Per-series totals in full Float64 over active points, fused so no Float64 copies of the inputs are needed.
        sum_fixed_per_series = Vector{Float64}(undef, n_series)
        sum_squared_fixed_per_series = Vector{Float64}(undef, n_series)
        sum_variable_per_series = Vector{Float64}(undef, n_series)
        sum_squared_variable_per_series = Vector{Float64}(undef, n_series)
        sum_product_per_series = Vector{Float64}(undef, n_series)
        for series_index in 1:n_series
            @views fixed_per_point = fixed_per_point_per_series[:, series_index]
            @views variable_per_point = variable_per_point_per_series[:, series_index]
            sum_x = 0.0
            sum_xx = 0.0
            sum_y = 0.0
            sum_yy = 0.0
            sum_xy = 0.0
            @turbo for point_index in 1:n_points
                contribution = Float64(is_active[point_index])
                fixed_value = Float64(fixed_per_point[point_index])
                variable_value = Float64(variable_per_point[point_index])
                sum_x += contribution * fixed_value
                sum_xx += contribution * fixed_value * fixed_value
                sum_y += contribution * variable_value
                sum_yy += contribution * variable_value * variable_value
                sum_xy += contribution * fixed_value * variable_value
            end
            sum_fixed_per_series[series_index] = sum_x
            sum_squared_fixed_per_series[series_index] = sum_xx
            sum_variable_per_series[series_index] = sum_y
            sum_squared_variable_per_series[series_index] = sum_yy
            sum_product_per_series[series_index] = sum_xy
        end

        # Per-group sums in Float32, matching the incremental `masked_group_sums` path used by `replace_group!`.
        sum_variable_per_series_per_group = Matrix{Float32}(undef, n_series, n_groups)
        sum_squared_variable_per_series_per_group = Matrix{Float32}(undef, n_series, n_groups)
        sum_product_per_series_per_group = Matrix{Float32}(undef, n_series, n_groups)
        for group_index in 1:n_groups
            first_point_index = first_point_index_per_group[group_index]
            n_group_points = first_point_index_per_group[group_index + 1] - first_point_index
            for series_index in 1:n_series
                (group_sum_y, group_sum_yy, group_sum_xy) = masked_group_sums(
                    fixed_per_point_per_series,
                    first_point_index,
                    variable_per_point_per_series,
                    first_point_index,
                    is_active,
                    first_point_index,
                    n_group_points,
                    series_index,
                )
                sum_variable_per_series_per_group[series_index, group_index] = group_sum_y
                sum_squared_variable_per_series_per_group[series_index, group_index] = group_sum_yy
                sum_product_per_series_per_group[series_index, group_index] = group_sum_xy
            end
        end

        return GroupedSeriesCorrelations{F}(
            n_points,
            n_series,
            n_groups,
            first_point_index_per_group,
            Matrix{F}(fixed_per_point_per_series),
            is_active,
            n_active_per_group,
            sum_fixed_per_series,
            sum_squared_fixed_per_series,
            sum_variable_per_series,
            sum_squared_variable_per_series,
            sum_product_per_series,
            sum_variable_per_series_per_group,
            sum_squared_variable_per_series_per_group,
            sum_product_per_series_per_group,
        )
    end
end

# Pearson correlation reconstructed from the sums and the effective active-point count. Returns 0 for an undefined case
# (fewer than 2 active points or zero variance), matching `zero_cor_between_vectors`.
function correlation_from_sums(
    sum_fixed::Float64,
    sum_squared_fixed::Float64,
    sum_variable::Float64,
    sum_squared_variable::Float64,
    sum_product::Float64,
    n_active::Int,
)::Float64
    if n_active < 2
        return 0.0
    end
    covariance = n_active * sum_product - sum_fixed * sum_variable
    variance_fixed = n_active * sum_squared_fixed - sum_fixed * sum_fixed
    variance_variable = n_active * sum_squared_variable - sum_variable * sum_variable
    if variance_fixed <= 0 || variance_variable <= 0
        return 0.0
    end
    return covariance / sqrt(variance_fixed * variance_variable)
end

# Sums (Σy, Σy², Σx·y) over a group's points for one series, with each point's contribution multiplied by the mask
# (inactive points contribute 0). Accumulated in Float32, no allocation.
@inline function masked_group_sums(
    fixed_per_point_per_series::AbstractMatrix{<:Real},
    fixed_first_point::Integer,
    variable_per_point_per_series::AbstractMatrix{<:Real},
    variable_first_point::Integer,
    is_active_per_point::AbstractVector{Bool},
    active_first_point::Integer,
    n_group_points::Integer,
    series_index::Integer,
)::NTuple{3, Float32}
    sum_variable = 0.0f0
    sum_squared_variable = 0.0f0
    sum_product = 0.0f0
    @turbo for point_offset in 0:(n_group_points - 1)
        contribution = Float32(is_active_per_point[active_first_point + point_offset])
        fixed_value = fixed_per_point_per_series[fixed_first_point + point_offset, series_index]
        variable_value = variable_per_point_per_series[variable_first_point + point_offset, series_index]
        sum_variable += contribution * variable_value
        sum_squared_variable += contribution * variable_value * variable_value
        sum_product += contribution * fixed_value * variable_value
    end
    return (sum_variable, sum_squared_variable, sum_product)
end

# Five sums for replacing a group's variable values AND its mask in one shot: the fixed-side delta `(Δsum_x, Δsum_xx)`
# coming from points whose activity flipped, plus the new variable-side sums `(sum_y, sum_yy, sum_xy)` over the points
# that are active under the new mask. Mask transitions are encoded branchlessly as `transition = new - old`: positive
# for inactive→active, negative for active→inactive, zero where the mask didn't move.
@inline function masked_replace_sums(
    fixed_per_point_per_series::AbstractMatrix{<:Real},
    fixed_first_point::Integer,
    variable_per_point_per_series::AbstractMatrix{<:Real},
    variable_first_point::Integer,
    is_active_old_per_point::AbstractVector{Bool},
    active_old_first_point::Integer,
    is_active_new_per_point::AbstractVector{Bool},
    active_new_first_point::Integer,
    n_group_points::Integer,
    series_index::Integer,
)::NTuple{5, Float32}
    delta_sum_fixed = 0.0f0
    delta_sum_squared_fixed = 0.0f0
    sum_variable = 0.0f0
    sum_squared_variable = 0.0f0
    sum_product = 0.0f0
    @turbo for point_offset in 0:(n_group_points - 1)
        was_active = Float32(is_active_old_per_point[active_old_first_point + point_offset])
        is_active_new = Float32(is_active_new_per_point[active_new_first_point + point_offset])
        transition = is_active_new - was_active
        fixed_value = fixed_per_point_per_series[fixed_first_point + point_offset, series_index]
        variable_value = variable_per_point_per_series[variable_first_point + point_offset, series_index]
        delta_sum_fixed += transition * fixed_value
        delta_sum_squared_fixed += transition * fixed_value * fixed_value
        sum_variable += is_active_new * variable_value
        sum_squared_variable += is_active_new * variable_value * variable_value
        sum_product += is_active_new * fixed_value * variable_value
    end
    return (delta_sum_fixed, delta_sum_squared_fixed, sum_variable, sum_squared_variable, sum_product)
end

# `masked_replace_sums` variant for callers that already have the new variable values + activity stored in a SHARED
# upstream matrix / vector indexed by some `source_index` axis (e.g. cell-position within a block), and a per-group
# point ordering `source_index_per_point` mapping the group's canonical layout to that shared axis. Lets the caller
# skip materializing a per-group `(n_group_points × n_series)` scratch matrix - the gather happens inline inside the
# @turbo loop via `source_index_per_point[…]`.
@inline function masked_replace_sums_indirect(
    fixed_per_point_per_series::AbstractMatrix{<:Real},
    fixed_first_point::Integer,
    variable_per_source_per_series::AbstractMatrix{<:Real},
    series_position::Integer,
    source_index_per_point::AbstractVector{<:Integer},
    source_first_point::Integer,
    is_active_new_per_source::AbstractVector{Bool},
    is_active_old_per_point::AbstractVector{Bool},
    active_old_first_point::Integer,
    n_group_points::Integer,
    series_index::Integer,
)::NTuple{5, Float32}
    delta_sum_fixed = 0.0f0
    delta_sum_squared_fixed = 0.0f0
    sum_variable = 0.0f0
    sum_squared_variable = 0.0f0
    sum_product = 0.0f0
    @turbo for point_offset in 0:(n_group_points - 1)
        was_active = Float32(is_active_old_per_point[active_old_first_point + point_offset])
        source_index = source_index_per_point[source_first_point + point_offset]
        is_active_new = Float32(is_active_new_per_source[source_index])
        transition = is_active_new - was_active
        fixed_value = fixed_per_point_per_series[fixed_first_point + point_offset, series_index]
        variable_value = variable_per_source_per_series[source_index, series_position]
        delta_sum_fixed += transition * fixed_value
        delta_sum_squared_fixed += transition * fixed_value * fixed_value
        sum_variable += is_active_new * variable_value
        sum_squared_variable += is_active_new * variable_value * variable_value
        sum_product += is_active_new * fixed_value * variable_value
    end
    return (delta_sum_fixed, delta_sum_squared_fixed, sum_variable, sum_squared_variable, sum_product)
end

# Sum of per-group active counts.
@inline function total_active_points(grouped::GroupedSeriesCorrelations)::Int
    total = 0
    @inbounds for group_index in 1:grouped.n_groups
        total += grouped.n_active_per_group[group_index]
    end
    return total
end

# Active count in a slice of a per-point mask.
@inline function count_active_in_group(
    is_active_per_point::AbstractVector{Bool},
    first_point::Integer,
    n_group_points::Integer,
)::Int
    n_active = 0
    @inbounds for point_offset in 0:(n_group_points - 1)
        n_active += is_active_per_point[first_point + point_offset]
    end
    return n_active
end

"""
    correlation_per_series!(
        grouped::GroupedSeriesCorrelations,
        correlation_per_series::AbstractVector{<:AbstractFloat},
    )::AbstractVector{<:AbstractFloat}

Fill `correlation_per_series` with the current per-series Pearson correlation between the fixed and variable values
(over the active points), and return it. Undefined cases (zero variance, or fewer than 2 active points) get 0.
"""
function correlation_per_series!(
    grouped::GroupedSeriesCorrelations,
    correlation_per_series::AbstractVector{<:AbstractFloat},
)::AbstractVector{<:AbstractFloat}
    @assert_vector(correlation_per_series, grouped.n_series)
    n_active = total_active_points(grouped)
    @inbounds for series_index in 1:grouped.n_series
        correlation_per_series[series_index] = correlation_from_sums(
            grouped.sum_fixed_per_series[series_index],
            grouped.sum_squared_fixed_per_series[series_index],
            grouped.sum_variable_per_series[series_index],
            grouped.sum_squared_variable_per_series[series_index],
            grouped.sum_product_per_series[series_index],
            n_active,
        )
    end
    return correlation_per_series
end

"""
    mean_correlation(grouped::GroupedSeriesCorrelations)::Float64

The mean over all series of the current per-series correlation (undefined cases counted as 0). Allocation-free.
"""
function mean_correlation(grouped::GroupedSeriesCorrelations)::Float64
    n_active = total_active_points(grouped)
    total_correlation = 0.0
    @inbounds for series_index in 1:grouped.n_series
        total_correlation += correlation_from_sums(
            grouped.sum_fixed_per_series[series_index],
            grouped.sum_squared_fixed_per_series[series_index],
            grouped.sum_variable_per_series[series_index],
            grouped.sum_squared_variable_per_series[series_index],
            grouped.sum_product_per_series[series_index],
            n_active,
        )
    end
    return total_correlation / grouped.n_series
end

"""
    mean_correlation_if_group_replaced(
        grouped::GroupedSeriesCorrelations,
        group_index::Integer,
        variable_per_point_per_series::AbstractMatrix{<:Real},
        [is_active_per_group_point::AbstractVector{Bool}],
    )::Float64

The mean per-series correlation that would result if group `group_index`'s variable values were replaced by
`variable_per_point_per_series` (a `group_size × n_series` matrix in the group's point order), without committing the
change. The values matrix is indexed by all `group_size` points (entries for inactive points are still passed - they
are multiplied out by the mask). If `is_active_per_group_point` is supplied (a `group_size`-vector of `Bool`), the
group's mask is also tentatively replaced and the fixed-side totals adjusted for the activity transitions; otherwise
the current mask is kept. Allocation-free and side-effect-free, so it is safe to call concurrently on a shared
`grouped`.
"""
function mean_correlation_if_group_replaced(
    grouped::GroupedSeriesCorrelations,
    group_index::Integer,
    variable_per_point_per_series::AbstractMatrix{<:Real},
)::Float64
    group_size = grouped.first_point_index_per_group[group_index + 1] - grouped.first_point_index_per_group[group_index]
    @assert_matrix(variable_per_point_per_series, group_size, grouped.n_series)
    @check_turbo_matrix(variable_per_point_per_series)
    first_point_index = grouped.first_point_index_per_group[group_index]
    n_active = total_active_points(grouped)
    total_correlation = 0.0
    @inbounds for series_index in 1:grouped.n_series
        (group_sum_y, group_sum_yy, group_sum_xy) = masked_group_sums(
            grouped.fixed_per_point_per_series,
            first_point_index,
            variable_per_point_per_series,
            1,
            grouped.is_active_per_point,
            first_point_index,
            group_size,
            series_index,
        )
        candidate_sum_variable =
            grouped.sum_variable_per_series[series_index] -
            grouped.sum_variable_per_series_per_group[series_index, group_index] + group_sum_y
        candidate_sum_squared_variable =
            grouped.sum_squared_variable_per_series[series_index] -
            grouped.sum_squared_variable_per_series_per_group[series_index, group_index] + group_sum_yy
        candidate_sum_product =
            grouped.sum_product_per_series[series_index] -
            grouped.sum_product_per_series_per_group[series_index, group_index] + group_sum_xy
        total_correlation += correlation_from_sums(
            grouped.sum_fixed_per_series[series_index],
            grouped.sum_squared_fixed_per_series[series_index],
            candidate_sum_variable,
            candidate_sum_squared_variable,
            candidate_sum_product,
            n_active,
        )
    end
    return total_correlation / grouped.n_series
end

function mean_correlation_if_group_replaced(
    grouped::GroupedSeriesCorrelations,
    group_index::Integer,
    variable_per_point_per_series::AbstractMatrix{<:Real},
    is_active_per_group_point::AbstractVector{Bool},
)::Float64
    group_size = grouped.first_point_index_per_group[group_index + 1] - grouped.first_point_index_per_group[group_index]
    @assert_matrix(variable_per_point_per_series, group_size, grouped.n_series)
    @assert_vector(is_active_per_group_point, group_size)
    @check_turbo_matrix(variable_per_point_per_series)
    @check_turbo_vector(is_active_per_group_point)
    first_point_index = grouped.first_point_index_per_group[group_index]
    new_n_active_in_group = count_active_in_group(is_active_per_group_point, 1, group_size)
    candidate_n_active = total_active_points(grouped) - grouped.n_active_per_group[group_index] + new_n_active_in_group
    total_correlation = 0.0
    @inbounds for series_index in 1:grouped.n_series
        (delta_sum_fixed, delta_sum_squared_fixed, group_sum_y, group_sum_yy, group_sum_xy) = masked_replace_sums(
            grouped.fixed_per_point_per_series,
            first_point_index,
            variable_per_point_per_series,
            1,
            grouped.is_active_per_point,
            first_point_index,
            is_active_per_group_point,
            1,
            group_size,
            series_index,
        )
        candidate_sum_fixed = grouped.sum_fixed_per_series[series_index] + delta_sum_fixed
        candidate_sum_squared_fixed = grouped.sum_squared_fixed_per_series[series_index] + delta_sum_squared_fixed
        candidate_sum_variable =
            grouped.sum_variable_per_series[series_index] -
            grouped.sum_variable_per_series_per_group[series_index, group_index] + group_sum_y
        candidate_sum_squared_variable =
            grouped.sum_squared_variable_per_series[series_index] -
            grouped.sum_squared_variable_per_series_per_group[series_index, group_index] + group_sum_yy
        candidate_sum_product =
            grouped.sum_product_per_series[series_index] -
            grouped.sum_product_per_series_per_group[series_index, group_index] + group_sum_xy
        total_correlation += correlation_from_sums(
            candidate_sum_fixed,
            candidate_sum_squared_fixed,
            candidate_sum_variable,
            candidate_sum_squared_variable,
            candidate_sum_product,
            candidate_n_active,
        )
    end
    return total_correlation / grouped.n_series
end

"""
    mean_correlation_if_group_replaced(
        grouped::GroupedSeriesCorrelations,
        group_index::Integer,
        variable_per_source_per_series_position::AbstractMatrix{<:Real},
        is_active_per_source::AbstractVector{Bool},
        source_index_per_point::AbstractVector{<:Integer},
        series_position_per_series::AbstractVector{<:Integer},
    )::Float64

Indirect-gather variant: like the 4-arg `mean_correlation_if_group_replaced` with a mask, but the new variable values
and activity flags live in `variable_per_source_per_series_position` and `is_active_per_source` indexed by an upstream
"source" axis (e.g. cell-position in a calling block) rather than in a freshly materialized per-group layout.
`source_index_per_point[p]` gives the source index for point `p` in the group's canonical layout;
`series_position_per_series[s]` gives the column position in `variable_per_source_per_series_position` for the group's
series `s`. Lets the caller skip materializing a `(group_size × n_series)` scratch matrix - the gather happens inside
`@turbo` via the indirection vectors. Allocation-free. Like the direct variant, does not mutate `grouped`.
"""
function mean_correlation_if_group_replaced(
    grouped::GroupedSeriesCorrelations,
    group_index::Integer,
    variable_per_source_per_series_position::AbstractMatrix{<:Real},
    is_active_per_source::AbstractVector{Bool},
    source_index_per_point::AbstractVector{<:Integer},
    series_position_per_series::AbstractVector{<:Integer},
)::Float64
    group_size = grouped.first_point_index_per_group[group_index + 1] - grouped.first_point_index_per_group[group_index]
    @assert_vector(source_index_per_point, group_size)
    @assert_vector(series_position_per_series, grouped.n_series)
    @check_turbo_matrix(variable_per_source_per_series_position)
    @check_turbo_vector(is_active_per_source)
    @check_turbo_vector(source_index_per_point)
    first_point_index = grouped.first_point_index_per_group[group_index]
    new_n_active_in_group = 0
    @inbounds for point_offset in 0:(group_size - 1)
        if is_active_per_source[source_index_per_point[1 + point_offset]]
            new_n_active_in_group += 1
        end
    end
    candidate_n_active = total_active_points(grouped) - grouped.n_active_per_group[group_index] + new_n_active_in_group
    total_correlation = 0.0
    @inbounds for series_index in 1:grouped.n_series
        series_position = series_position_per_series[series_index]
        (delta_sum_fixed, delta_sum_squared_fixed, group_sum_y, group_sum_yy, group_sum_xy) =
            masked_replace_sums_indirect(
                grouped.fixed_per_point_per_series,
                first_point_index,
                variable_per_source_per_series_position,
                series_position,
                source_index_per_point,
                1,
                is_active_per_source,
                grouped.is_active_per_point,
                first_point_index,
                group_size,
                series_index,
            )
        candidate_sum_fixed = grouped.sum_fixed_per_series[series_index] + delta_sum_fixed
        candidate_sum_squared_fixed = grouped.sum_squared_fixed_per_series[series_index] + delta_sum_squared_fixed
        candidate_sum_variable =
            grouped.sum_variable_per_series[series_index] -
            grouped.sum_variable_per_series_per_group[series_index, group_index] + group_sum_y
        candidate_sum_squared_variable =
            grouped.sum_squared_variable_per_series[series_index] -
            grouped.sum_squared_variable_per_series_per_group[series_index, group_index] + group_sum_yy
        candidate_sum_product =
            grouped.sum_product_per_series[series_index] -
            grouped.sum_product_per_series_per_group[series_index, group_index] + group_sum_xy
        total_correlation += correlation_from_sums(
            candidate_sum_fixed,
            candidate_sum_squared_fixed,
            candidate_sum_variable,
            candidate_sum_squared_variable,
            candidate_sum_product,
            candidate_n_active,
        )
    end
    return total_correlation / grouped.n_series
end

"""
    replace_group!(
        grouped::GroupedSeriesCorrelations,
        group_index::Integer,
        variable_per_point_per_series::AbstractMatrix{<:Real},
        [is_active_per_group_point::AbstractVector{Bool}],
    )::Nothing

Commit new variable values for group `group_index` (a `group_size × n_series` matrix in the group's point order),
updating the kept sums in place. If `is_active_per_group_point` is supplied (a `group_size`-vector of `Bool`), the
group's mask is also updated and the fixed-side totals are adjusted for the activity transitions; otherwise the current
mask is kept. Does not compute correlations - call [`mean_correlation`](@ref) or [`correlation_per_series!`](@ref) once
after committing all the changed groups. Allocation-free; not safe to call concurrently with itself or with queries on
the same `grouped`.
"""
function replace_group!(
    grouped::GroupedSeriesCorrelations,
    group_index::Integer,
    variable_per_point_per_series::AbstractMatrix{<:Real},
)::Nothing
    group_size = grouped.first_point_index_per_group[group_index + 1] - grouped.first_point_index_per_group[group_index]
    @assert_matrix(variable_per_point_per_series, group_size, grouped.n_series)
    @check_turbo_matrix(variable_per_point_per_series)
    first_point_index = grouped.first_point_index_per_group[group_index]
    @inbounds for series_index in 1:grouped.n_series
        (group_sum_y, group_sum_yy, group_sum_xy) = masked_group_sums(
            grouped.fixed_per_point_per_series,
            first_point_index,
            variable_per_point_per_series,
            1,
            grouped.is_active_per_point,
            first_point_index,
            group_size,
            series_index,
        )
        grouped.sum_variable_per_series[series_index] +=
            group_sum_y - grouped.sum_variable_per_series_per_group[series_index, group_index]
        grouped.sum_squared_variable_per_series[series_index] +=
            group_sum_yy - grouped.sum_squared_variable_per_series_per_group[series_index, group_index]
        grouped.sum_product_per_series[series_index] +=
            group_sum_xy - grouped.sum_product_per_series_per_group[series_index, group_index]
        grouped.sum_variable_per_series_per_group[series_index, group_index] = group_sum_y
        grouped.sum_squared_variable_per_series_per_group[series_index, group_index] = group_sum_yy
        grouped.sum_product_per_series_per_group[series_index, group_index] = group_sum_xy
    end
    return nothing
end

function replace_group!(
    grouped::GroupedSeriesCorrelations,
    group_index::Integer,
    variable_per_point_per_series::AbstractMatrix{<:Real},
    is_active_per_group_point::AbstractVector{Bool},
)::Nothing
    group_size = grouped.first_point_index_per_group[group_index + 1] - grouped.first_point_index_per_group[group_index]
    @assert_matrix(variable_per_point_per_series, group_size, grouped.n_series)
    @assert_vector(is_active_per_group_point, group_size)
    @check_turbo_matrix(variable_per_point_per_series)
    @check_turbo_vector(is_active_per_group_point)
    first_point_index = grouped.first_point_index_per_group[group_index]
    new_n_active_in_group = count_active_in_group(is_active_per_group_point, 1, group_size)
    @inbounds for series_index in 1:grouped.n_series
        (delta_sum_fixed, delta_sum_squared_fixed, group_sum_y, group_sum_yy, group_sum_xy) = masked_replace_sums(
            grouped.fixed_per_point_per_series,
            first_point_index,
            variable_per_point_per_series,
            1,
            grouped.is_active_per_point,
            first_point_index,
            is_active_per_group_point,
            1,
            group_size,
            series_index,
        )
        grouped.sum_fixed_per_series[series_index] += delta_sum_fixed
        grouped.sum_squared_fixed_per_series[series_index] += delta_sum_squared_fixed
        grouped.sum_variable_per_series[series_index] +=
            group_sum_y - grouped.sum_variable_per_series_per_group[series_index, group_index]
        grouped.sum_squared_variable_per_series[series_index] +=
            group_sum_yy - grouped.sum_squared_variable_per_series_per_group[series_index, group_index]
        grouped.sum_product_per_series[series_index] +=
            group_sum_xy - grouped.sum_product_per_series_per_group[series_index, group_index]
        grouped.sum_variable_per_series_per_group[series_index, group_index] = group_sum_y
        grouped.sum_squared_variable_per_series_per_group[series_index, group_index] = group_sum_yy
        grouped.sum_product_per_series_per_group[series_index, group_index] = group_sum_xy
    end
    # Apply the new mask and update the active count for this group.
    @inbounds for point_offset in 0:(group_size - 1)
        grouped.is_active_per_point[first_point_index + point_offset] = is_active_per_group_point[point_offset + 1]
    end
    grouped.n_active_per_group[group_index] = new_n_active_in_group
    return nothing
end

"""
    replace_group!(
        grouped::GroupedSeriesCorrelations,
        group_index::Integer,
        variable_per_source_per_series_position::AbstractMatrix{<:Real},
        is_active_per_source::AbstractVector{Bool},
        source_index_per_point::AbstractVector{<:Integer},
        series_position_per_series::AbstractVector{<:Integer},
    )::Nothing

Indirect-gather variant of the 5-arg `replace_group!` with a mask: the new variable values and activity flags live in
`variable_per_source_per_series_position` and `is_active_per_source` indexed by an upstream source axis (e.g. cell-
position in a calling block) rather than in a freshly materialized per-group layout. `source_index_per_point[p]` gives
the source index for point `p` in the group's canonical layout; `series_position_per_series[s]` gives the column
position in `variable_per_source_per_series_position` for the group's series `s`. Lets the caller skip materializing a
`(group_size × n_series)` scratch matrix - the gather happens inside `@turbo` via the indirection vectors. Same
mutation contract as the direct variant.
"""
function replace_group!(
    grouped::GroupedSeriesCorrelations,
    group_index::Integer,
    variable_per_source_per_series_position::AbstractMatrix{<:Real},
    is_active_per_source::AbstractVector{Bool},
    source_index_per_point::AbstractVector{<:Integer},
    series_position_per_series::AbstractVector{<:Integer},
)::Nothing
    group_size = grouped.first_point_index_per_group[group_index + 1] - grouped.first_point_index_per_group[group_index]
    @assert_vector(source_index_per_point, group_size)
    @assert_vector(series_position_per_series, grouped.n_series)
    @check_turbo_matrix(variable_per_source_per_series_position)
    @check_turbo_vector(is_active_per_source)
    @check_turbo_vector(source_index_per_point)
    first_point_index = grouped.first_point_index_per_group[group_index]
    new_n_active_in_group = 0
    @inbounds for point_offset in 0:(group_size - 1)
        if is_active_per_source[source_index_per_point[1 + point_offset]]
            new_n_active_in_group += 1
        end
    end
    @inbounds for series_index in 1:grouped.n_series
        series_position = series_position_per_series[series_index]
        (delta_sum_fixed, delta_sum_squared_fixed, group_sum_y, group_sum_yy, group_sum_xy) =
            masked_replace_sums_indirect(
                grouped.fixed_per_point_per_series,
                first_point_index,
                variable_per_source_per_series_position,
                series_position,
                source_index_per_point,
                1,
                is_active_per_source,
                grouped.is_active_per_point,
                first_point_index,
                group_size,
                series_index,
            )
        grouped.sum_fixed_per_series[series_index] += delta_sum_fixed
        grouped.sum_squared_fixed_per_series[series_index] += delta_sum_squared_fixed
        grouped.sum_variable_per_series[series_index] +=
            group_sum_y - grouped.sum_variable_per_series_per_group[series_index, group_index]
        grouped.sum_squared_variable_per_series[series_index] +=
            group_sum_yy - grouped.sum_squared_variable_per_series_per_group[series_index, group_index]
        grouped.sum_product_per_series[series_index] +=
            group_sum_xy - grouped.sum_product_per_series_per_group[series_index, group_index]
        grouped.sum_variable_per_series_per_group[series_index, group_index] = group_sum_y
        grouped.sum_squared_variable_per_series_per_group[series_index, group_index] = group_sum_yy
        grouped.sum_product_per_series_per_group[series_index, group_index] = group_sum_xy
    end
    # Apply the new mask and update the active count for this group via the same indirection.
    @inbounds for point_offset in 0:(group_size - 1)
        grouped.is_active_per_point[first_point_index + point_offset] =
            is_active_per_source[source_index_per_point[1 + point_offset]]
    end
    grouped.n_active_per_group[group_index] = new_n_active_in_group
    return nothing
end

end
