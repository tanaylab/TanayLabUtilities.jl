"""
Parallel versions of functions from `Distances.jl`.
"""
module ParallelDistances

export parallel_pairwise
export parallel_pairwise_closest
export parallel_colwise

using Base.Threads
using Distances
using ProgressMeter

using ..FlameTime
using ..MatrixLayouts
using ..ParallelLoops
using ..Types

"""
    parallel_pairwise(
        distance, X[, Y];
        dims::Integer,
        policy::Symbol = :greedy,
        progress::Maybe{Progress} = nothing
        progress_chunk::Maybe{Integer} = nothing,
    )::AbstractMatrix

A parallel version of `pairwise`. This will use [`parallel_loop_wo_rng`](@ref) over the columns of `Y`, with the
specified `policy` and `progress`. If `policy` is `:serial`, then the standard version of `pairwise` is called and
`progress` is ignored.

```jldoctest
using Distances
using Random

Random.seed!(123456)

m1 = rand(10, 20)
m2 = rand(10, 30)

d = pairwise(Euclidean(), m1; dims = 2)
@assert maximum(abs.(d .- parallel_pairwise(Euclidean(), m1; dims = 2, policy = :serial))) < 1e-6
@assert maximum(abs.(d .- parallel_pairwise(Euclidean(), m1; dims = 2, policy = :greedy))) < 1e-6
@assert maximum(abs.(d .- parallel_pairwise(Euclidean(), m1; dims = 2, policy = :dynamic))) < 1e-6
@assert maximum(abs.(d .- parallel_pairwise(Euclidean(), m1; dims = 2, policy = :static))) < 1e-6

d = pairwise(Euclidean(), m1, m2; dims = 2)
@assert maximum(abs.(d .- parallel_pairwise(Euclidean(), m1, m2; dims = 2, policy = :serial))) < 1e-6
@assert maximum(abs.(d .- parallel_pairwise(Euclidean(), m1, m2; dims = 2, policy = :greedy))) < 1e-6
@assert maximum(abs.(d .- parallel_pairwise(Euclidean(), m1, m2; dims = 2, policy = :dynamic))) < 1e-6
@assert maximum(abs.(d .- parallel_pairwise(Euclidean(), m1, m2; dims = 2, policy = :static))) < 1e-6

# output

```
"""
function parallel_pairwise(
    distance,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    dims::Integer,
    policy::Symbol = :greedy,
    progress::Maybe{Progress} = nothing,
    progress_chunk::Maybe{Integer} = nothing,
)::AbstractMatrix
    @assert dims in (1, 2)
    if dims == 1
        return flipped(parallel_pairwise(distance, flipped(X), flipped(Y); dims = 2, policy, progress, progress_chunk))  # UNTESTED
    end

    @assert policy in (:serial, :greedy, :dynamic, :static)
    if policy == :serial
        return flame_timed("pairwise." * string(nameof(typeof(distance)))) do
            return pairwise(distance, X, Y; dims)
        end
    else
        _, n_X_columns = size(X)
        _, n_Y_columns = size(Y)
        @views X_column = X[:, 1]
        @views Y_column = Y[:, 1]
        first_distance = evaluate(distance, X_column, Y_column)
        result = Matrix{typeof(first_distance)}(undef, n_X_columns, n_Y_columns)
        parallel_loop_wo_rng(
            1:n_Y_columns;
            name = "pairwise." * string(nameof(typeof(distance))),
            policy,
            progress,
        ) do Y_column_index
            @views Y_column_view = Y[:, Y_column_index]
            result[:, Y_column_index] = colwise(distance, X, Y_column_view)
            return nothing
        end
    end

    return result
end

function parallel_pairwise(
    distance,
    X::AbstractMatrix;
    dims::Integer,
    policy::Symbol = :greedy,
    progress::Maybe{Progress} = nothing,
    progress_chunk::Maybe{Integer} = nothing,
)::AbstractMatrix
    @assert dims in (1, 2)
    if dims == 1
        return flipped(parallel_pairwise(distance, flipped(X); dims = 2, policy))  # UNTESTED
    end

    @assert policy in (:serial, :greedy, :dynamic, :static)
    if policy == :serial
        return flame_timed("pairwise." * string(nameof(typeof(distance)))) do
            return pairwise(distance, X; dims)
        end
    else
        _, n_columns = size(X)
        @views column = X[:, 1]
        first_distance = evaluate(distance, column, column)
        result = Matrix{typeof(first_distance)}(undef, n_columns, n_columns)
        parallel_loop_wo_rng(
            1:n_columns;
            name = "pairwise." * string(nameof(typeof(distance))),
            policy,
            progress,
            progress_chunk,
        ) do column_index
            @views column_view = X[:, column_index]
            @views columns_view = X[:, column_index:n_columns]
            result[column_index, column_index:n_columns] =
                result[column_index:n_columns, column_index] = colwise(distance, column_view, columns_view)
            return nothing
        end
    end

    return result
end

"""
    parallel_pairwise_closest(
        distance, X, Y;
        dims::Integer,
        closest_index::Maybe{AbstractVector{<:Integer}} = nothing,
        closest_distance::Maybe{AbstractVector} = nothing,
        policy::Symbol = :greedy,
        progress::Maybe{Progress} = nothing,
        progress_chunk::Maybe{Integer} = nothing,
    )::Nothing

For each entry of `Y`, find the closest entry of `X`. This is equivalent to computing [`parallel_pairwise`](@ref) and
reducing each column to its minimal value and its location, except that the full distances matrix is never stored. The
distances are computed and reduced on the fly using [`parallel_loop_wo_rng`](@ref) over the entries of `Y`, with the
specified `policy` and `progress`.

The results are written into the pre-allocated `closest_index` and/or `closest_distance` vectors (whose length must be
the number of entries of `Y`); at least one of them must be specified. This performs no allocations of its own.

```jldoctest
using Distances
using Random

Random.seed!(123456)

m1 = rand(10, 20)
m2 = rand(10, 30)

d = pairwise(Euclidean(), m1, m2; dims = 2)

closest_index = Vector{Int}(undef, 30)
closest_distance = Vector{Float64}(undef, 30)
parallel_pairwise_closest(Euclidean(), m1, m2; dims = 2, closest_index, closest_distance)

@assert closest_index == [argmin(d[:, column]) for column in 1:30]
@assert maximum(abs.(closest_distance .- [minimum(d[:, column]) for column in 1:30])) < 1e-6

# output

```
"""
function parallel_pairwise_closest(
    distance,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    dims::Integer,
    closest_index::Maybe{AbstractVector{<:Integer}} = nothing,
    closest_distance::Maybe{AbstractVector} = nothing,
    policy::Symbol = :greedy,
    progress::Maybe{Progress} = nothing,
    progress_chunk::Maybe{Integer} = nothing,
)::Nothing
    @assert dims in (1, 2)
    if dims == 1
        parallel_pairwise_closest(  # UNTESTED
            distance,
            flipped(X),
            flipped(Y);
            dims = 2,
            closest_index,
            closest_distance,
            policy,
            progress,
            progress_chunk,
        )
        return nothing  # UNTESTED
    end

    @assert closest_index !== nothing || closest_distance !== nothing "neither closest_index nor closest_distance given"

    _, n_X_columns = size(X)
    _, n_Y_columns = size(Y)
    if closest_index !== nothing
        @assert length(closest_index) == n_Y_columns
    end
    if closest_distance !== nothing
        @assert length(closest_distance) == n_Y_columns
    end

    parallel_loop_wo_rng(
        1:n_Y_columns;
        name = "pairwise_closest." * string(nameof(typeof(distance))),
        policy,
        progress,
        progress_chunk,
    ) do Y_column_index
        @views Y_column_view = Y[:, Y_column_index]
        best_index = 1
        @views best_distance = evaluate(distance, X[:, 1], Y_column_view)
        for X_column_index in 2:n_X_columns
            @views X_column_view = X[:, X_column_index]
            current_distance = evaluate(distance, X_column_view, Y_column_view)
            if current_distance < best_distance
                best_index = X_column_index
                best_distance = current_distance
            end
        end
        if closest_index !== nothing
            closest_index[Y_column_index] = best_index
        end
        if closest_distance !== nothing
            closest_distance[Y_column_index] = best_distance
        end
        return nothing
    end

    return nothing
end

"""
    parallel_colwise(
        distance, X, Y;
        policy::Symbol = :greedy,
        progress::Maybe{Progress} = nothing,
    )::AbstractVector

A parallel version of `colwise`. This will use [`parallel_loop_wo_rng`](@ref) over the columns of `X` and `Y`, using the
specified `policy` and `progress`. If `policy` is `:serial`, then the standard version of `pairwise` is called and
`progress` is ignored.

```jldoctest
using Distances
using Random

Random.seed!(123456)

m1 = rand(10, 20)
m2 = rand(10, 20)

d = colwise(Euclidean(), m1, m2)
@assert maximum(abs.(d .- parallel_colwise(Euclidean(), m1, m2; policy = :serial))) < 1e-6
@assert maximum(abs.(d .- parallel_colwise(Euclidean(), m1, m2; policy = :greedy))) < 1e-6
@assert maximum(abs.(d .- parallel_colwise(Euclidean(), m1, m2; policy = :dynamic))) < 1e-6
@assert maximum(abs.(d .- parallel_colwise(Euclidean(), m1, m2; policy = :static))) < 1e-6

# output

"""
function parallel_colwise(  # FLAKY TESTED
    distance,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    policy::Symbol = :greedy,
    progress::Maybe{Progress} = nothing,
)::AbstractVector
    @assert policy in (:serial, :greedy, :dynamic, :static);
    if policy == :serial
        return flame_timed("colwise." * string(nameof(typeof(distance)))) do
            return colwise(distance, X, Y)
        end
    else
        n_X_rows, n_X_columns = size(X)
        n_Y_rows, n_Y_columns = size(Y)
        @assert n_X_columns == n_Y_columns;
        @assert n_X_rows == n_Y_rows;
        @views X_column = X[:, 1]
        @views Y_column = Y[:, 1]
        first_distance = evaluate(distance, X_column, Y_column)
        result = Vector{typeof(first_distance)}(undef, n_X_columns)
        parallel_loop_wo_rng(
            1:n_X_columns;
            name = "colwise." * string(nameof(typeof(distance))),
            policy,
            progress,
        ) do column_index
            @views X_column_view = X[:, column_index]
            @views Y_column_view = Y[:, column_index]
            result[column_index] = evaluate(distance, X_column_view, Y_column_view)
            return nothing
        end
        return result
    end
end

end
