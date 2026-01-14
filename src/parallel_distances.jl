"""
Parallel versions of functions from `Distances.jl`.
"""
module ParallelDistances

export parallel_pairwise
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
    )::AbstractMatrix

A parallel version of `pairwise`. This will use [`parallel_loop_wo_rng`](@ref) over the columns of `Y`, with the
specified `policy` and `progress`. If `policy` is `:serial`, then the standard version of `pairwise` is called and
`progress` is ignored.

```jldoctest
using Test
using Distances
using Random

Random.seed!(123456)

m1 = rand(10, 20)
m2 = rand(10, 30)

d = pairwise(Euclidean(), m1; dims = 2)
@test maximum(abs.(d .- parallel_pairwise(Euclidean(), m1; dims = 2, policy = :serial))) < 1e-6
@test maximum(abs.(d .- parallel_pairwise(Euclidean(), m1; dims = 2, policy = :greedy))) < 1e-6
@test maximum(abs.(d .- parallel_pairwise(Euclidean(), m1; dims = 2, policy = :dynamic))) < 1e-6
@test maximum(abs.(d .- parallel_pairwise(Euclidean(), m1; dims = 2, policy = :static))) < 1e-6

d = pairwise(Euclidean(), m1, m2; dims = 2)
@test maximum(abs.(d .- parallel_pairwise(Euclidean(), m1, m2; dims = 2, policy = :serial))) < 1e-6
@test maximum(abs.(d .- parallel_pairwise(Euclidean(), m1, m2; dims = 2, policy = :greedy))) < 1e-6
@test maximum(abs.(d .- parallel_pairwise(Euclidean(), m1, m2; dims = 2, policy = :dynamic))) < 1e-6
@test maximum(abs.(d .- parallel_pairwise(Euclidean(), m1, m2; dims = 2, policy = :static))) < 1e-6

println("OK")

# output

OK
```
"""
function parallel_pairwise(
    distance,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    dims::Integer,
    policy::Symbol = :greedy,
    progress::Maybe{Progress} = nothing,
)::AbstractMatrix
    @assert dims in (1, 2)
    if dims == 1
        return flipped(parallel_pairwise(distance, flipped(X), flipped(Y); dims = 2, policy, progress))  # UNTESTED
    end

    @assert policy in (:serial, :greedy, :dynamic, :static)
    if policy == :serial
        return flame_timed("pairwise_" * string(typeof(distance))) do
            return pairwise(distance, X, Y; dims)
        end
    else
        _, n_X_columns = size(X)
        _, n_Y_columns = size(Y)
        @views X_column = X[:, 1]
        @views Y_column = Y[:, 1]
        first_distance = evaluate(distance, X_column, Y_column)
        result = Matrix{typeof(first_distance)}(undef, n_X_columns, n_Y_columns)
        parallel_loop_wo_rng("pairwise_" * string(typeof(distance)), 1:n_Y_columns; policy, progress) do Y_column_index
            @views Y_column = Y[:, Y_column_index]
            result[:, Y_column_index] = colwise(distance, X, Y_column)
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
)::AbstractMatrix
    @assert dims in (1, 2)
    if dims == 1
        return flipped(parallel_pairwise(distance, flipped(X); dims = 2, policy))  # UNTESTED
    end

    @assert policy in (:serial, :greedy, :dynamic, :static)
    if policy == :serial
        return flame_timed("pairwise_" * string(typeof(distance))) do
            return pairwise(distance, X; dims)
        end
    else
        _, n_columns = size(X)
        @views column = X[:, 1]
        first_distance = evaluate(distance, column, column)
        result = Matrix{typeof(first_distance)}(undef, n_columns, n_columns)
        parallel_loop_wo_rng("pairwise_" * string(typeof(distance)), 1:n_columns; policy, progress) do column_index
            @views column = X[:, column_index]
            @views columns = X[:, column_index:n_columns]
            result[column_index, column_index:n_columns] =
                result[column_index:n_columns, column_index] = colwise(distance, column, columns)
            return nothing
        end
    end

    return result
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
using Test
using Distances
using Random

Random.seed!(123456)

m1 = rand(10, 20)
m2 = rand(10, 20)

d = colwise(Euclidean(), m1, m2)
@test maximum(abs.(d .- parallel_colwise(Euclidean(), m1, m2; policy = :serial))) < 1e-6
@test maximum(abs.(d .- parallel_colwise(Euclidean(), m1, m2; policy = :greedy))) < 1e-6
@test maximum(abs.(d .- parallel_colwise(Euclidean(), m1, m2; policy = :dynamic))) < 1e-6
@test maximum(abs.(d .- parallel_colwise(Euclidean(), m1, m2; policy = :static))) < 1e-6

println("OK")

# output

OK
"""
function parallel_colwise(  # FLAKY TESTED
    distance,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    policy::Symbol = :greedy,
    progress::Maybe{Progress} = nothing,
)::AbstractVector
    @assert policy in (:serial, :greedy, :dynamic, :static)
    if policy == :serial
        return flame_timed("colwise_" * string(typeof(distance))) do
            return colwise(distance, X, Y)
        end
    else
        n_X_rows, n_X_columns = size(X)
        n_Y_rows, n_Y_columns = size(Y)
        @assert n_X_columns == n_Y_columns
        @assert n_X_rows == n_Y_rows
        @views X_column = X[:, 1]
        @views Y_column = Y[:, 1]
        first_distance = evaluate(distance, X_column, Y_column)
        result = Vector{typeof(first_distance)}(undef, n_X_columns)
        parallel_loop_wo_rng("colwise_" * string(typeof(distance)), 1:n_X_columns; policy, progress) do column_index
            @views X_column = X[:, column_index]
            @views Y_column = Y[:, column_index]
            return result[column_index] = evaluate(distance, X_column, Y_column)
        end
    end
end

end
