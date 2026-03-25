"""
Higher-level K-Means functions.
"""
module KMeans

export KMeansBuffers
export KmeansResultView
export kmeans_in_buffers
export kmeans_in_buffers!
export kmeans_in_rounds

using Clustering
import Clustering.assignments
import Clustering.counts
import Clustering.nclusters
import Clustering.wcounts
using Distances
using LoopVectorization
using Random
using StatsBase

using ..Documentation
using ..FlameTime
using ..MatrixFormats
using ..Types

import Random.default_rng

function _sq_euclidean_colwise!(
    ds::AbstractVector{<:AbstractFloat},
    X::AbstractMatrix{<:Real},
    v::AbstractVector{<:Real},
)::Nothing
    d = length(v)
    n = size(X, 2)
    @turbo for j in 1:n
        s = zero(eltype(ds))
        for l in 1:d
            diff = X[l, j] - v[l]
            s += diff * diff
        end
        ds[j] = s
    end
    return nothing
end

"""
    KMeansBuffers{T}(; n_dims::Integer, max_k::Integer, n_points::Integer)::KMeansBuffers where {T<: AbstractFloat}
    KMeansBuffers(buffers::KMeansBuffers{T}; n_dims::Integer, k::Integer, n_points::Integer) where {T <: AbstractFloat}

Pre-allocate buffers for allocation-free K-Means computation. Create a smaller view sharing the storage with
`KMeansBuffers(buffers; n_dims, k, n_points)`.
"""
struct KMeansBuffers{T <: AbstractFloat}
    centers::AbstractMatrix{T}         # d × k: working copy of centers
    assignments::AbstractVector{Int}   # n: point-to-cluster assignments
    costs::AbstractVector{T}           # n: assignment costs
    counts::AbstractVector{Int}        # k: points per cluster
    wcounts::AbstractVector{Int}       # k: weighted counts (Int for unweighted)
    to_update::AbstractVector{Bool}    # k: which centers need update
    unused::Vector{Int}                # <=k: clusters with no points (dynamic)
    ds::AbstractVector{T}              # n: scratch (repick / seed mincosts)
    tcosts::AbstractVector{T}          # n: scratch (repick / seed tmpcosts)
    seed_indices::AbstractVector{Int}  # k: seed point indices for kmpp
end

function KMeansBuffers{T}(; n_dims::Integer, max_k::Integer, n_points::Integer) where {T <: AbstractFloat}
    return KMeansBuffers{T}(
        Matrix{T}(undef, n_dims, max_k),    # centers
        Vector{Int}(undef, n_points),       # assignments
        Vector{T}(undef, n_points),         # costs
        Vector{Int}(undef, max_k),          # counts
        Vector{Int}(undef, max_k),          # wcounts
        Vector{Bool}(undef, max_k),         # to_update
        sizehint!(Vector{Int}(), max_k),    # unused
        Vector{T}(undef, n_points),         # ds
        Vector{T}(undef, n_points),         # tcosts
        Vector{Int}(undef, max_k),          # seed_indices
    )
end

function KMeansBuffers(
    buffers::KMeansBuffers{T};
    n_dims::Integer,
    k::Integer,
    n_points::Integer,
) where {T <: AbstractFloat}
    return KMeansBuffers{T}(
        @view(buffers.centers[1:n_dims, 1:k]),   # centers
        @view(buffers.assignments[1:n_points]),  # assignments
        @view(buffers.costs[1:n_points]),        # costs
        @view(buffers.counts[1:k]),              # counts
        @view(buffers.wcounts[1:k]),             # wcounts
        @view(buffers.to_update[1:k]),           # to_update
        buffers.unused,                          # unused (shared)
        @view(buffers.ds[1:n_points]),           # ds
        @view(buffers.tcosts[1:n_points]),       # tcosts
        @view(buffers.seed_indices[1:k]),        # seed_indices
    )
end

"""
    KmeansResultView

A view into `KMeansBuffers` that provides the same accessor interface as `Clustering.KmeansResult`
(`assignments`, `counts`, `wcounts`, `nclusters`, `totalcost`) without allocating.
"""
struct KmeansResultView{T <: AbstractFloat}
    centers::AbstractMatrix{T}
    assignments::AbstractVector{Int}
    costs::AbstractVector{T}
    counts::AbstractVector{Int}
    wcounts::AbstractVector{Int}
    totalcost::T
    iterations::Int
    converged::Bool
end

Clustering.assignments(r::KmeansResultView) = r.assignments
Clustering.counts(r::KmeansResultView) = r.counts
Clustering.wcounts(r::KmeansResultView) = r.wcounts
Clustering.nclusters(r::KmeansResultView) = length(r.counts)

"""
    kmeans_in_buffers!(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:AbstractFloat};
        buffers::Maybe{KMeansBuffers} = $(DEFAULT.buffers),
        maxiter::Integer = $(DEFAULT.maxiter),
        tol::Real = $(DEFAULT.tol),
        distance::SemiMetric = $(DEFAULT.distance),
        rng::AbstractRNG = default_rng(),
    )::Union{KmeansResult, KmeansResultView}


Same as `kmeans!`, but if `buffers` are specified, run allocation-free code. Seeding is restricted to `:kmpp`.
Implementation is otherwise identical to `Clustering.kmeans!.
"""
@documented function kmeans_in_buffers!(
    X::AbstractMatrix{<:Real},
    centers::AbstractMatrix{<:AbstractFloat};
    buffers::Maybe{KMeansBuffers} = nothing,
    maxiter::Integer = 100,
    tol::Real = 1.0e-6,
    distance::SemiMetric = SqEuclidean(),
    rng::AbstractRNG = default_rng(),
)::Union{KmeansResult, KmeansResultView}
    if buffers === nothing
        return Clustering.kmeans!(X, centers; maxiter, tol, distance, rng)  # NOJET
    end

    @assert distance isa SqEuclidean "Buffered kmeans only supports SqEuclidean, got: $(typeof(distance))"

    d, n = size(X)
    dc, k = size(centers)
    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    (1 <= k <= n) || throw(ArgumentError("k must be from 1:n (n=$n), k=$k given."))

    if k == n
        copyto!(centers, X)
        buffers.assignments .= 1:k
        fill!(buffers.costs, 0)
        fill!(buffers.counts, 1)
        fill!(buffers.wcounts, 1)
        return KmeansResultView(
            centers,
            buffers.assignments,
            buffers.costs,
            buffers.counts,
            buffers.wcounts,
            zero(eltype(centers)),
            0,
            true,
        )
    end

    if k == 1
        mean!(centers, X)  # NOLINT
    end

    return _kmeans_buffered!(X, centers, buffers, Int(maxiter), Float64(tol), rng)
end

"""
    kmeans_in_buffers(
        X::AbstractMatrix{<:Real},
        k::Integer;
        buffers::Maybe{KMeansBuffers} = $(DEFAULT.buffers),
        maxiter::Integer = $(DEFAULT.maxiter),
        tol::Real = $(DEFAULT.tol),
        distance::SemiMetric = $(DEFAULT.distance),
        rng::AbstractRNG = default_rng(),
    )::Union{KmeansResult, KmeansResultView}

Same as `kmeans`, but if `buffers` are specified, run allocation-free code. Seeding is restricted to `:kmpp`.
Implementation is otherwise identical to `Clustering.kmeans.
"""
@documented function kmeans_in_buffers(
    X::AbstractMatrix{<:Real},
    k::Integer;
    buffers::Maybe{KMeansBuffers} = nothing,
    maxiter::Integer = 100,
    tol::Real = 1.0e-6,
    distance::SemiMetric = SqEuclidean(),
    rng::AbstractRNG = default_rng(),
)::Union{KmeansResult, KmeansResultView}
    if buffers === nothing
        return Clustering.kmeans(X, k; maxiter, tol, distance, rng)
    end

    @assert distance isa SqEuclidean "Buffered kmeans only supports SqEuclidean, got: $(typeof(distance))"

    n = size(X, 2)
    (1 <= k <= n) || throw(ArgumentError("k must be from 1:n (n=$n), k=$k given."))

    _kmpp_seed!(X, k, buffers, rng)
    @views copyto!(buffers.centers[:, 1:k], X[:, buffers.seed_indices[1:k]])

    return kmeans_in_buffers!(X, @view(buffers.centers[:, 1:k]); buffers, maxiter, tol, distance, rng)
end

function _kmpp_seed!(X::AbstractMatrix{<:Real}, k::Integer, buffers::KMeansBuffers, rng::AbstractRNG)::Nothing
    n = size(X, 2)
    iseeds = buffers.seed_indices
    mincosts = buffers.ds
    tmpcosts = buffers.tcosts

    p = rand(rng, 1:n)
    iseeds[1] = p

    if k > 1
        _sq_euclidean_colwise!(mincosts, X, view(X, :, p))
        mincosts[p] = 0

        for j in 2:k
            p = wsample(rng, 1:n, mincosts)
            iseeds[j] = p
            _sq_euclidean_colwise!(tmpcosts, X, view(X, :, p))
            @turbo for i in eachindex(mincosts)
                mincosts[i] = min(mincosts[i], tmpcosts[i])
            end
            mincosts[p] = 0
        end
    end

    return nothing
end

function _kmeans_buffered!(
    X::AbstractMatrix{<:Real},
    centers::AbstractMatrix{<:AbstractFloat},
    buffers::KMeansBuffers,
    maxiter::Int,
    tol::Float64,
    rng::AbstractRNG,
)::Union{KmeansResult, KmeansResultView}
    k = size(centers, 2)

    assignments = buffers.assignments
    costs = buffers.costs
    counts = buffers.counts
    wcounts = buffers.wcounts
    to_update = buffers.to_update
    unused = buffers.unused

    dist_scratch = buffers.tcosts

    _update_assignments!(centers, X, true, assignments, costs, counts, to_update, unused, dist_scratch)
    objv = sum(costs)

    t = 0
    converged = false

    while !converged && t < maxiter
        t += 1

        _update_centers!(X, assignments, to_update, centers, wcounts)

        if !isempty(unused)
            _repick_unused_centers!(centers, unused, X, costs, buffers.ds, buffers.tcosts, rng)
            for i in unused
                to_update[i] = true
            end
        end

        _update_assignments!(centers, X, false, assignments, costs, counts, to_update, unused, dist_scratch)

        prev_objv = objv
        objv = sum(costs)
        objv_change = objv - prev_objv

        if (k == 1) || (abs(objv_change) < tol)
            converged = true
        end
    end

    return KmeansResultView(centers, assignments, costs, counts, wcounts, objv, t, converged)
end

function _update_assignments!(
    centers::AbstractMatrix{<:AbstractFloat},
    X::AbstractMatrix{<:Real},
    is_init::Bool,
    assignments::AbstractVector{Int},
    costs::AbstractVector{<:AbstractFloat},
    counts::AbstractVector{Int},
    to_update::AbstractVector{Bool},
    unused::Vector{Int},
    dist_scratch::AbstractVector{<:AbstractFloat},
)::Nothing
    d = size(X, 1)
    n = size(X, 2)
    k = size(centers, 2)
    T = eltype(costs)

    fill!(counts, 0)
    empty!(unused)
    fill!(to_update, is_init)

    @inbounds for j in 1:n
        # Compute squared distances to all k centers, vectorized across i
        @turbo for i in 1:k
            s = zero(T)
            for l in 1:d
                diff = centers[l, i] - X[l, j]
                s += diff * diff
            end
            dist_scratch[i] = s
        end

        # Scalar argmin over k centers
        min_cost = dist_scratch[1]
        min_a = 1
        for i in 2:k
            c = dist_scratch[i]
            if c < min_cost
                min_cost = c
                min_a = i
            end
        end

        if !is_init
            pa = assignments[j]
            if pa != min_a
                to_update[min_a] = true
                to_update[pa] = true
            end
        end
        assignments[j] = min_a
        costs[j] = min_cost
        counts[min_a] += 1
    end

    for i in 1:k
        if counts[i] == 0
            push!(unused, i)
            to_update[i] = false
        end
    end

    return nothing
end

function _update_centers!(
    X::AbstractMatrix{<:Real},
    assignments::AbstractVector{Int},
    to_update::AbstractVector{Bool},
    centers::AbstractMatrix{<:AbstractFloat},
    wcounts::AbstractVector{Int},
)::Nothing
    d, n = size(X)
    k = size(centers, 2)

    @inbounds for j in 1:k
        if to_update[j]
            wcounts[j] = 0
            for i in 1:d
                centers[i, j] = zero(eltype(centers))
            end
        end
    end

    @inbounds for j in 1:n
        cj = assignments[j]
        if to_update[cj]
            for i in 1:d
                centers[i, cj] += X[i, j]
            end
            wcounts[cj] += 1
        end
    end

    @inbounds for j in 1:k
        if to_update[j]
            cj = wcounts[j]
            inv_cj = one(eltype(centers)) / cj
            for i in 1:d
                centers[i, j] *= inv_cj
            end
        end
    end

    return nothing
end

function _repick_unused_centers!(
    centers::AbstractMatrix{<:AbstractFloat},
    unused::Vector{Int},
    X::AbstractMatrix{<:Real},
    costs::AbstractVector{<:Real},
    ds::AbstractVector{<:Real},
    sampling_weights::AbstractVector{<:Real},
    rng::AbstractRNG,
)::Nothing
    n = size(X, 2)
    copyto!(sampling_weights, costs)

    for i in unused
        j = wsample(rng, 1:n, sampling_weights)
        centers[:, i] .= view(X, :, j)
        _sq_euclidean_colwise!(ds, X, view(X, :, j))
        ds[j] = 0
        @turbo for l in eachindex(sampling_weights)  # NOJET
            sampling_weights[l] = min(sampling_weights[l], ds[l])
        end
    end

    return nothing
end

"""
    kmeans_in_rounds(
        values_of_points::AbstractMatrix{<:AbstractFloat},
        k::Integer;
        centers::Maybe{AbstractMatrix{<:AbstractFloat}} = $(DEFAULT.centers),
        buffers::Maybe{Tuple{KMeansBuffers, KMeansBuffers}} = $(DEFAULT.buffers),
        rounds::Integer = $(DEFAULT.rounds),
        rng::AbstractRNG = default_rng(),
    )::Union{KmeansResult, KmeansResultView}

Run `kmeans` multiple times with different random seeds (using `rng`) and return the best results. This is needed
because K-Means is a heuristic and tends to occasionally get stuck in a local minimum.

If `buffers` are specified, runs allocation-free using [`kmeans_in_buffers`](@ref) / [`kmeans_in_buffers!`](@ref).
Otherwise (the default), falls back to using `Clustering.kmeans` / `Clustering.kmeans!`.
"""
@documented function kmeans_in_rounds(
    values_of_points::AbstractMatrix{<:AbstractFloat},
    k::Integer;
    centers::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
    buffers::Maybe{Tuple{KMeansBuffers, KMeansBuffers}} = nothing,
    rounds::Integer = 10,
    rng::AbstractRNG = default_rng(),
)::Union{KmeansResult, KmeansResultView}
    if buffers !== nothing
        n_dims = size(values_of_points, 1)
        n_points = size(values_of_points, 2)
        buffers = (KMeansBuffers(buffers[1]; n_dims, k, n_points), KMeansBuffers(buffers[2]; n_dims, k, n_points))
    end
    best_buf = buffers === nothing ? nothing : buffers[1]
    current_buf = buffers === nothing ? nothing : buffers[2]
    best_cost = typemax(Float64)
    best_kmeans_result = nothing

    for _ in 1:rounds
        if centers === nothing
            kmeans_result = flame_timed("kmeans") do
                return kmeans_in_buffers(values_of_points, k; buffers = current_buf, rng)  # NOJET
            end
        else
            if current_buf !== nothing
                copyto!(@view(current_buf.centers[:, 1:k]), centers)
            end
            kmeans_result = flame_timed("kmeans") do
                return kmeans_in_buffers!(
                    values_of_points,
                    current_buf === nothing ? copy(centers) : @view(current_buf.centers[:, 1:k]);
                    buffers = current_buf,
                    rng,
                )
            end
        end

        if kmeans_result.totalcost < best_cost
            best_cost = kmeans_result.totalcost
            best_kmeans_result = kmeans_result
            if buffers !== nothing
                best_buf, current_buf = current_buf, best_buf
            end
        end
    end

    @assert best_kmeans_result !== nothing
    return best_kmeans_result
end

end
