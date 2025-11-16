"""
Higher-level K-Means functions.
"""
module KMeans

export kmeans_in_rounds

using Random
using Clustering

using ..Documentation
using ..MatrixFormats
using ..Types

import Random.default_rng

"""
    kmeans_in_rounds(
        values_of_points::AbstractMatrix{<:AbstractFloat},
        k::Integer;
        centers::Maybe{AbstractMatrix{<:AbstractFloat}} = $(DEFAULT.centers),
        rounds::Integer = $(DEFAULT.rounds),
        rng::AbstractRNG = default_rng(),
    )::KmeansResult

Run `kmeans` multiple times with different random seeds (using `rng`) and return the best results. This is needed
because K-Means is a heuristic and tends to occasionally get stuck in a local minimum.
"""
@documented function kmeans_in_rounds(
    values_of_points::AbstractMatrix{<:AbstractFloat},
    k::Integer;
    centers::Maybe{AbstractMatrix{<:AbstractFloat}} = nothing,
    rounds::Integer = 10,
    rng::AbstractRNG = default_rng(),
)::KmeansResult
    best_kmeans_result = nothing

    for _ in 1:rounds
        if centers === nothing
            kmeans_result = kmeans(values_of_points, k; rng)  # NOJET
        else
            kmeans_result = kmeans!(values_of_points, copy_array(centers); rng)
        end

        if best_kmeans_result === nothing || kmeans_result.totalcost < best_kmeans_result.totalcost
            best_kmeans_result = kmeans_result
        end
    end

    @assert best_kmeans_result !== nothing
    return best_kmeans_result
end

end
