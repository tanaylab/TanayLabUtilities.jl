"""
Enhancements of [hclust]*https://github.com/JuliaStats/Clustering.jl/blob/master/src/hclust.jl)
"""
module EnhancedHclust

export ehclust

using Clustering
using LinearAlgebra

import Clustering.assertdistancematrix
import Clustering.AverageDistance
import Clustering.hclust_minimum
import Clustering.hclust_nn_lw
import Clustering.HclustMerges
import Clustering.MaximumDistance
import Clustering.nnodes
import Clustering.orderbranches_barjoseph!
import Clustering.orderbranches_r!
import Clustering.Symmetric
import Clustering.WardDistance

## Given a hierarchical cluster and some ideal positions of the leaves,
## reorder the branches so they will be the most compatible with these positions.
##
## This modifies the `Hclust` in-place, instead of the normal API of  `orderbranches_...`,
## because we
function orderbranches_byorder!(hmer::HclustMerges, positions::AbstractVector{<:Real})::Nothing
    node_summaries = Vector{Tuple{Float32, Int32}}(undef, nnodes(hmer) - 1)  # sum and num of positions of leaves of each node

    for v in 1:(nnodes(hmer) - 1)
        @inbounds vl = hmer.mleft[v]
        @inbounds vr = hmer.mright[v]

        if vl < 0
            @inbounds l_center = l_sum = positions[-vl]
            l_num = 1
        else
            @inbounds l_sum, l_num = node_summaries[vl]
            l_center = l_sum / l_num
        end

        if vr < 0
            @inbounds r_center = r_sum = positions[-vr]
            r_num = 1
        else
            @inbounds r_sum, r_num = node_summaries[vr]
            r_center = r_sum / r_num
        end

        if l_center > r_center
            @inbounds hmer.mleft[v] = vr
            @inbounds hmer.mright[v] = vl
        end

        @inbounds node_summaries[v] = (l_sum + r_sum, l_num + r_num)
    end

    return nothing
end

"""
    ehclust(d::AbstractMatrix; [linkage], [uplo], [branchorder]) -> Hclust

Enhanced [hclust]*https://github.com/JuliaStats/Clustering.jl/blob/master/src/hclust.jl).
This is similar to `hclust` with the following extensions:

- If `branchorder` is a vector of `Real` numbers, one per leaf, then we reorder the branches so that each leaf position
  would be as close as possible to its `branchorder` value. Technically we compute a center of gravity for each node and
  reorder the tree such that that at each branch, the left sub-tree center of gravity is to the left (lower than) the
  center of gravity of the right sub-tree.

```jldoctest
using Test
using Distances

data = rand(4, 10)
distances = pairwise(Euclidean(), data; dims = 2)
positions = rand(10)
result = ehclust(distances, branchorder = positions)
merges_data = Vector{Tuple{Int32, Float32}}(undef, 9)
for merge_index in 1:9
    left = result.merges[merge_index, 1]
    if left < 0
        left_size = 1
        left_center = positions[-left]
    else
        left_size, left_center = merges_data[left]
    end

    right = result.merges[merge_index, 2]
    if right < 0
        right_size = 1
        right_center = positions[-right]
    else
        right_size, right_center = merges_data[right]
    end

    @test left_center <= right_center
    merged_size = left_size + right_size
    merged_center = (left_center * left_size + right_center * right_size) / merged_size
    merges_data[merge_index] = (merged_size, merged_center)
end

println("OK")

# output

OK
```
"""
function ehclust(
    d::AbstractMatrix;
    linkage::Symbol = :single,
    uplo::Union{Symbol, Nothing} = nothing,
    branchorder::Union{Symbol, AbstractVector{<:Real}} = :r,
)::Hclust
    if uplo !== nothing
        sd = Symmetric(d, uplo) # use upper/lower part of d  # NOJET # UNTESTED
    else
        assertdistancematrix(d)
        sd = d
    end
    if linkage == :single
        hmer = hclust_minimum(sd)
    elseif linkage == :complete  # UNTESTED
        hmer = hclust_nn_lw(sd, MaximumDistance(sd))  # UNTESTED
    elseif linkage == :average  # UNTESTED
        hmer = hclust_nn_lw(sd, AverageDistance(sd))  # UNTESTED
    elseif linkage == :ward_presquared  # UNTESTED
        hmer = hclust_nn_lw(sd, WardDistance(sd))  # UNTESTED
    elseif linkage == :ward  # UNTESTED
        if sd === d  # UNTESTED
            sd = abs2.(sd)  # UNTESTED
        else
            sd .= abs2.(sd)  # UNTESTED
        end
        hmer = hclust_nn_lw(sd, WardDistance(sd))  # UNTESTED
        hmer.heights .= sqrt.(hmer.heights)  # UNTESTED
    else
        throw(ArgumentError("Unsupported cluster linkage $linkage"))  # UNTESTED
    end

    if branchorder == :barjoseph || branchorder == :optimal
        orderbranches_barjoseph!(hmer, sd)  # NOJET  # UNTESTED
    elseif branchorder == :r
        orderbranches_r!(hmer)  # UNTESTED
    elseif branchorder isa AbstractVector{<:Real}
        orderbranches_byorder!(hmer, branchorder)
    else
        throw(ArgumentError("Unsupported branchorder=$branchorder method"))  # UNTESTED
    end
    return Hclust(hmer, linkage)
end

end
