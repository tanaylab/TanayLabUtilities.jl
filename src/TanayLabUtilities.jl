"""
Generic utilities used by the Julia code for the Tanay lab.

This is a somewhat arbitrary collection of "generally useful" functions, created on an as-needed bases. No claim is made
for true universality or for suitability for a particular purpose.

If you write `using TanayLabUtilities` everything will be exported into your global namespaces. Since this is a collection
of loosely related functions, you may prefer to write `using TanayLabUtilities.SomethingSpecific` instead, or just import
the specific symbol(s) you need.

The included modules are:

![](assets/modules.svg)
"""
module TanayLabUtilities

using Reexport

include("types.jl")
@reexport using .Types

include("brief.jl")
@reexport using .Brief

include("flame_time.jl")
@reexport using .FlameTime

include("logger.jl")
@reexport using .Logger

include("parallel_storage.jl")
@reexport using .ParallelStorage

include("unique_names.jl")
@reexport using .UniqueNames

include("handlers.jl")
@reexport using .Handlers

include("documentation.jl")
@reexport using .Documentation

include("read_only_arrays.jl")
@reexport using .ReadOnlyArrays

include("matrix_layouts.jl")
@reexport using .MatrixLayouts

include("matrix_formats.jl")
@reexport using .MatrixFormats

include("cross_validations.jl")
@reexport using .CrossValidations

include("global_weak_cache.jl")
@reexport using .GlobalWeakCache

include("kmeans.jl")
@reexport using .KMeans

include("cached_ispath.jl")
@reexport using .CachedIsPath

include("parallel_loops.jl")
@reexport using .ParallelLoops

include("parallel_distances.jl")
@reexport using .ParallelDistances

include("mostly_read_write_locks.jl")
@reexport using .MostlyReadWriteLocks

include("extended_read_write_locks.jl")
@reexport using .ExtendedReadWriteLocks

include("downsample.jl")
@reexport using .Downsample

include("zero_correlation.jl")
@reexport using .ZeroCorrelation

end  # module
