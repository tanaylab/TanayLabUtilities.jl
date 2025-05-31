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

include("logger.jl")
@reexport using .Logger

include("parallel_storage.jl")
@reexport using .ParallelStorage

include("parallel_rng.jl")
@reexport using .ParallelRNG

include("locks.jl")
@reexport using .Locks

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

include("downsample.jl")
@reexport using .Downsample

include("cross_validations.jl")
@reexport using .CrossValidations

include("global_weak_cache.jl")
@reexport using .GlobalWeakCache

end  # module
