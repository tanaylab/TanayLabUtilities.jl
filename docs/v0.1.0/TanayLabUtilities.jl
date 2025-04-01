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

#TODOX include("parallel.jl")
#TODOX @reexport using .Parallel

include("locks.jl")
@reexport using .Locks

include("names.jl")
@reexport using .Names

include("handlers.jl")
@reexport using .Handlers

end  # module
