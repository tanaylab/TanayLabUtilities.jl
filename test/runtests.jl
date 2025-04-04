using Documenter
using Logging
using TanayLabUtilities
using Test

import Random

Random.seed!(123456)

setup_logger(; level = Info)

TanayLabUtilities.MatrixLayouts.GLOBAL_INEFFICIENT_ACTION_HANDLER = ErrorHandler

@testset "doctests" begin
    DocMeta.setdocmeta!(TanayLabUtilities, :DocTestSetup, :(using TanayLabUtilities); recursive = true)
    return doctest(TanayLabUtilities; manual = false)
end
