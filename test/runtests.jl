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

@testset "cached_ispath" begin
    base = mktempdir()

    real = joinpath(base, "real")
    mkdir(real)

    alias = joinpath(base, "alias")
    symlink(real, alias)

    target_real = joinpath(real, "junk.txt")
    target_alias = joinpath(alias, "junk.txt")

    cached_ispath(target_alias)

    open(target_real, "w") do io
        return write(io, "hi")
    end
    report_modified!(target_alias)

    @assert cached_ispath(target_real)
    @assert cached_ispath(target_alias)
end
