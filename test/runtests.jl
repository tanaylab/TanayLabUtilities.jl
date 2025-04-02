using Documenter
using Logging
using NestedTests
using TanayLabUtilities
using Test

import Random

Random.seed!(123456)

setup_logger(; level = Info)

#TODOX inefficient_action_handler(ErrorHandler)
abort_on_first_failure(true)

nested_test("doctests") do
    DocMeta.setdocmeta!(TanayLabUtilities, :DocTestSetup, :(using TanayLabUtilities); recursive = true)
    return doctest(TanayLabUtilities; manual = false)
end
