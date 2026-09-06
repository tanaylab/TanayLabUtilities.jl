push!(LOAD_PATH, ".")

using Aqua
using Test
using TanayLabUtilities
Aqua.test_ambiguities([TanayLabUtilities])
Aqua.test_all(
    TanayLabUtilities;
    ambiguities = false,
    unbound_args = false,
    deps_compat = false,
    persistent_tasks = false,
)

# Aqua's own default of 10 seconds is not the time the package spends starting tasks; it is the time the whole
# precompilation subprocess takes to exit after loading, which on a loaded machine writing its cache over the
# network is regularly longer than that. A larger budget tests the same thing without failing at random.
@testset "Persistent tasks" begin
    Aqua.test_persistent_tasks(TanayLabUtilities; tmax = 60)
end
