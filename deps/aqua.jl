push!(LOAD_PATH, ".")

using Aqua
using TanayLabUtilities
Aqua.test_ambiguities([TanayLabUtilities])
Aqua.test_all(TanayLabUtilities; ambiguities = false, unbound_args = false, deps_compat = false)
