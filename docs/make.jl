# This file is copied into `docs` so that when the package is published its documentation will be built automatically,
# that is, pretend the package uses the usual "docs/make.jl" idiom. Normally this isn't used because we build the
# documentation locally and publish them in github pages. This way, in github we have the head version documentation,
# while in the standard Julia packages documentation we have the documentation of the last published version.

using Documenter

push!(LOAD_PATH, "..")

using NestedTests
using Pkg
using TanayLabUtilities

for file in readdir("docs"; join = true)
    if !endswith(file, "make.jl")
        rm(file; force = true, recursive = true)
    end
end

PROJECT_TOML = Pkg.TOML.parsefile("Project.toml")
NAME = PROJECT_TOML["name"]
AUTHORS = PROJECT_TOML["authors"]

makedocs(;
    authors = join(" ", AUTHORS),
    source = "../src",
    clean = true,
    doctest = true,
    modules = [TanayLabUtilities],
    highlightsig = true,
    sitename = "$(NAME).jl",
    draft = false,
    strict = true,
    linkcheck = true,
    format = Documenter.HTML(; prettyurls = false),
    pages = [
        "index.md",
        "types.md",
        "brief.md",
        "logger.md",
        "parallel_storage.md",
        "parallel_rng.md",
        "locks.md",
        "unique_names.md",
        "documentation.md",
        "handlers.md",
        "read_only_arrays.md",
        "downsample.md",
        "cross_validations.md",
        "global_weak_cache.md",
    ],
)

for file in readdir("docs/build"; join = true)
    if endswith(file, ".cov")
        rm(file)
    end
end
