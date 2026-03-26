using Documenter
using Logging
using SparseArrays
using SpecialFunctions
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

@testset "chi_squared" begin
    # Manual reference implementation for verification
    function chi2_manual(a, c_val, col_sum1, col_sum2; yates = true)
        a, c_val, col_sum1, col_sum2 = Float64(a), Float64(c_val), Float64(col_sum1), Float64(col_sum2)
        b = col_sum1 - a
        d = col_sum2 - c_val
        n_total = col_sum1 + col_sum2
        c1 = a + c_val
        c2 = b + d
        denom = col_sum1 * col_sum2 * c1 * c2
        if denom == 0.0
            return (chi2 = 0.0, pval = 1.0)  # UNTESTED
        end
        ad_bc = abs(a * d - b * c_val)
        if yates
            numerator = max(ad_bc - n_total / 2.0, 0.0)
        else
            numerator = ad_bc
        end
        chi2_val = numerator^2 * n_total / denom
        pval = erfc(sqrt(chi2_val / 2.0))
        return (chi2 = chi2_val, pval = pval)
    end

    function chi2_expected(x; yates = true)
        col_sum1 = sum(x[:, 1])
        col_sum2 = sum(x[:, 2])
        n = size(x, 1)
        result = Matrix{Float64}(undef, n, 2)
        for i in 1:n
            ref = chi2_manual(x[i, 1], x[i, 2], col_sum1, col_sum2; yates)
            result[i, 1] = ref.chi2
            result[i, 2] = ref.pval
        end
        return result
    end

    @testset "matches manual reference with Yates" begin
        x = [10 20; 30 40; 50 60; 5 15; 25 35]
        result = chi_squared(x; yates = true)
        expected = chi2_expected(x; yates = true)
        @test result ≈ expected atol = 1e-12
    end

    @testset "matches manual reference without Yates" begin
        x = [10 20; 30 40; 50 60; 5 15; 25 35]
        result = chi_squared(x; yates = false)
        expected = chi2_expected(x; yates = false)
        @test result ≈ expected atol = 1e-12
    end

    @testset "integer matrix" begin
        x = Int[10 20; 30 40; 50 60]
        result = chi_squared(x)
        expected = chi2_expected(x)
        @test result ≈ expected atol = 1e-12
    end

    @testset "float matrix" begin
        x = Float64[10 20; 30 40; 50 60]
        result = chi_squared(x)
        expected = chi2_expected(x)
        @test result ≈ expected atol = 1e-12
    end

    @testset "sparse matrix" begin
        x_dense = [0 5; 10 0; 0 0; 3 7; 0 12]
        x_sparse = sparse(x_dense)
        result_dense = chi_squared(x_dense)
        result_sparse = chi_squared(x_sparse)
        @test result_dense ≈ result_sparse atol = 1e-12
    end

    @testset "zero row gives chi2=0, pval=1" begin
        x = [10 5; 0 0; 3 7]
        result = chi_squared(x)
        @test result[2, 1] == 0.0
        @test result[2, 2] == 1.0
    end

    @testset "single row gives chi2=0, pval=1" begin
        x = reshape([10, 20], 1, 2)
        result = chi_squared(x)
        @test result[1, 1] == 0.0
        @test result[1, 2] == 1.0
    end

    @testset "all-zero column gives chi2=0, pval=1" begin
        x = [10 0; 20 0; 30 0]
        result = chi_squared(x)
        @test all(result[:, 1] .== 0.0)
        @test all(result[:, 2] .== 1.0)
    end

    @testset "errors on NaN" begin
        @test_throws AssertionError chi_squared([10.0, NaN], [20.0, 25.0])
        @test_throws AssertionError chi_squared([10.0, 30.0], [NaN, 25.0])
    end

    @testset "Yates reduces chi2" begin
        x = [10 20; 30 40; 50 60; 5 15]
        result_yates = chi_squared(x; yates = true)
        result_no_yates = chi_squared(x; yates = false)
        @test all(result_yates[:, 1] .<= result_no_yates[:, 1] .+ 1e-12)
        @test all(result_yates[:, 2] .>= result_no_yates[:, 2] .- 1e-12)
    end

    @testset "chi2 >= 0 and pval in [0, 1]" begin
        x = rand(0:100, 50, 2)
        for yates in (true, false)
            result = chi_squared(x; yates)
            @test all(result[:, 1] .>= 0)
            @test all(0 .<= result[:, 2] .<= 1)
        end
    end

    @testset "large counts numerical stability" begin
        x = [1_000_000 1_000_100; 500_000 500_100; 10_000_000 9_999_950]
        result = chi_squared(x; yates = false)
        expected = chi2_expected(x; yates = false)
        @test result ≈ expected atol = 1e-8
    end

    @testset "errors on wrong number of columns" begin
        @test_throws AssertionError chi_squared(rand(3, 3))
        @test_throws AssertionError chi_squared(rand(3, 1))
    end

    @testset "errors on negative counts" begin
        @test_throws AssertionError chi_squared([-1 3; 2 4])
    end
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
