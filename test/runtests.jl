using Documenter
using Logging
using SparseArrays
using SpecialFunctions
using StatsBase
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
        result = Matrix{Float64}(undef, n, 3)
        for i in 1:n
            ref = chi2_manual(x[i, 1], x[i, 2], col_sum1, col_sum2; yates)
            result[i, 1] = ref.chi2
            result[i, 2] = ref.pval
        end
        # Benjamini-Hochberg FDR
        p_values = result[:, 2]
        order = sortperm(p_values)
        for (rank, row_index) in enumerate(order)
            result[row_index, 3] = min(p_values[row_index] * n / rank, 1.0)
        end
        for rank in (n - 1):-1:1
            result[order[rank], 3] = min(result[order[rank], 3], result[order[rank + 1], 3])
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

@testset "sparse_statistics" begin
    @testset "matches Statistics.quantile on dense vectors" begin
        for vector in (Float64[1, 2, 3, 4, 5], Float64[-3, -1, 0, 0, 2, 4], Float64[7], Float64[1, 1, 1, 1])
            for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
                @test sparse_quantile(vector, p) ≈ quantile(vector, p)
            end
            @test sparse_median(vector) ≈ median(vector)
        end
    end

    @testset "matches Statistics.quantile on sparse vectors" begin
        for dense_vector in
            ([0, 0, 1, 2, 3, 0, 0], [-3, 0, -1, 0, 0, 2, 4, 0], [0, 0, 0, 0, 0], [5, 5, 5, 5], [-1, 0, 1])
            sparse_vec = sparse(dense_vector)
            for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
                @test sparse_quantile(sparse_vec, p) ≈ quantile(dense_vector, p)
            end
            @test sparse_median(sparse_vec) ≈ median(dense_vector)
        end
    end

    @testset "positive flag matches when all values are non-negative" begin
        sparse_vec = sparse([0, 0, 3, 0, 1, 4, 0, 1])
        for p in (0.0, 0.25, 0.5, 0.75, 1.0)
            @test sparse_quantile(sparse_vec, p; positive = true) ≈ sparse_quantile(sparse_vec, p)
        end
        @test sparse_median(sparse_vec; positive = true) ≈ sparse_median(sparse_vec)
    end

    @testset "respects explicit zeros stored in nzval" begin
        # SparseVector with an explicit zero entry should give the same answer as the dense vector.
        sparse_vec = SparseVector(6, [2, 4, 5], [0, 1, 2])
        dense_vec = [0, 0, 0, 1, 2, 0]
        for p in (0.0, 0.25, 0.5, 0.75, 1.0)
            @test sparse_quantile(sparse_vec, p) ≈ quantile(dense_vec, p)
        end
    end

    @testset "matrix per-column reduction matches dense" begin
        dense_matrix = [
            0.0 1.0 0.0 4.0
            2.0 0.0 0.0 5.0
            0.0 3.0 0.0 6.0
            0.0 0.0 0.0 7.0
        ]
        sparse_matrix = sparse(dense_matrix)
        for p in (0.0, 0.25, 0.5, 0.75, 1.0)
            expected = [quantile(view(dense_matrix, :, j), p) for j in 1:4]
            @test sparse_quantile(sparse_matrix, p; dims = Rows) ≈ expected
            @test sparse_quantile(dense_matrix, p; dims = Rows) ≈ expected
        end
        expected_median = [median(view(dense_matrix, :, j)) for j in 1:4]
        @test sparse_median(sparse_matrix; dims = Rows) ≈ expected_median
        @test sparse_median(dense_matrix; dims = Rows) ≈ expected_median
    end

    @testset "matrix per-row reduction matches dense for row-major inputs" begin
        dense_matrix = [
            0.0 2.0 0.0 0.0
            1.0 0.0 3.0 0.0
            0.0 0.0 0.0 0.0
            4.0 5.0 6.0 7.0
        ]
        # Build a row-major sparse view by transposing a column-major sparse matrix.
        transposed_sparse = transpose(sparse(transpose(dense_matrix)))
        # Build a row-major dense view via flip.
        transposed_dense = flip(collect(transpose(dense_matrix)))
        for p in (0.0, 0.25, 0.5, 0.75, 1.0)
            expected = [quantile(view(dense_matrix, i, :), p) for i in 1:4]
            @test sparse_quantile(transposed_sparse, p; dims = Columns) ≈ expected
            @test sparse_quantile(transposed_dense, p; dims = Columns) ≈ expected
        end
    end

    @testset "sparse_var/sparse_std match Statistics" begin
        for vector in (Float64[1, 2, 3, 4, 5], Float64[-3, -1, 0, 0, 2, 4], Float64[5, 5, 5, 5])
            @test sparse_var(vector) ≈ var(vector)
            @test sparse_var(vector; corrected = false) ≈ var(vector; corrected = false)
            @test sparse_std(vector) ≈ std(vector)
            @test sparse_std(vector; corrected = false) ≈ std(vector; corrected = false)
        end

        for dense_vector in ([0, 0, 1, 2, 3, 0, 0], [-3, 0, -1, 0, 0, 2, 4, 0], [0, 0, 0, 0, 0], [-1, 0, 1])
            sparse_vec = sparse(dense_vector)
            @test sparse_var(sparse_vec) ≈ var(dense_vector)
            @test sparse_var(sparse_vec; corrected = false) ≈ var(dense_vector; corrected = false)
            @test sparse_std(sparse_vec) ≈ std(dense_vector)
        end

        # Single-element edge cases
        @test isnan(sparse_var([7.0]))
        @test sparse_var([7.0]; corrected = false) == 0.0
        @test isnan(sparse_var(Float64[]))

        # Per-axis reductions
        dense_matrix = Float64[0 1 0 4; 2 0 0 5; 0 3 0 6; 0 0 0 7]
        sparse_matrix = sparse(dense_matrix)
        for corrected in (true, false)
            expected_per_col = [var(view(dense_matrix, :, j); corrected) for j in 1:4]
            @test sparse_var(dense_matrix; dims = Rows, corrected) ≈ expected_per_col
            @test sparse_var(sparse_matrix; dims = Rows, corrected) ≈ expected_per_col
            @test sparse_std(sparse_matrix; dims = Rows, corrected) ≈ sqrt.(expected_per_col)
        end

        # Row-major dense / sparse for dims = Columns (avoids inefficient action handler)
        transposed_dense = flip(collect(transpose(dense_matrix)))
        transposed_sparse = transpose(sparse(transpose(dense_matrix)))
        expected_per_row = [var(view(dense_matrix, i, :)) for i in 1:4]
        @test sparse_var(transposed_dense; dims = Columns) ≈ expected_per_row
        @test sparse_var(transposed_sparse; dims = Columns) ≈ expected_per_row

        # Whole-matrix scalar reduction
        @test sparse_var(dense_matrix) ≈ var(vec(dense_matrix))
        @test sparse_var(sparse_matrix) ≈ var(vec(Matrix(sparse_matrix)))
        @test sparse_std(sparse_matrix) ≈ std(vec(Matrix(sparse_matrix)))

        @test_throws AssertionError sparse_var(dense_matrix; dims = 3)

        # Caller-provided result buffer (exercises the size-check assertion).
        provided_result = Vector{Float64}(undef, size(dense_matrix, 2))
        sparse_var(sparse_matrix; dims = Rows, result = provided_result)
        @test provided_result ≈ [var(view(dense_matrix, :, j)) for j in 1:size(dense_matrix, 2)]
    end

    @testset "matrix scalar reduction without dims flattens" begin
        dense_matrix = Float64[1 2 3; 4 5 6]
        @test sparse_median(dense_matrix) ≈ median(vec(dense_matrix))
        for p in (0.0, 0.25, 0.5, 0.75, 1.0)
            @test sparse_quantile(dense_matrix, p) ≈ quantile(vec(dense_matrix), p)
        end

        sparse_matrix = sparse([0.0 1.0 0.0 4.0; 2.0 0.0 0.0 5.0; 0.0 3.0 0.0 6.0; 0.0 0.0 0.0 7.0])
        flat_dense = vec(Matrix(sparse_matrix))
        @test sparse_median(sparse_matrix) ≈ median(flat_dense)
        for p in (0.0, 0.25, 0.5, 0.75, 1.0)
            @test sparse_quantile(sparse_matrix, p) ≈ quantile(flat_dense, p)
        end
    end

    @testset "errors on invalid p and dims" begin
        @test_throws AssertionError sparse_quantile([1.0, 2.0], -0.1)
        @test_throws AssertionError sparse_quantile([1.0, 2.0], 1.1)
        @test_throws AssertionError sparse_quantile([1.0 2.0; 3.0 4.0], 0.5; dims = 3)
        @test_throws AssertionError sparse_median([1.0 2.0; 3.0 4.0]; dims = 0)
    end

    @testset "caller-provided result and scratch buffers" begin
        sparse_vec = sparse([0, 0, 1, 2, 3])
        scratch_vec = Vector{Float64}(undef, nnz(sparse_vec))
        @test sparse_quantile(sparse_vec, 0.5; scratch = scratch_vec) ≈ quantile([0, 0, 1, 2, 3], 0.5)

        dense_vec = Float64[1, 2, 3, 4, 5]
        dense_scratch = Vector{Float64}(undef, length(dense_vec))
        @test sparse_quantile(dense_vec, 0.5; scratch = dense_scratch) ≈ quantile(dense_vec, 0.5)

        sparse_matrix = sparse([0 1 0 4; 2 0 0 5; 0 3 0 6; 0 0 0 7])
        sparse_result = Vector{Float64}(undef, size(sparse_matrix, 2))
        sparse_scratch = Vector{Float64}(undef, nnz(sparse_matrix))
        sparse_quantile(sparse_matrix, 0.5; dims = Rows, result = sparse_result, scratch = sparse_scratch)
        @test sparse_result ≈ [quantile(view(sparse_matrix, :, j), 0.5) for j in 1:size(sparse_matrix, 2)]

        dense_matrix = Float64[1 2 3 4; 5 6 7 8; 9 10 11 12]
        dense_result = Vector{Float64}(undef, size(dense_matrix, 2))
        dense_scratch = Vector{Float64}(undef, length(dense_matrix))
        sparse_quantile(dense_matrix, 0.5; dims = Rows, result = dense_result, scratch = dense_scratch)
        @test dense_result ≈ [quantile(view(dense_matrix, :, j), 0.5) for j in 1:size(dense_matrix, 2)]

        flat_scratch = Vector{Float64}(undef, nnz(sparse_matrix))
        @test sparse_quantile(sparse_matrix, 0.5; scratch = flat_scratch) ≈ quantile(vec(Matrix(sparse_matrix)), 0.5)
    end

    @testset "quickselect partition path on large shuffled inputs" begin
        # Shuffled values with > 16 stored entries exercise median-of-three swaps and the Hoare
        # partition swap in `quickselect!`; sorted data leaves those branches dead.
        rng = Random.MersenneTwister(42)
        dense_vector = Random.shuffle!(rng, vcat(collect(1.0:80.0), collect(-40.0:-1.0)))
        @test length(dense_vector) >= 16
        for p in (0.0, 0.05, 0.25, 0.5, 0.6, 0.75, 0.95, 1.0)
            @test sparse_quantile(dense_vector, p) ≈ quantile(dense_vector, p)
        end

        # Sparse matrix variant where each column has > 16 stored values and few implicit zeros,
        # so most quantile positions land on stored values and reach `quickselect!`.
        n_rows = 130
        column_values = Random.shuffle!(rng, vcat(collect(1.0:80.0), collect(-40.0:-1.0)))
        rows = sort!(Random.shuffle!(rng, collect(1:n_rows))[1:length(column_values)])
        sparse_matrix = sparse(rows, fill(1, length(column_values)), column_values, n_rows, 1)
        @test nnz(sparse_matrix) >= 16
        for p in (0.0, 0.05, 0.25, 0.5, 0.6, 0.75, 0.95, 1.0)
            expected = quantile(Vector(sparse_matrix[:, 1]), p)
            @test sparse_quantile(sparse_matrix, p; dims = Rows)[1] ≈ expected
            @test sparse_quantile(sparse_matrix[:, 1], p) ≈ expected
        end
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
