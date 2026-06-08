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

@testset "grouped_correlations" begin
    # Reference: per-series Pearson correlation between fixed and variable, averaged over series,
    # using the same zero-variance-as-0 convention as the incremental code.
    function reference_mean(fixed, variable)
        n_series = size(fixed, 2)
        return mean(zero_cor_between_vectors(fixed[:, s], variable[:, s]) for s in 1:n_series)
    end

    @testset "build + mean_correlation match per-series Pearson" begin
        fixed = Float32[1 2 3; 3 1 4; 2 4 2; 5 1 3]
        variable = Float32[2 1 4; 1 3 2; 4 2 1; 1 5 3]
        for sizes in ([2, 2], [1, 3], [4], [1, 1, 1, 1])
            grouped = GroupedSeriesCorrelations(fixed, variable, sizes)
            @test mean_correlation(grouped) ≈ reference_mean(fixed, variable) atol = 1e-6
        end
    end

    @testset "correlation_per_series! matches zero_cor per series" begin
        fixed = Float32[1 2; 3 1; 2 4; 5 1]
        variable = Float32[2 1; 1 3; 4 2; 1 5]
        grouped = GroupedSeriesCorrelations(fixed, variable, [2, 2])
        per_series = Vector{Float64}(undef, 2)
        correlation_per_series!(grouped, per_series)
        @test per_series[1] ≈ zero_cor_between_vectors(fixed[:, 1], variable[:, 1]) atol = 1e-6
        @test per_series[2] ≈ zero_cor_between_vectors(fixed[:, 2], variable[:, 2]) atol = 1e-6
    end

    @testset "what-if is non-mutating and matches a fresh build" begin
        fixed = Float32[1 2; 3 1; 2 4; 5 1]
        variable = Float32[2 1; 1 3; 4 2; 1 5]
        grouped = GroupedSeriesCorrelations(fixed, variable, [2, 2])
        baseline = mean_correlation(grouped)
        new_variable = Float32[5 5; 1 2]
        score = mean_correlation_if_group_replaced(grouped, 1, new_variable)
        @test mean_correlation(grouped) == baseline
        replaced = copy(variable)
        replaced[1:2, :] .= new_variable
        @test score ≈ reference_mean(fixed, replaced) atol = 1e-6
    end

    @testset "replace_group! commits and matches a fresh build" begin
        fixed = Float32[1 2; 3 1; 2 4; 5 1]
        variable = Float32[2 1; 1 3; 4 2; 1 5]
        grouped = GroupedSeriesCorrelations(fixed, variable, [2, 2])
        new_variable = Float32[5 5; 1 2]
        what_if_score = mean_correlation_if_group_replaced(grouped, 1, new_variable)
        replace_group!(grouped, 1, new_variable)
        @test mean_correlation(grouped) ≈ what_if_score atol = 1e-7
        replaced = copy(variable)
        replaced[1:2, :] .= new_variable
        @test mean_correlation(grouped) ≈ reference_mean(fixed, replaced) atol = 1e-6
    end

    @testset "multi-group commit matches a fresh build" begin
        n_points, n_series = 12, 5
        sizes = [4, 4, 4]
        fixed = rand(Float32, n_points, n_series)
        variable = rand(Float32, n_points, n_series)
        grouped = GroupedSeriesCorrelations(fixed, variable, sizes)
        new_variable_1 = rand(Float32, 4, n_series)
        new_variable_3 = rand(Float32, 4, n_series)
        replace_group!(grouped, 1, new_variable_1)
        replace_group!(grouped, 3, new_variable_3)
        replaced = copy(variable)
        replaced[1:4, :] .= new_variable_1
        replaced[9:12, :] .= new_variable_3
        @test mean_correlation(grouped) ≈ reference_mean(fixed, replaced) atol = 1e-5
    end

    @testset "zero-variance series returns 0" begin
        constant_fixed = ones(Float32, 4, 2)
        variable = Float32[2 1; 1 3; 4 2; 1 5]
        @test mean_correlation(GroupedSeriesCorrelations(constant_fixed, variable, [2, 2])) == 0.0

        fixed = Float32[2 1; 1 3; 4 2; 1 5]
        constant_variable = ones(Float32, 4, 2)
        @test mean_correlation(GroupedSeriesCorrelations(fixed, constant_variable, [2, 2])) == 0.0
    end

    @testset "wrong-size variable matrix is rejected" begin
        fixed = Float32[1 2; 3 1; 2 4; 5 1]
        variable = Float32[2 1; 1 3; 4 2; 1 5]
        grouped = GroupedSeriesCorrelations(fixed, variable, [2, 2])
        @test_throws ErrorException mean_correlation_if_group_replaced(grouped, 1, Float32[5 5 5; 1 2 3])
        @test_throws ErrorException mean_correlation_if_group_replaced(grouped, 1, Float32[5 5; 1 2; 7 8])
        @test_throws ErrorException replace_group!(grouped, 1, Float32[5 5 5; 1 2 3])
        @test_throws ErrorException GroupedSeriesCorrelations(fixed, Float32[1 2; 3 4], [1, 1])
    end

    @testset "check_turbo macros reject non-turbo arrays" begin
        @test_throws ErrorException @check_turbo_vector(["a", "b"])
        @test_throws ErrorException @check_turbo_matrix(["a" "b"; "c" "d"])
    end

    @testset "check_turbo_vector rejects non-vector values" begin
        @test_throws ErrorException @check_turbo_vector(Float32[1 2; 3 4])
        @test_throws ErrorException @check_turbo_vector(42)
    end

    @testset "check_turbo_matrix rejects non-matrix values" begin
        @test_throws ErrorException @check_turbo_matrix(Float32[1, 2, 3])
        @test_throws ErrorException @check_turbo_matrix(42)
    end

    @testset "assert_matrix rejects non-matrix values" begin
        @test_throws ErrorException @assert_matrix(Float32[1, 2, 3])
        @test_throws ErrorException @assert_matrix(42)
    end

    @testset "mask: all-active default is identical to explicit trues" begin
        fixed = Float32[1 2; 3 1; 2 4; 5 1]
        variable = Float32[2 1; 1 3; 4 2; 1 5]
        grouped_default = GroupedSeriesCorrelations(fixed, variable, [2, 2])
        grouped_explicit = GroupedSeriesCorrelations(fixed, variable, [2, 2]; is_active_per_point = trues(4))
        @test mean_correlation(grouped_default) == mean_correlation(grouped_explicit)
    end

    @testset "mask: build with mask matches per-series Pearson over active points only" begin
        fixed = Float32[1 2 3; 3 1 4; 2 4 2; 5 1 3]
        variable = Float32[2 1 4; 1 3 2; 4 2 1; 1 5 3]
        is_active = Bool[true, false, true, true]
        grouped = GroupedSeriesCorrelations(fixed, variable, [2, 2]; is_active_per_point = is_active)
        active_fixed = fixed[is_active, :]
        active_variable = variable[is_active, :]
        @test mean_correlation(grouped) ≈ reference_mean(active_fixed, active_variable) atol = 1e-6
    end

    @testset "mask: replace_group! with new mask updates totals + active count" begin
        fixed = Float32[1 2; 3 1; 2 4; 5 1]
        variable = Float32[2 1; 1 3; 4 2; 1 5]
        grouped = GroupedSeriesCorrelations(fixed, variable, [2, 2])
        new_variable = Float32[5 5; 1 2]
        new_mask = Bool[false, true]
        replace_group!(grouped, 1, new_variable, new_mask)
        replaced_variable = copy(variable)
        replaced_variable[1:2, :] .= new_variable
        is_active_now = Bool[false, true, true, true]
        @test mean_correlation(grouped) ≈ reference_mean(fixed[is_active_now, :], replaced_variable[is_active_now, :]) atol =
            1e-5
        @test grouped.is_active_per_point == is_active_now
        @test grouped.n_active_per_group == [1, 2]
    end

    @testset "mask: what-if with new mask is non-mutating and matches the commit" begin
        fixed = Float32[1 2; 3 1; 2 4; 5 1]
        variable = Float32[2 1; 1 3; 4 2; 1 5]
        grouped = GroupedSeriesCorrelations(fixed, variable, [2, 2])
        baseline = mean_correlation(grouped)
        baseline_mask = copy(grouped.is_active_per_point)
        new_variable = Float32[5 5; 1 2]
        new_mask = Bool[false, true]
        score_what_if = mean_correlation_if_group_replaced(grouped, 1, new_variable, new_mask)
        @test mean_correlation(grouped) == baseline
        @test grouped.is_active_per_point == baseline_mask
        replace_group!(grouped, 1, new_variable, new_mask)
        @test mean_correlation(grouped) ≈ score_what_if atol = 1e-7
    end

    @testset "mask: all points masked off returns 0" begin
        fixed = Float32[1 2; 3 1; 2 4; 5 1]
        variable = Float32[2 1; 1 3; 4 2; 1 5]
        is_active = Bool[false, false, false, false]
        grouped = GroupedSeriesCorrelations(fixed, variable, [2, 2]; is_active_per_point = is_active)
        @test mean_correlation(grouped) == 0.0
    end

    @testset "mask: flipping points back to active restores their fixed-side contribution" begin
        fixed = Float32[1 2; 3 1; 2 4; 5 1]
        variable = Float32[2 1; 1 3; 4 2; 1 5]
        baseline = GroupedSeriesCorrelations(fixed, variable, [2, 2])
        baseline_mean = mean_correlation(baseline)
        # Mask a point off, then mask it back on with the original variable values - should round-trip.
        grouped = GroupedSeriesCorrelations(fixed, variable, [2, 2])
        replace_group!(grouped, 1, variable[1:2, :], Bool[false, true])
        replace_group!(grouped, 1, variable[1:2, :], Bool[true, true])
        @test mean_correlation(grouped) ≈ baseline_mean atol = 1e-5
        @test grouped.n_active_per_group == [2, 2]
    end

    @testset "indirect-gather: replace_group! and what-if match the direct variant" begin
        fixed = Float32[1 2; 3 1; 2 4; 5 1]
        variable = Float32[2 1; 1 3; 4 2; 1 5]
        # The indirect-gather variants take a wider source matrix indexed via a per-point row map and a per-series
        # column map. Build a wider source whose (row_for_point, col_for_series) cells hold the same per-point/per-
        # series values as the direct call would use; both variants must then produce the same result.
        wide_variable = Float32[
            0   0   0   0
            5   0   5   0
            0   0   0   0
            1   0   2   0
        ]
        wide_is_active = Bool[true, true, false, true]  # rows 2 and 4 active; row 3 disabled
        source_index_per_point = [2, 4]            # group 1's two points pull from wide rows 2 and 4
        series_position_per_series = [1, 3]        # series 1 and 2 read wide columns 1 and 3
        is_active_per_group_point = wide_is_active[source_index_per_point]

        # What-if against a reference built from the direct 4-arg variant.
        reference = GroupedSeriesCorrelations(fixed, variable, [2, 2])
        direct_new = Float32[5 5; 1 2]
        direct_what_if = mean_correlation_if_group_replaced(reference, 1, direct_new, is_active_per_group_point)

        indirect = GroupedSeriesCorrelations(fixed, variable, [2, 2])
        indirect_what_if = mean_correlation_if_group_replaced(
            indirect,
            1,
            wide_variable,
            wide_is_active,
            source_index_per_point,
            series_position_per_series,
        )
        @test indirect_what_if ≈ direct_what_if atol = 1e-7
        @test mean_correlation(indirect) == mean_correlation(reference)  # non-mutating

        # Commit via the indirect-gather variant; result must match a direct commit.
        replace_group!(reference, 1, direct_new, is_active_per_group_point)
        replace_group!(indirect, 1, wide_variable, wide_is_active, source_index_per_point, series_position_per_series)
        @test mean_correlation(indirect) ≈ mean_correlation(reference) atol = 1e-7
        @test indirect.is_active_per_point == reference.is_active_per_point
        @test indirect.n_active_per_group == reference.n_active_per_group
    end
end

@testset "parallel_loops" begin
    using ProgressMeter

    @testset "weights derives heaviest-first order and validates inputs" begin
        weights = [3, 1, 5, 2, 4]
        # `:serial` deterministically iterates in derived order, so we observe the dispatch sequence.
        visits = Int[]
        parallel_loop_wo_rng(1:5; weights, policy = :serial) do index
            return push!(visits, index)
        end
        @test visits == sortperm(weights; rev = true)
        # Length mismatch on weights.
        @test_throws AssertionError parallel_loop_wo_rng(_ -> nothing, 1:5; weights = [1, 2, 3])
    end

    @testset "explicit order kwarg is validated and respected" begin
        explicit_order = [5, 4, 3, 2, 1]
        visits = Int[]
        parallel_loop_wo_rng(1:5; order = copy(explicit_order), policy = :serial) do index
            return push!(visits, index)
        end
        @test visits == explicit_order
        # Length mismatch on order.
        @test_throws AssertionError parallel_loop_wo_rng(_ -> nothing, 1:5; order = [1, 2, 3])
    end

    @testset ":greedy_sticky degrades to :static when work fits in one chunk" begin
        # Smaller than `nthreads()` items: the `:greedy_sticky → :static` path takes over.
        n_items = max(1, Threads.nthreads() ÷ 2)
        results = zeros(Int, n_items)
        parallel_loop_wo_rng(1:n_items; policy = :greedy_sticky) do index
            results[index] = index
            return nothing
        end
        @test results == collect(1:n_items)
    end

    @testset ":static with order shuffles via round-robin" begin
        # `:static` + `order` triggers `round_robin_for_static_chunks!` on a copy of the user's order.
        weights = collect(10:-1:1)
        results = zeros(Int, length(weights))
        parallel_loop_wo_rng(1:length(weights); policy = :static, order = sortperm(weights; rev = true)) do index
            results[index] = index
            return nothing
        end
        @test results == collect(1:length(weights))
    end

    @testset "progress advances one per iteration when no weights or chunk are given" begin
        n_items = 20
        progress = Progress(n_items)
        parallel_loop_wo_rng(1:n_items; policy = :serial, progress) do _
            return nothing
        end
        @test progress.counter == n_items
    end

    @testset "progress weighted step matches sum(weights)" begin
        weights = [3, 1, 5, 2, 4]
        progress = Progress(sum(weights))
        parallel_loop_wo_rng(1:length(weights); weights, policy = :serial, progress) do _
            return nothing
        end
        @test progress.counter == sum(weights)
    end

    @testset "progress_chunk throttles to one update per chunk" begin
        n_items = 25
        progress_chunk = 10
        progress = Progress(n_items)
        parallel_loop_wo_rng(1:n_items; policy = :serial, progress, progress_chunk) do _
            return nothing
        end
        @test progress.counter == n_items
    end
end

@testset "parallel_distances" begin
    using Distances

    @testset "parallel_colwise matches colwise for every policy" begin
        m1 = rand(10, 20)
        m2 = rand(10, 20)
        expected = colwise(Euclidean(), m1, m2)
        for policy in (:serial, :greedy, :dynamic, :static)
            @test parallel_colwise(Euclidean(), m1, m2; policy) ≈ expected atol = 1e-6
        end
    end

    @testset "parallel_colwise works with non-Euclidean distances" begin
        m1 = rand(8, 15)
        m2 = rand(8, 15)
        for distance in (Cityblock(), Chebyshev(), CosineDist())
            expected = colwise(distance, m1, m2)
            @test parallel_colwise(distance, m1, m2; policy = :static) ≈ expected atol = 1e-6
        end
    end

    @testset "parallel_colwise handles a single column" begin
        m1 = reshape(collect(1.0:5.0), 5, 1)
        m2 = reshape(collect(2.0:6.0), 5, 1)
        expected = colwise(Euclidean(), m1, m2)
        @test parallel_colwise(Euclidean(), m1, m2; policy = :greedy) ≈ expected atol = 1e-6
    end

    @testset "parallel_colwise preserves element type" begin
        m1 = rand(Float32, 6, 12)
        m2 = rand(Float32, 6, 12)
        result = parallel_colwise(Euclidean(), m1, m2; policy = :greedy)
        @test eltype(result) === Float32
        @test result ≈ colwise(Euclidean(), m1, m2) atol = 1.0f-5
    end

    @testset "parallel_colwise rejects invalid policy" begin
        m1 = rand(4, 6)
        m2 = rand(4, 6)
        @test_throws AssertionError parallel_colwise(Euclidean(), m1, m2; policy = :bogus)
    end

    @testset "parallel_colwise rejects mismatched matrix sizes" begin
        @test_throws AssertionError parallel_colwise(Euclidean(), rand(5, 4), rand(5, 6); policy = :greedy)
        @test_throws AssertionError parallel_colwise(Euclidean(), rand(5, 4), rand(7, 4); policy = :greedy)
    end
end
