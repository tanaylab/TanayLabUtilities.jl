using Documenter
using LinearAlgebra
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

@logged function logged_sum(x::Int; y::Int = 1)::Int
    return x + y
end

@documented function documented_untyped(x; y = 1)
    return x + y
end

@documented function documented_dependent(x; y::Int = 1, z::Int = x + y)::Int
    return z
end

# A read-only array type whose reads are costly, like a lazy array that reads its data on demand.
struct CostlyReadOnly <: AbstractVector{Int}
    values::Vector{Int}
    n_reads::Base.RefValue{Int}
end

Base.size(array::CostlyReadOnly) = size(array.values)

function Base.getindex(array::CostlyReadOnly, index::Int)::Int
    array.n_reads[] += 1
    return array.values[index]
end

function TanayLabUtilities.ReadOnlyArrays.is_read_only_array(::CostlyReadOnly)::Bool
    return true
end

# A `Bool` vector that counts how many of its elements were read.
struct CountingBools <: AbstractVector{Bool}
    values::Vector{Bool}
    n_reads::Base.RefValue{Int}
end

Base.size(array::CountingBools) = size(array.values)

function Base.getindex(array::CountingBools, index::Int)::Bool
    array.n_reads[] += 1
    return array.values[index]
end

# A matrix that counts how many of its elements were read, like a lazy matrix. Its plain storage is the wrapped matrix.
struct CountingMatrix <: AbstractMatrix{Float64}
    values::Matrix{Float64}
    n_reads::Base.RefValue{Int}
end

Base.size(matrix::CountingMatrix) = size(matrix.values)

function Base.getindex(matrix::CountingMatrix, row::Int, column::Int)::Float64
    matrix.n_reads[] += 1
    return matrix.values[row, column]
end

function TanayLabUtilities.MatrixLayouts.plain_array(
    matrix::CountingMatrix;
    transposed_as::Maybe{Type} = nothing,  # NOLINT
    contiguous::Bool = false,  # NOLINT
)::Matrix{Float64}
    return matrix.values
end

# A lazy matrix that reads its columns one at a time. Reading it element by element, or all of it, fails.
struct LazyColumnsMatrix <: AbstractMatrix{Float64}
    values::Matrix{Float64}
    n_scratches::Threads.Atomic{Int}
end

Base.size(matrix::LazyColumnsMatrix) = size(matrix.values)

function Base.getindex(::LazyColumnsMatrix, ::Int, ::Int)::Float64
    return error("read a lazy matrix element by element")
end

function TanayLabUtilities.ParallelLoops.column_scratch(matrix::LazyColumnsMatrix)::Vector{Float64}
    Threads.atomic_add!(matrix.n_scratches, 1)
    return Vector{Float64}(undef, size(matrix, 1))
end

function TanayLabUtilities.ParallelLoops.read_column!(
    scratch::Vector{Float64},
    matrix::LazyColumnsMatrix,
    column_index::Integer,
)::Vector{Float64}
    return copyto!(scratch, view(matrix.values, :, column_index))
end

# The sum of each of the `columns` of the `matrix`, computed by `parallel_foreach_column`.
function column_sums(
    matrix::AbstractMatrix;
    columns::AbstractVector{<:Integer} = 1:size(matrix, 2),
    kwargs...,
)::Vector{Float64}
    sums = zeros(length(columns))
    parallel_foreach_column(matrix; columns, kwargs...) do position, _, column
        sums[position] = sum(column)
        return nothing
    end
    return sums
end

# Like `column_sums`, serially, into preallocated `sums`, so it allocates only what `parallel_foreach_column` does.
function sum_columns!(sums::Vector{Float64}, matrix::AbstractMatrix, columns::AbstractVector{<:Integer})::Nothing
    parallel_foreach_column(matrix; columns, policy = :serial) do position, _, column
        sums[position] = sum(column)
        return nothing
    end
    return nothing
end

@testset "doctests" begin
    DocMeta.setdocmeta!(TanayLabUtilities, :DocTestSetup, :(using TanayLabUtilities); recursive = true)
    return doctest(TanayLabUtilities; manual = false)
end

@testset "types" begin
    maybe_value(is_present::Bool)::Maybe{Int} = is_present ? 1 : nothing
    unsure_value(is_known::Bool)::Unsure{Int} = is_known ? 1 : missing
    @test Base.return_types(maybe_value, (Bool,)) == [Maybe{Int}]
    @test Base.return_types(unsure_value, (Bool,)) == [Unsure{Int}]
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

    @testset "in place with implicit zeros" begin
        for dense_vector in ([0, 0, 1, 2, 3, 0, 0], [-3, 0, -1, 0, 0, 2, 4, 0], [5, 5, 5, 5], [-1, 0, 1])
            stored_values = Float64.(filter(!iszero, dense_vector))
            n_implicit_zero = count(iszero, dense_vector)
            for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
                @test sparse_quantile!(copy(stored_values), p; n_implicit_zero) ≈ quantile(dense_vector, p)
            end
        end
    end

    @testset "views of sparse arrays" begin
        dense_matrix = Float64[0 1 0; 2 0 3; 0 4 0; 5 0 6; 0 0 7]
        rows = [1, 2, 4, 5]
        sparse_rows = view(sparse(dense_matrix), rows, :)
        @test sparse_median(sparse_rows; dims = Rows) ≈ vec(median(dense_matrix[rows, :]; dims = 1))
        @test sparse_quantile(sparse_rows, 0.25; dims = Rows) ≈
              [quantile(dense_matrix[rows, column], 0.25) for column in 1:3]
        @test sparse_var(sparse_rows; dims = Rows) ≈ vec(var(dense_matrix[rows, :]; dims = 1))
        @test sparse_std(sparse_rows; dims = Rows) ≈ vec(std(dense_matrix[rows, :]; dims = 1))
        sparse_column = view(sparse(dense_matrix), rows, 3)
        @test sparse_median(sparse_column) ≈ median(dense_matrix[rows, 3])
        @test sparse_var(sparse_column) ≈ var(dense_matrix[rows, 3])
    end

    @testset "whole matrices read their plain storage" begin
        dense_matrix = Float64[0 1 0; 2 0 3; 0 4 0; 5 0 6]
        counting = CountingMatrix(dense_matrix, Ref(0))
        @test sparse_median(counting) ≈ median(dense_matrix)
        @test sparse_quantile(counting, 0.25) ≈ quantile(vec(dense_matrix), 0.25)
        @test sparse_var(counting) ≈ var(dense_matrix)
        @test sparse_std(counting) ≈ std(dense_matrix)
        @test counting.n_reads[] == 0
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
        sparse_quantile(sparse_matrix, 0.5; dims = Rows, result = sparse_result)
        @test sparse_result ≈ [quantile(view(sparse_matrix, :, j), 0.5) for j in 1:size(sparse_matrix, 2)]

        dense_matrix = Float64[1 2 3 4; 5 6 7 8; 9 10 11 12]
        dense_result = Vector{Float64}(undef, size(dense_matrix, 2))
        sparse_quantile(dense_matrix, 0.5; dims = Rows, result = dense_result)
        @test dense_result ≈ [quantile(view(dense_matrix, :, j), 0.5) for j in 1:size(dense_matrix, 2)]

        flat_scratch = Vector{Float64}(undef, nnz(sparse_matrix))
        @test sparse_quantile(sparse_matrix, 0.5; scratch = flat_scratch) ≈ quantile(vec(Matrix(sparse_matrix)), 0.5)

        for dims in (Rows, Columns)
            @test_throws "a scratch is only used for a vector or for a whole matrix" sparse_quantile(
                dense_matrix,
                0.5;
                dims,
                scratch = Vector{Float64}(undef, length(dense_matrix)),
            )
        end
    end

    @testset "lazy matrices are read one column at a time" begin
        values = Float64[(row + column) % 3 == 0 ? row + column / 10 : 0 for row in 1:9, column in 1:6]
        lazy_values = LazyColumnsMatrix(values, Threads.Atomic{Int}(0))
        @test sparse_var(lazy_values; dims = Rows) ≈ vec(var(values; dims = Rows))
        @test sparse_std(lazy_values; dims = Rows, corrected = false) ≈ vec(std(values; dims = Rows, corrected = false))
        @test sparse_quantile(lazy_values, 0.3; dims = Rows) ≈ [quantile(values[:, column], 0.3) for column in 1:6]
        @test sparse_median(lazy_values; dims = Rows, positive = true) ≈ vec(median(values; dims = Rows))
        @test lazy_values.n_scratches[] > 0
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

@testset "sparse_columns" begin
    matrix = sprand(Random.MersenneTwister(1), 200, 7, 0.3)
    matrix[:, 7] .= 0
    dropzeros!(matrix)
    row_indices_per_case = (Int[], [17], [3, 8, 12, 17, 150], collect(1:2:200), collect(1:200))

    @testset "stored_values_at_rows" begin
        for row_indices in row_indices_per_case, column_index in 1:7
            expected = [
                (position, matrix[row_index, column_index]) for
                (position, row_index) in enumerate(row_indices) if matrix[row_index, column_index] != 0
            ]
            @test collect(stored_values_at_rows(matrix, column_index, row_indices)) == expected
            @test collect(stored_values_at_rows(view(matrix, :, column_index), row_indices)) == expected
            @test collect(stored_values_at_rows(matrix[:, column_index], row_indices)) == expected
        end
        @test_throws "row indices are not strictly increasing" stored_values_at_rows(matrix, 1, [3, 2])
        @test_throws "row indices are not strictly increasing" stored_values_at_rows(matrix, 1, [3, 3])
        @test_throws "row indices are not strictly increasing" stored_values_at_rows(matrix[:, 1], [3, 2])
        @test eltype(stored_values_at_rows(matrix, 1, [1])) == Tuple{Int, eltype(matrix)}
    end

    @testset "gather_sparse_column!" begin
        for row_indices in row_indices_per_case, column_index in 1:7
            destination = fill(-1.0, length(row_indices))
            is_any_nonzero = gather_sparse_column!(destination, matrix, column_index, row_indices)
            @test destination == matrix[row_indices, column_index]
            @test is_any_nonzero == any(!iszero, destination)
            for column in (view(matrix, :, column_index), matrix[:, column_index])
                fill!(destination, -1.0)
                @test gather_sparse_column!(destination, column, row_indices) == is_any_nonzero
                @test destination == matrix[row_indices, column_index]
            end
        end
    end

    @testset "sum_sparse_columns_per_group!" begin
        column_indices_per_group = [[1, 4], [2], [3, 5, 6, 7]]
        for row_indices in row_indices_per_case
            destination = fill(1.0, 3, length(row_indices))
            sum_sparse_columns_per_group!(destination, matrix, row_indices, column_indices_per_group)
            for (group_index, column_indices) in enumerate(column_indices_per_group)
                for (position, row_index) in enumerate(row_indices)
                    @test destination[group_index, position] ≈ 1.0 + sum(matrix[row_index, column_indices])
                end
            end
        end
    end
end

@testset "auc" begin
    function reference_folded_auc(feature::AbstractVector{<:Real}, is_in_group::AbstractVector{Bool})::Float64
        in_group = feature[is_in_group]
        out_group = feature[.!is_in_group]
        if isempty(in_group) || isempty(out_group)
            return 0.0  # UNTESTED
        end
        wins = 0.0
        for value_in in in_group, value_out in out_group
            wins += value_in > value_out ? 1.0 : (value_in == value_out ? 0.5 : 0.0)
        end
        auc = wins / (length(in_group) * length(out_group))
        return max(auc, 1 - auc)
    end

    @testset "matches a brute-force AUC" begin
        rng = Random.MersenneTwister(1234)
        for _ in 1:20
            n = rand(rng, 5:40)
            feature = round.(rand(rng, n) .* 6 .- 2)  # includes negatives, zeros and ties
            category = rand(rng, 1:4, n)
            categories, aucs = auc_per_category(feature, category)
            for (group, value) in enumerate(categories)
                @test aucs[group] ≈ reference_folded_auc(feature, category .== value)
            end
        end
    end

    @testset "sparse matches dense" begin
        rng = Random.MersenneTwister(5678)
        for _ in 1:20
            n = rand(rng, 5:60)
            dense_feature = Float64.(rand(rng, -2:4, n))  # many zeros, some negatives
            category = rand(rng, 1:5, n)
            _, dense_aucs = auc_per_category(dense_feature, category)
            _, sparse_aucs = auc_per_category(sparse(dense_feature), category)
            @test sparse_aucs ≈ dense_aucs
        end
    end

    @testset "positive flag matches for non-negative features" begin
        feature = sparse(Float64[0, 0, 3, 1, 0, 4, 2, 0])
        group_index_per_sample = [1, 2, 1, 2, 1, 2, 1, 2]
        n_samples_per_group = [4, 4]
        with_flag = auc_per_group!(;
            auc_per_group = zeros(2),
            feature_per_sample = feature,
            group_index_per_sample,
            n_samples_per_group,
            positive = true,
        )
        without_flag = auc_per_group!(;
            auc_per_group = zeros(2),
            feature_per_sample = feature,
            group_index_per_sample,
            n_samples_per_group,
        )
        @test with_flag ≈ without_flag
    end

    @testset "excluded samples and undefined groups" begin
        # Group 0 excludes samples 4, 5; the population is samples 1..3 across groups 1 and 2.
        feature = Float64[1, 2, 3, 4, 5]
        auc = auc_per_group!(;
            auc_per_group = zeros(2),
            feature_per_sample = feature,
            group_index_per_sample = [1, 1, 2, 0, 0],
            n_samples_per_group = [2, 1],
        )
        @test auc[1] ≈ reference_folded_auc(feature[1:3], [true, true, false])
        @test auc[2] ≈ reference_folded_auc(feature[1:3], [false, false, true])

        # A group covering the whole population is undefined, reported as 0.
        one_group = auc_per_group!(;
            auc_per_group = zeros(1),
            feature_per_sample = Float64[1, 2, 3],
            group_index_per_sample = [1, 1, 1],
            n_samples_per_group = [3],
        )
        @test one_group[1] == 0.0

        # An all-zero feature cannot distinguish anything, so its folded AUC is 0.5 (not the undefined 0).
        flat = auc_per_group!(;
            auc_per_group = zeros(2),
            feature_per_sample = sparse(zeros(6)),
            group_index_per_sample = [1, 1, 1, 2, 2, 2],
            n_samples_per_group = [3, 3],
        )
        @test all(flat .== 0.5)
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

    @test cached_ispath("/")
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
        expected_mean = reference_mean(fixed[is_active_now, :], replaced_variable[is_active_now, :])
        @test mean_correlation(grouped) ≈ expected_mean atol = 1e-5
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

    @testset "is_in_parallel_loop is true only inside a threaded loop" begin
        @test !is_in_parallel_loop()
        n_items = 2 * Threads.nthreads()
        is_in_loop_per_item = fill(false, n_items)
        parallel_loop_wo_rng(1:n_items; policy = :static) do index
            is_in_loop_per_item[index] = is_in_parallel_loop()
            return nothing
        end
        @test all(is_in_loop_per_item) == (Threads.nthreads() > 1)
        @test !is_in_parallel_loop()
    end

    @testset "debug progress exists only when its group is debugged" begin
        withenv("JULIA_DEBUG" => "") do
            @test DebugProgress(3; group = :tlu_test) === nothing
            @test DebugProgressUnknown(; group = :tlu_test) === nothing
        end
        withenv("JULIA_DEBUG" => "tlu_other") do
            @test DebugProgress(3; group = :tlu_test) === nothing
        end
        withenv("JULIA_DEBUG" => "tlu_test") do
            @test DebugProgress(3; group = :tlu_test) isa Progress
            @test DebugProgressUnknown(; group = :tlu_test) isa ProgressUnknown
        end
        withenv("JULIA_DEBUG" => "tlu_other") do
            with_logger(ConsoleLogger(stderr, Logging.Debug)) do
                @test DebugProgress(3; group = :tlu_test) isa Progress
            end
        end
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

    @testset "parallel_foreach_column" begin
        values = Float64[row + column / 10 for row in 1:7, column in 1:5]
        expected_sums = vec(sum(values; dims = 1))
        sparse_values = sparse(Float64[row == column || row == 7 ? row + column / 10 : 0 for row in 1:7, column in 1:5])
        expected_sparse_sums = vec(sum(sparse_values; dims = 1))

        @testset "dense columns are views of the storage" begin
            @test column_sums(values) == expected_sums
            @test column_sums(read_only_array(values)) == expected_sums
            is_view_per_column = fill(false, 5)
            parallel_foreach_column(read_only_array(values)) do _, column_index, column
                is_view_per_column[column_index] = parent(column) === values
                return nothing
            end
            @test all(is_view_per_column)
        end

        @testset "sparse columns are views of the storage" begin
            @test column_sums(sparse_values) == expected_sparse_sums
            is_view_per_column = fill(false, 5)
            parallel_foreach_column(sparse_values) do _, column_index, column
                is_view_per_column[column_index] = parent(nonzeros(column)) === nonzeros(sparse_values)
                return nothing
            end
            @test all(is_view_per_column)
        end

        @testset "column subsets are visited by position" begin
            @test column_sums(values; columns = [4, 2]) == expected_sums[[4, 2]]
            @test column_sums(sparse_values; columns = [5, 1, 3]) == expected_sparse_sums[[5, 1, 3]]
        end

        @testset "strided views are kept unless contiguous" begin
            strided_values = view(values, 1:2:7, :)
            expected_strided_sums = vec(sum(strided_values; dims = 1))
            @test column_sums(strided_values) == expected_strided_sums
            @test column_sums(strided_values; contiguous = true) == expected_strided_sums
            is_vector_per_column = fill(false, 5)
            parallel_foreach_column(strided_values; contiguous = true) do _, column_index, column
                is_vector_per_column[column_index] = column isa Vector{Float64}
                return nothing
            end
            @test all(is_vector_per_column)
        end

        @testset "rows are read against the grain" begin
            transposed_values = transpose(Matrix(transpose(values)))
            transposed_sparse_values = transpose(SparseMatrixCSC(transpose(sparse_values)))
            @test_throws "inefficient major axis: Rows" column_sums(transposed_values)
            TanayLabUtilities.MatrixLayouts.GLOBAL_INEFFICIENT_ACTION_HANDLER = IgnoreHandler
            try
                @test column_sums(transposed_values) == expected_sums
                @test column_sums(transposed_values; contiguous = true) == expected_sums
                @test column_sums(transposed_sparse_values) == expected_sparse_sums
            finally
                TanayLabUtilities.MatrixLayouts.GLOBAL_INEFFICIENT_ACTION_HANDLER = ErrorHandler
            end
        end

        @testset "lazy matrices are read through one scratch per thread" begin
            lazy_values = LazyColumnsMatrix(values, Threads.Atomic{Int}(0))
            @test column_sums(lazy_values) == expected_sums
            @test lazy_values.n_scratches[] == (Threads.nthreads() == 1 ? 1 : Threads.maxthreadid())
            lazy_values.n_scratches[] = 0
            @test column_sums(lazy_values; policy = :serial) == expected_sums
            @test lazy_values.n_scratches[] == 1
        end

        @testset "nested loops run serially and report only when they allocate" begin
            n_outer = 2 * Threads.nthreads()
            is_reported_per_outer = fill(false, n_outer)
            sums_per_outer = Vector{Vector{Float64}}(undef, n_outer)
            parallel_loop_wo_rng(1:n_outer; policy = :static) do outer_index
                sums_per_outer[outer_index] = column_sums(values)
                try
                    column_sums(LazyColumnsMatrix(values, Threads.Atomic{Int}(0)))
                catch exception
                    is_reported_per_outer[outer_index] = contains(exception.msg, "allocating column scratch")
                end
                return nothing
            end
            @test all(sums == expected_sums for sums in sums_per_outer)
            @test all(is_reported_per_outer) == (Threads.nthreads() > 1)
        end

        @testset "nothing is allocated per column" begin
            wide_values = rand(7, 200)
            sums = zeros(200)
            for matrix in (wide_values, LazyColumnsMatrix(wide_values, Threads.Atomic{Int}(0)))
                sum_columns!(sums, matrix, 1:10)
                sum_columns!(sums, matrix, 1:200)
                n_few_bytes = @allocated sum_columns!(sums, matrix, 1:10)
                n_many_bytes = @allocated sum_columns!(sums, matrix, 1:200)
                @test n_many_bytes - n_few_bytes < 1000
            end
        end
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

@testset "downsample" begin
    @testset "dense vector matches sparse" begin
        dense = [0, 5, 0, 0, 3, 7, 0, 1]
        @test downsample(dense, 10; rng = Random.Xoshiro(7)) == downsample(sparse(dense), 10; rng = Random.Xoshiro(7))
    end

    @testset "sparse vector keeps zeros and the total" begin
        dense = [0, 5, 0, 0, 3, 7, 0, 1]
        for _ in 1:100
            downsampled = downsample(sparse(dense), 10)
            @test downsampled isa Vector{Int}
            @test sum(downsampled) == 10
            @test all(0 .<= downsampled .<= dense)
        end
        @test downsample(sparse(dense), 100) == dense
        @test downsample(sparse([0, 4, 0]), 2) == [0, 2, 0]
        @test downsample(sparse([0, 0, 0]), 2) == [0, 0, 0]
    end

    @testset "sparse vector samples with the counts as weights" begin
        dense = [0, 1, 0, 3]
        n_draws = 20_000
        total_per_value = zeros(Int, 4)
        for _ in 1:n_draws
            total_per_value .+= downsample(sparse(dense), 1)
        end
        @test all(abs.(total_per_value ./ n_draws .- dense ./ 4) .< 0.02)
    end

    @testset "sparse matrix keeps zeros and the totals" begin
        dense = [0 5 0; 2 0 3; 0 4 0; 6 0 9]
        for (matrix, dims, samples_per_slice) in (
            (sparse(dense), Columns, vec(sum(dense; dims = 1))),
            (flip(sparse(flip(dense))), Rows, vec(sum(dense; dims = 2))),
        )
            downsampled = downsample(matrix, 4; dims)
            @test all(0 .<= downsampled .<= dense)
            @test vec(sum(downsampled; dims = other_axis(dims))) == min.(samples_per_slice, 4)
        end
    end
end

@testset "round_counts" begin
    n_draws = 20_000

    @testset "a single unit is placed with the fractions as probabilities" begin
        values = [0.25, 0.25, 0.5]
        total_per_value = zeros(Int, 3)
        for _ in 1:n_draws
            rounded = round_counts(values)
            @test sum(rounded) == 1
            @test all((rounded .== 0) .| (rounded .== 1))
            total_per_value .+= rounded
        end
        @test all(abs.(total_per_value ./ n_draws .- values) .< 0.02)
    end

    @testset "expected values hold when the fractions sum to more than one" begin
        values = [0.9, 0.9, 0.2]
        total_per_value = zeros(Int, 3)
        for _ in 1:n_draws
            rounded = round_counts(values)
            @test sum(rounded) == 2
            total_per_value .+= rounded
        end
        @test all(abs.(total_per_value ./ n_draws .- values) .< 0.02)
    end

    @testset "integer parts are kept and the total is preserved" begin
        values = [1.5, 2.25, 0.25, 3.0]
        for _ in 1:100
            rounded = round_counts(values)
            @test sum(rounded) == 7
            @test all((rounded .== floor.(values)) .| (rounded .== ceil.(values)))
        end
    end

    @testset "a fractional total rounds to its floor or ceil" begin
        values = [0.3, 0.4]
        total = 0
        for _ in 1:n_draws
            rounded_total = sum(round_counts(values))
            @test rounded_total in (0, 1)
            total += rounded_total
        end
        @test abs(total / n_draws - 0.7) < 0.02
    end

    @testset "integer input is copied" begin
        values = [3, 0, 5, 1]
        @test round_counts(values) == values
        @test round_counts([2.0, 0.0, 4.0]) == [2, 0, 4]
    end

    @testset "trivial inputs" begin
        @test round_counts(Float64[]) == Int[]
        @test round_counts([0.0, 0.0]) == [0, 0]
        @test round_counts([3.0]) == [3]
        @test round_counts([0.5]) in ([0], [1])
    end

    @testset "negative input is rejected" begin
        @test_throws AssertionError round_counts([1.5, -0.5])
    end

    @testset "given output is filled" begin
        output = fill(-1, 3)
        @test round_counts([1.0, 2.0, 3.0]; output) === output
        @test output == [1, 2, 3]

        output = fill(-1, 2, 2)
        @test round_counts([1.0 2.0; 3.0 4.0]; dims = Columns, output) === output
        @test output == [1 2; 3 4]
    end

    @testset "sparse vector matches dense" begin
        dense = [0.0, 1.5, 0.0, 0.25, 2.75, 0.0, 0.5]
        @test round_counts(sparse(dense); rng = Random.Xoshiro(7)) == round_counts(dense; rng = Random.Xoshiro(7))
    end

    @testset "sparse matrix matches dense" begin
        dense = [0.0 1.5 0.0; 0.25 0.0 2.75; 0.5 0.0 0.0; 0.0 3.25 0.75]
        @test round_counts(sparse(dense); dims = Columns, rng = Random.Xoshiro(7)) ==
              round_counts(dense; dims = Columns, rng = Random.Xoshiro(7))
        @test round_counts(flip(sparse(dense)); dims = Rows, rng = Random.Xoshiro(7)) ==
              round_counts(flip(dense); dims = Rows, rng = Random.Xoshiro(7))
    end
end

@testset "matrix_formats" begin
    using NamedArrays

    @testset "brief of a view of a sparse matrix" begin
        sparse_matrix = sparse([0 1 0; 2 0 3])
        @test brief(view(sparse_matrix, [2, 1], :)) == "2 x 3 x Int64 in Columns (Sparse ? [Int64])"
        @test brief(view(sparse_matrix, 1:2, :)) == "2 x 3 x Int64 in Columns (Sparse ? [Int64])"
        @test brief(view(sparse_matrix, :, 3)) == "2 x Int64 (Sparse 1 (50%) [Int64])"
    end

    matrix = sparse([0 1 0; 2 0 3])

    @testset "sparse queries see through wrappers" begin
        read_only = read_only_array(matrix)
        @test nnz(read_only) == 3
        @test SparseArrays.indtype(read_only) == SparseArrays.indtype(matrix)

        read_only_columns = view(read_only, :, 2:3)
        @test nnz(read_only_columns) == 2
        @test SparseArrays.indtype(read_only_columns) == SparseArrays.indtype(matrix)

        named = NamedArray(matrix)
        @test nnz(named) == 3
        @test SparseArrays.indtype(named) == SparseArrays.indtype(matrix)
    end

    @testset "base_array unwraps to the array holding the data" begin
        dense = [1 2; 3 4]
        @test base_array(dense) === dense
        @test base_array(NamedArray(dense)) === dense

        values = [1, 2, 3]
        @test base_array(SparseArrays.ReadOnly(values)) === values

        column = view(dense, :, 1)
        @test base_array(column) === column

        read_only_column = base_array(view(SparseArrays.ReadOnly(dense), :, 1))
        @test parent(read_only_column) === dense
        @test read_only_column == [1, 3]
    end

    @testset "sparse_matrix_csc of a wrapped sparse matrix doesn't read it element by element" begin
        large = sparse(Int32[1, 50_000], Int32[1, 5_000], [1.0, 2.0], 50_000, 5_000)
        wrapped = NamedArray(read_only_array(large))
        sparse_matrix_csc(wrapped)
        elapsed = @elapsed copied = sparse_matrix_csc(wrapped)
        @test copied == large
        @test copied !== large
        @test SparseArrays.indtype(copied) == Int32
        @test elapsed < 0.1
    end

    @testset "sparse_vector of a wrapped sparse vector doesn't read it element by element" begin
        large = sparsevec(Int32[1, 100_000_000], [1.0, 2.0], 100_000_000)
        wrapped = NamedArray(read_only_array(large))
        sparse_vector(wrapped)
        elapsed = @elapsed copied = sparse_vector(wrapped)
        @test copied == large
        @test copied !== large
        @test SparseArrays.indtype(copied) == Int32
        @test elapsed < 0.1
    end

    @testset "sparse_matrix_csc of a lazy matrix reads its plain storage" begin
        counting = CountingMatrix([0.0 1.0; 2.0 0.0], Ref(0))
        @test sparse_matrix_csc(counting) == [0.0 1.0; 2.0 0.0]
        @test counting.n_reads[] == 0
    end

    @testset "brief of a Bool array reads it once" begin
        counting = CountingBools([true, false, true], Ref(0))
        brief(counting)
        @test counting.n_reads[] == 3
    end

    @testset "plain_array" begin
        dense = Float64[1 2 3; 4 5 6]
        strings = ["a" "b"; "c" "d"]
        sparse_matrix = sparse(dense)

        @testset "plain is itself" begin
            @test plain_array(dense) === dense
            @test plain_array(sparse_matrix) === sparse_matrix
            @test plain_array(sparse(Float64[0, 1])) isa SparseVector
        end

        @testset "wrappers of plain are unwrapped" begin
            @test plain_array(read_only_array(dense)) === dense
            @test plain_array(NamedArray(read_only_array(sparse_matrix))) === sparse_matrix
        end

        @testset "transposes are rewrapped, not copied" begin
            for transposed in (transpose(dense), PermutedDimsArray(dense, (2, 1)), read_only_array(flip(dense)))
                result = plain_array(transposed)
                @test result isa Transpose
                @test parent(result) === dense
                result = plain_array(transposed; transposed_as = PermutedDimsArray)
                @test result isa PermutedDimsArray
                @test parent(result) === dense
            end
            result = plain_array(read_only_array(flip(sparse_matrix)); transposed_as = Adjoint)
            @test result isa Adjoint
            @test parent(result) === sparse_matrix
            result = plain_array(PermutedDimsArray(dense, (2, 1)); transposed_as = Transpose)
            @test result isa Transpose
            @test parent(result) === dense
            result = plain_array(PermutedDimsArray(strings, (2, 1)))
            @test result isa PermutedDimsArray
            @test parent(result) === strings
            @test plain_array(transpose(PermutedDimsArray(dense, (2, 1)))) === dense
        end

        @testset "strided views are kept" begin
            for strided in (view(dense, :, 1:2:3), view(dense, :, 1:2), view(read_only_array(dense), :, 1:2:3))
                result = plain_array(strided)
                @test result isa SubArray
                @test parent(result) === dense
                @test result == strided
            end
        end

        @testset "contiguous copies only views with gaps" begin
            contiguous_view = view(dense, :, 1:2)
            @test plain_array(contiguous_view; contiguous = true) === contiguous_view
            gapped_view = view(dense, :, 1:2:3)
            result = plain_array(gapped_view; contiguous = true)
            @test result isa Matrix{Float64}
            @test result == gapped_view
            result = plain_array(transpose(gapped_view); contiguous = true)
            @test result isa Transpose
            @test parent(result) isa Matrix{Float64}
            @test parent(result) == gapped_view
        end

        @testset "other arrays are copied" begin
            result = plain_array(falses(2))
            @test result isa Vector{Bool}
            @test result == [false, false]
        end

        @testset "views of other arrays copy only the view" begin
            counting = CountingMatrix(dense, Ref(0))
            result = plain_array(view(counting, :, 2))
            @test result == [2.0, 5.0]
            @test counting.n_reads[] == 2
        end

        @testset "sparse views are copied" begin
            sparse_columns = view(sparse_matrix, :, 1:2:3)
            result = plain_array(sparse_columns)
            @test result isa SparseMatrixCSC
            @test result == sparse_columns
        end
    end

    @testset "reinterpreted_array" begin
        @testset "shares the bytes" begin
            bytes = Vector{UInt8}(reinterpret(UInt8, Float64[1, 2, 3, 4]))
            matrix = reinterpreted_array(Float64, bytes, (2, 2))
            @test matrix == [1.0 3.0; 2.0 4.0]
            matrix[1, 1] = 5
            @test reinterpret(Float64, bytes)[1] == 5
        end

        @testset "offset" begin
            bytes = Vector{UInt8}(reinterpret(UInt8, Float64[1, 2, 3, 4]))
            @test reinterpreted_array(Float64, bytes, 2; offset = 16) == [3.0, 4.0]
            @test reinterpreted_array(Float64, view(bytes, 17:32), 2) == [3.0, 4.0]
        end

        @testset "empty" begin
            @test reinterpreted_array(Float64, UInt8[], 0) == Float64[]
        end

        @testset "keeps the bytes alive" begin
            function collect_garbage()::Nothing
                for _ in 1:5
                    GC.gc(true)
                end
                return nothing
            end

            # Each reference is dropped by returning from the function holding it. Setting a variable to `nothing`
            # inside a running function doesn't reliably drop the old value.

            # Only a `reshape` of the result survives this.
            @noinline function wrapped_bytes(; is_view::Bool)::Tuple{Matrix{Float64}, WeakRef}
                bytes = Vector{UInt8}(reinterpret(UInt8, Float64.(0:1000)))
                if is_view
                    vector = reinterpreted_array(Float64, view(bytes, 9:length(bytes)), 1000)
                else
                    vector = reinterpreted_array(Float64, bytes, 1000; offset = 8)
                end
                return (reshape(vector, 10, 100), WeakRef(bytes))
            end

            # Whether the bytes stayed alive while the `reshape` was used. It is dropped when this returns.
            @noinline function used_then_dropped(; is_view::Bool, is_collecting::Bool)::Tuple{Bool, WeakRef}
                reshaped_matrix, weak_bytes = wrapped_bytes(; is_view)
                if is_collecting
                    collect_garbage()
                end
                is_alive = weak_bytes.value !== nothing && vec(reshaped_matrix) == Float64.(1:1000)
                return (is_alive, weak_bytes)
            end

            for is_view in (false, true)
                @testset "is_view: $(is_view)" begin
                    is_alive, weak_bytes = used_then_dropped(; is_view, is_collecting = true)
                    @test is_alive
                    collect_garbage()
                    @test weak_bytes.value === nothing
                end
            end

            @testset "threads" begin
                weak_bytes_per_task = Vector{WeakRef}(undef, 1000)
                Threads.@threads for task_index in 1:1000
                    is_alive, weak_bytes = used_then_dropped(; is_view = isodd(task_index), is_collecting = false)
                    @assert is_alive
                    weak_bytes_per_task[task_index] = weak_bytes
                end
                collect_garbage()
                @test all(weak_bytes.value === nothing for weak_bytes in weak_bytes_per_task)
            end
        end

        @testset "rejects bad input" begin
            bytes = zeros(UInt8, 16)
            @test_throws AssertionError reinterpreted_array(Float64, bytes, 3)
            @test_throws AssertionError reinterpreted_array(Float64, bytes, 1; offset = 1)
            @test_throws AssertionError reinterpreted_array(String, bytes, 1)
        end
    end
end

@testset "read_only_arrays" begin
    using NamedArrays

    @testset "is_read_only_array does not read the data" begin
        costly = CostlyReadOnly([1, 2, 3], Ref(0))
        @test is_read_only_array(costly)
        @test is_read_only_array(NamedArray(costly))
        @test is_read_only_array(view(costly, 1:2))
        @test costly.n_reads[] == 0
        @test costly == [1, 2, 3]
        @test costly.n_reads[] == 3
    end
end

@testset "matrix_layouts" begin
    @test major_axis(falses(2, 3)) == Columns

    # Relayout the `source` with the given budget and minimal write sizes, so large matrices are relayouted in bands.
    function relayout_with(source::AbstractMatrix; budget_kb::Integer, min_write_kb::Integer = 1)::AbstractMatrix
        saved_budget_kb = TanayLabUtilities.MatrixLayouts.RELAYOUT_BUDGET_KB
        saved_min_write_kb = TanayLabUtilities.MatrixLayouts.RELAYOUT_MIN_WRITE_KB
        TanayLabUtilities.MatrixLayouts.RELAYOUT_BUDGET_KB = budget_kb
        TanayLabUtilities.MatrixLayouts.RELAYOUT_MIN_WRITE_KB = min_write_kb
        try
            n_rows, n_columns = size(source)
            if issparse(source)
                colptr = fill(eltype(source.colptr)(nnz(source) + 1), n_rows + 1)
                colptr[1] = 1
                destination = SparseMatrixCSC(
                    n_columns,
                    n_rows,
                    colptr,
                    ones(eltype(source.rowval), nnz(source)),
                    zeros(eltype(source), nnz(source)),
                )
            else
                destination = similar(source, n_columns, n_rows)
            end
            return relayout!(flip(destination), source)
        finally
            TanayLabUtilities.MatrixLayouts.RELAYOUT_BUDGET_KB = saved_budget_kb
            TanayLabUtilities.MatrixLayouts.RELAYOUT_MIN_WRITE_KB = saved_min_write_kb
        end
    end

    # Whether the `relayout` of the `source` has the same data as `transpose!`, down to the stored arrays.
    function is_same_as_transpose(relayout::AbstractMatrix, source::AbstractMatrix)::Bool
        expected = copy(transpose(source))
        actual = parent(relayout)
        if issparse(source)
            return actual.colptr == expected.colptr &&
                   actual.rowval[1:nnz(source)] == expected.rowval &&
                   actual.nzval[1:nnz(source)] == expected.nzval
        else
            return actual == expected
        end
    end

    @testset "sparse relayout in bands" begin
        rng = Random.Xoshiro(123456)

        @testset "one task and many bands" begin
            source = SparseMatrixCSC{Float32, Int64}(sprand(rng, Float32, 200, 300, 0.1))
            @test is_same_as_transpose(relayout_with(source; budget_kb = 1), source)
        end

        @testset "several tasks" begin
            source = SparseMatrixCSC{Float32, Int64}(sprand(rng, Float32, 200, 300, 0.1))
            @test is_same_as_transpose(relayout_with(source; budget_kb = 64), source)
        end

        @testset "narrow bands of Int32 indices" begin
            source = SparseMatrixCSC{Bool, Int32}(sprand(rng, Bool, 3000, 1_100_000, 2e-6))
            @test is_same_as_transpose(relayout_with(source; budget_kb = 1), source)
        end

        @testset "a column larger than a band" begin
            source = sprand(rng, Float64, 50, 2000, 0.05)
            source[7, :] .= 1.0
            @test is_same_as_transpose(relayout_with(source; budget_kb = 1), source)
        end

        @testset "empty columns" begin
            source = sprand(rng, Float64, 200, 300, 0.1)
            source[1:50, :] .= 0.0
            dropzeros!(source)
            @test is_same_as_transpose(relayout_with(source; budget_kb = 1), source)
        end

        @testset "a single row" begin
            source = sprand(rng, Float64, 1, 5000, 0.5)
            @test is_same_as_transpose(relayout_with(source; budget_kb = 1), source)
        end

        @testset "a single column" begin
            source = sprand(rng, Float64, 5000, 1, 0.5)
            @test is_same_as_transpose(relayout_with(source; budget_kb = 1), source)
        end

        @testset "an empty matrix" begin
            source = spzeros(Float64, 0, 10)
            @test is_same_as_transpose(relayout_with(source; budget_kb = 1), source)
        end
    end

    @testset "dense relayout in bands" begin
        rng = Random.Xoshiro(123456)

        @testset "one task and many bands" begin
            source = rand(rng, Float32, 37, 1000)
            @test is_same_as_transpose(relayout_with(source; budget_kb = 1), source)
        end

        @testset "several tasks" begin
            source = rand(rng, Float32, 37, 1000)
            @test is_same_as_transpose(relayout_with(source; budget_kb = 64), source)
        end

        @testset "a column larger than a band" begin
            source = rand(rng, 1:100, 3, 20000)
            @test is_same_as_transpose(relayout_with(source; budget_kb = 1), source)
        end

        @testset "a single row" begin
            source = rand(rng, Float64, 1, 5000)
            @test is_same_as_transpose(relayout_with(source; budget_kb = 1), source)
        end

        @testset "a single column" begin
            source = rand(rng, Float64, 5000, 1)
            @test is_same_as_transpose(relayout_with(source; budget_kb = 1), source)
        end
    end
end

@testset "mostly_read_write_locks" begin
    import ConcurrentUtils

    @testset "trylock succeeds only when there is no reader" begin
        lock = MostlyReadWriteLock()
        @test trylock(lock)
        unlock(lock)

        ConcurrentUtils.lock_read(lock)
        @test !trylock(lock)
        ConcurrentUtils.unlock_read(lock)
    end

    @testset "trylock fails while another task holds the lock" begin
        lock = MostlyReadWriteLock()
        Base.lock(lock)
        @test !fetch(Threads.@spawn trylock(lock))
        unlock(lock)
    end

    @testset "trylock_read succeeds only when there is no writer" begin
        lock = MostlyReadWriteLock()
        @test ConcurrentUtils.trylock_read(lock)
        ConcurrentUtils.unlock_read(lock)
        @test ConcurrentUtils.trylock_read(() -> 1, lock) == Some(1)

        Base.lock(lock)
        @test !ConcurrentUtils.trylock_read(lock)
        @test ConcurrentUtils.trylock_read(() -> 1, lock) === nothing
        unlock(lock)
    end
end

@testset "logged" begin
    @test logged_sum(1; y = 2) == 3
end

@testset "documented" begin
    @test documented_untyped(1) == 2
    @test function_default(documented_untyped, :y) == 1

    # A default which refers to other arguments isn't stashed, but still applies when the function is called.
    @test documented_dependent(1) == 2
    @test function_default(documented_dependent, :y) == 1
    @test_throws "no parameter (with default): z" function_default(documented_dependent, :z)

    # A default which fails for another reason still fails the definition.
    @test_throws "bad default" @eval @documented function documented_failing(; x::Int = error("bad default"))::Int
        return x
    end
end

@testset "parallel_storage" begin
    @test reset_reusable_storage!(1) === nothing
end

@testset "kmeans" begin
    # Two well separated clouds of five points each, so every correct clustering splits them the same way.
    values_of_points = Float64[
        0.0 0.1 0.2 0.1 0.0 10.0 10.1 10.2 10.1 10.0
        0.0 0.1 0.0 0.2 0.1 10.0 10.1 10.0 10.2 10.1
    ]

    function is_split_by_cloud(assignments::AbstractVector{<:Integer})::Bool
        return allequal(assignments[1:5]) && allequal(assignments[6:10]) && assignments[1] != assignments[6]
    end

    function make_buffers()::KMeansBuffers
        return KMeansBuffers{Float64}(; n_dims = 2, max_k = 4, n_points = 20)
    end

    @testset "kmeans_in_buffers" begin
        @testset "without buffers" begin
            result = kmeans_in_buffers(values_of_points, 2; rng = Random.Xoshiro(1))
            @test is_split_by_cloud(result.assignments)
        end

        @testset "with buffers" begin
            buffers = KMeansBuffers(make_buffers(); n_dims = 2, k = 2, n_points = 10)
            result = kmeans_in_buffers(values_of_points, 2; buffers, rng = Random.Xoshiro(1))
            @test result isa KmeansResultView
            @test is_split_by_cloud(result.assignments)
            @test sort(collect(result.counts)) == [5, 5]
        end
    end

    @testset "kmeans_in_buffers!" begin
        @testset "without buffers" begin
            centers = Float64[0.0 10.0; 0.0 10.0]
            result = kmeans_in_buffers!(values_of_points, centers; rng = Random.Xoshiro(1))
            @test is_split_by_cloud(result.assignments)
        end

        @testset "repicks a center with no points" begin
            buffers = KMeansBuffers(make_buffers(); n_dims = 2, k = 2, n_points = 10)
            centers = Float64[0.0 100.0; 0.0 100.0]
            result = kmeans_in_buffers!(values_of_points, centers; buffers, rng = Random.Xoshiro(1))
            @test is_split_by_cloud(result.assignments)
        end

        @testset "one cluster per point" begin
            two_points = values_of_points[:, [1, 6]]
            buffers = KMeansBuffers(make_buffers(); n_dims = 2, k = 2, n_points = 2)
            result = kmeans_in_buffers!(two_points, zeros(2, 2); buffers, rng = Random.Xoshiro(1))
            @test result.assignments == [1, 2]
            @test result.centers == two_points
            @test result.totalcost == 0
        end

        @testset "a single cluster" begin
            buffers = KMeansBuffers(make_buffers(); n_dims = 2, k = 1, n_points = 10)
            result = kmeans_in_buffers!(values_of_points, zeros(2, 1); buffers, rng = Random.Xoshiro(1))
            @test all(==(1), result.assignments)
            @test vec(result.centers) ≈ vec(sum(values_of_points; dims = 2) ./ 10)
        end
    end

    @testset "kmeans_in_rounds" begin
        @testset "default" begin
            result = kmeans_in_rounds(values_of_points, 2; rounds = 3, rng = Random.Xoshiro(1))
            @test is_split_by_cloud(result.assignments)
        end

        @testset "buffers" begin
            result = kmeans_in_rounds(
                values_of_points,
                2;
                buffers = (make_buffers(), make_buffers()),
                rounds = 3,
                rng = Random.Xoshiro(1),
            )
            @test is_split_by_cloud(result.assignments)
        end

        @testset "buffer_pool" begin
            buffer_pool = Channel{KMeansBuffers}(Threads.nthreads())
            for _ in 1:Threads.nthreads()
                put!(buffer_pool, make_buffers())
            end
            result = kmeans_in_rounds(values_of_points, 2; buffer_pool, rounds = 3, rng = Random.Xoshiro(1))
            @test is_split_by_cloud(result.assignments)

            centers = Float64[0.0 10.0; 0.0 10.0]
            result = kmeans_in_rounds(values_of_points, 2; centers, buffer_pool, rounds = 2, rng = Random.Xoshiro(1))
            @test is_split_by_cloud(result.assignments)
        end

        @testset "centers" begin
            centers = Float64[0.0 10.0; 0.0 10.0]
            result = kmeans_in_rounds(
                values_of_points,
                2;
                centers,
                buffers = (make_buffers(), make_buffers()),
                rounds = 2,
                rng = Random.Xoshiro(1),
            )
            @test is_split_by_cloud(result.assignments)
        end

        @testset "min_size" begin
            result = kmeans_in_rounds(values_of_points, 2; min_size = 3, rounds = 3, rng = Random.Xoshiro(1))
            @test is_split_by_cloud(result.assignments)
        end
    end
end
