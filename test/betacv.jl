using Base.Test
using FluxMO.Validation

# http://swl.htwsaar.de/lehre/ss17/ml/slides/2017-vl-ml-ch4-1-clustering.pdf
function naive_intra_cluster_weights(C::AbstractArray)
    W_in = 0.0
    N_in = 0
    for (i,c) in enumerate(C)
        W_in += weights(c,c)
        N_in += length(c) * (length(c)-1)
    end
    0.5 * W_in, convert(Int64, 0.5 * N_in)
end

# http://swl.htwsaar.de/lehre/ss17/ml/slides/2017-vl-ml-ch4-1-clustering.pdf
function naive_n_in(C::AbstractArray)
    N_in = 0
    for (i,c) in enumerate(C)
        N_in += length(c) * (length(c)-1)
    end
    convert(Int64, 0.5 * N_in)
end

# http://swl.htwsaar.de/lehre/ss17/ml/slides/2017-vl-ml-ch4-1-clustering.pdf
function naive_inter_cluster_weights(C::AbstractArray)
    W_out = 0.0
    N_out = 0
    for (i,S) in enumerate(C)
        for (j,R) in enumerate(C)
            if i != j
                W_out += weights(S,R)
                N_out += length(S) * length(R)
            end
        end
    end
    0.5 * W_out, convert(Int64, 0.5 * N_out)
end

# http://swl.htwsaar.de/lehre/ss17/ml/slides/2017-vl-ml-ch4-1-clustering.pdf
function naive_n_out(C::AbstractArray)
    N_out = 0
    for (i, S) in enumerate(C)
        for (j, R) in enumerate(C)
            if i != j
                N_out += length(S) * length(R)
            end
        end
    end
    convert(Int64, 0.5 * N_out)
end


@testset "naive vs optimized for loops" begin

    # low BetaCV
    C = [
        [[1.0, 1.0], [1.1, 1.1]], # 1
        [[2.0, 2.0], [2.1, 2.1]], # 2
        [[3.0, 3.0], [3.1, 3.1]], # 3
        [[4.0, 4.0], [4.1, 4.1]], # 4
        [[5.0, 5.0], [5.1, 5.1]], # 5
        [[6.0, 6.0], [6.1, 6.1]], # 6
    ]

    # intra
    expected = naive_intra_cluster_weights(C)
    result   = intra_cluster_weights(C)
    @test expected[1] ≈ result[1]
    @test expected[2] ≈ result[2]
    @test naive_n_in(C) == 6
    @test naive_n_in(C) == result[2]
    
    # inter
    expected = naive_inter_cluster_weights(C)    
    result   = inter_cluster_weights(C)
    @test expected[1] ≈ result[1]
    @test expected[1] ≈ 197.98989873223329
    @test expected[2] == result[2]
    @test expected[2] == 60
    @test naive_n_out(C) == 60

    @test betacv(C) ≈ 0.0428571428571428
    for _ in 1:10
        @time betacv(C)
    end

    @test betacv_fused(C) ≈ 0.0428571428571428
    for _ in 1:10
        @time betacv_fused(C)
    end

    # high BetaCV
    C = [
        [[1.0, 1.0], [6.1, 6.1]], # 1
        [[2.0, 2.0], [5.1, 5.1]], # 2
        [[3.0, 3.0], [4.1, 4.1]], # 3
        [[4.0, 4.0], [3.1, 3.1]], # 4
        [[5.0, 5.0], [2.1, 2.1]], # 5
        [[6.0, 6.0], [1.1, 1.1]], # 6
    ]

    @test betacv(C) ≈ 1.4681892332789561
    #               > 0.0428571428571428

end


@testset "tracked betacv with for loops" begin
    using Flux.Tracker

    C = [
        [param([1.0, 1.0]), param([1.1, 1.1])], # 1
        [param([2.0, 2.0]), param([2.1, 2.1])], # 2
        [param([3.0, 3.0]), param([3.1, 3.1])], # 3
        [param([4.0, 4.0]), param([4.1, 4.1])], # 4
        [param([5.0, 5.0]), param([5.1, 5.1])], # 5
        [param([6.0, 6.0]), param([6.1, 6.1])], # 6
    ]

    # intra
    expected = naive_intra_cluster_weights(C)
    result   = intra_cluster_weights(C)
    @test expected[1] ≈ result[1]
    @test expected[1].tracker.data ≈ 0.8485281374238982
    @test expected[2] == result[2]
    @test expected[2] == 6
    
    # inter
    expected = naive_inter_cluster_weights(C)
    result   = inter_cluster_weights(C)
    @test expected[1] ≈ result[1]
    @test expected[1] ≈ 197.98989873223329
    @test expected[2] == result[2]
    @test expected[2] == 60
    @test naive_n_out(C) == 60

    @test betacv(C).tracker.data ≈ 0.0428571428571428
    for _ in 1:10
        @time betacv(C)
    end
    @test betacv_fused(C).tracker.data ≈ 0.0428571428571428
    for _ in 1:10
        @time betacv_fused(C)
    end

    C[1][1].grad ≈ [0.0, 0.0]

    back!(betacv(C))

    @test C[1][1].grad[1] ≈ -0.034183673469387735
    @test C[1][1].grad[2] ≈ -0.034183673469387735
    @test C[1][2].grad[1] ≈ 0.03724489795918369
    @test C[1][2].grad[2] ≈ 0.03724489795918369

end


@testset "tracked betacv with distance-matrices" begin
    using Flux.Tracker

    C = [
        [1.0 1.1; 1.0 1.1], # 1
        [2.0 2.1; 2.0 2.1], # 2
        [3.0 3.1; 3.0 3.1], # 3
        [4.0 4.1; 4.0 4.1], # 4
        [5.0 5.1; 5.0 5.1], # 5
        [6.0 6.1; 6.0 6.1], # 6
    ]
    @test typeof(C[1]) == Array{Float64,2}

    C_tracked = [
        [param(1.0) param(1.1); param(1.0) param(1.1)], # 1
        [param(2.0) param(2.1); param(2.0) param(2.1)], # 2
        [param(3.0) param(3.1); param(3.0) param(3.1)], # 3
        [param(4.0) param(4.1); param(4.0) param(4.1)], # 4
        [param(5.0) param(5.1); param(5.0) param(5.1)], # 5
        [param(6.0) param(6.1); param(6.0) param(6.1)], # 6
    ]
    @test typeof(C_tracked[1]) == Array{Flux.Tracker.TrackedReal{Float64},2}

    # intra
    expected = intra_cluster_weights_pairwise(C)
    result   = intra_cluster_weights_pairwise(C_tracked)
    @test expected[1] ≈ result[1]
    @test expected[1] ≈ 0.8485281374238982
    @test expected[2] == result[2]
    @test expected[2] == 6
    
    # inter
    expected = inter_cluster_weights_pairwise(C)
    result   = inter_cluster_weights_pairwise(C_tracked)
    @test expected[1] ≈ result[1]
    @test expected[1] ≈ 197.98989873223329
    @test expected[2] == result[2]
    @test expected[2] == 60
    
    @time expected = betacv_pairwise(C)
    @time result = betacv_pairwise(C_tracked)
    @test expected            ≈ 0.0428571428571428
    @test result.tracker.data ≈ expected

    before = [t.tracker.grad for t in C_tracked[1]]
    @test isapprox(before, [0.0 0.0; 0.0 0.0], atol=1e-20)

    @time back!(betacv_pairwise(C_tracked))
    after = [t.tracker.grad for t in C_tracked[1]]
    @test isapprox(after,
        [-0.034183673469387735 0.03724489795918369;
         -0.034183673469387735 0.03724489795918369],
        atol=1e-20)

    # NOTE: it is not possible to compute the gradient
    # 2×2 Array{Float64,2}:
    # NaN  NaN
    # NaN  NaN

end
