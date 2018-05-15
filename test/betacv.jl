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


@testset "naive vs optimized" begin

    # using Clustering
    # low_betacv = [
    #     1.0  1.1  2.0  2.1  3.0  3.1  4.0  4.1  5.0  5.1  6.0  6.1;
    #     1.0  1.1  2.0  2.1  3.0  3.1  4.0  4.1  5.0  5.1  6.0  6.1;
    # ]
    # DM = pairwise(Euclidean(), low, low)
    # C = dbscan(DM, 0.15, 2)
    # ... view(data assignments)

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
    @test expected[2] ≈ result[2]
    @test naive_n_out(C) == 60
    @test naive_n_out(C) == result[2]

    @test betacv(C) ≈ 0.0428571428571428
    @test betacv(C) < (1.4681892332789561 / 10)

    # high BetaCV
    C = [
        [[1.0, 1.0], [6.1, 6.1]], # 1
        [[2.0, 2.0], [5.1, 5.1]], # 2
        [[3.0, 3.0], [4.1, 4.1]], # 3
        [[4.0, 4.0], [3.1, 3.1]], # 4
        [[5.0, 5.0], [2.1, 2.1]], # 5
        [[6.0, 6.0], [1.1, 1.1]], # 6
    ]

    @test betacv(C) > (0.0428571428571428 * 10)
    @test betacv(C) ≈ 1.4681892332789561

end


@testset "Flux.Tracker: betacv (tracked)" begin
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
    @test expected[2] ≈ result[2]
    
    # inter
    expected = naive_inter_cluster_weights(C)
    result   = inter_cluster_weights(C)
    @test expected[1] ≈ result[1]
    @test expected[2] ≈ result[2]
    @test naive_n_out(C) == 60
    @test naive_n_out(C) == result[2]

    @test betacv(C) ≈ 0.0428571428571428

    C[1][1].grad ≈ [0.0, 0.0]

    back!(betacv(C))

    @test C[1][1].grad[1] ≈ -0.034183673469387735
    @test C[1][1].grad[2] ≈ -0.034183673469387735

end
