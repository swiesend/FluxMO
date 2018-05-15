using Base.Test
using LogClustering.Clustering

@testset "test soft & hard clustering on small sample data" begin
    data = [
        1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0;
        1.1 2.1 3.1 4.1 5.1 6.1 7.1 8.1 9.1;
        1.2 2.2 3.2 4.2 5.2 6.2 7.2 8.2 9.2;
    ]
    L, N = size(data)

    Cs, used = knn_clustering(data, k = 2, mode=:soft)
    @show length(Cs), Cs
    @test length(Cs) == N
    @show used
    # @test length(used) in 4:N
    @test length(used) == N

    Cs, used = knn_clustering(data, k = 2)
    @show length(Cs), Cs
    @test length(Cs) in 2:N
    @show used
    # @test length(used) in 4:N
    @test length(used) == N

    Vs = map(c->map(i->data[:,i],c),Cs)
    @show Vs
end

@testset "test soft & hard clustering on small diff sample data" begin
    data = [
        1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
        1.0 1.0 1.0 1.0 1.0000003 1.0000004 1.0000005 1.0000006 1.0000007 1.0000008 1.0000009;
    ]
    L, N = size(data)

    Cs, used = knn_clustering(data, k = 4, mode=:soft)
    @show length(Cs), Cs
    @test length(Cs) == N
    @show used
    # @test length(used) in 4:N
    @test length(used) == N

    Cs, used = knn_clustering(data, k = 4)
    @show length(Cs), Cs
    @test length(Cs) in 2:N
    @show used
    # @test length(used) in 4:N
    @test length(used) == N

    Vs = map(c->map(i->data[:,i],c),Cs)
    @show Vs
end


@testset "Validate Clustering with BetaCV" begin
    using LogClustering.Validation

    L = 3
    N = 1000
    data = randn(L,N)
    min = minimum(data)
    max = maximum(data)
    @show min, max

    C, used = knn_clustering(data)

    @show length(C)
    @show length(used) / N
    @show mean(map(length, values(C)))
    @show median(map(length, values(C)))
    
    Vs = map(c->data[:,c],C)
    bcv = @time betacv(Vs)
    @show bcv
    @test bcv > 0.7
    @test bcv < 0.8
end
