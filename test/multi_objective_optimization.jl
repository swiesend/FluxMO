# @testset "Deep-Autoencoder with Crossentropy and BetaCV loss" begin
using Flux
using Flux.Tracker
using Flux: crossentropy, throttle

using FluxMO.Clustering: knn_clustering
using FluxMO.Validation: betacv

# @testset "Deep-Autoencoder with Crossentropy and BetaCV loss" begin

function train(seed::Int)
    srand(seed)

    N = 10000   # samples
    M = 100     # data size
    L = 2       # latent space

    D = map(_->rand(M),1:N)
    X = deepcopy(D[1:end-1])
    Y = deepcopy(D[2:end])

    a = tanh
    m = Chain(
        Dense(M,   100, a),
        Dense(100, 10,  a),
        Dense(10,  L,   a),
        Dense(L,   10,  a),
        Dense(10,  100, a),
        Dense(100, M,   sigmoid),
    )
    el = 3

    function supervised_betacv()
        # Embed all samples from X into the latent space of size L
        embds_tracked = map(x-> m[1:el](x), X)
        embds  = hcat(map(ta->ta.data, embds_tracked)...)
        
        # Cluster the current embeddings without tracking to generate
        # a supervised scenario.
        # This clould also be done by DBSCAN, OPTICS, K-Means ...
        clustering, _ = knn_clustering(embds)

        # Obtain tracked embedded values from the clustering
        values_tracked = map(c->map(i->embds_tracked[i],c), clustering)

        # Validate the clustering via BetaCV measure (small is good).
        # This is done the tracked values, as it should influence
        # the model weights and biases towards optimizing this measure.
        bcv = betacv(values_tracked)

        bcv
    end

    function loss(x, y)
        # unsupervised
        ŷ = m(x)
        ce = crossentropy(ŷ, y)

        # supervised
        bcv = 0.0
        
        # apply after a certain amount of training
        if false 
            bcv = supervised_betacv()
        end

        # optimize both metrics
        ce + bcv
    end

    opt = Flux.ADAM(params(m))

    function callback()
        ns = rand(1:N-1, round(Int, sqrt(N)))
        ls = map(n->loss(X[n], Y[n]).tracker.data, ns)
        println("Training:  loss: ", sum(ls), "\tstd: ", std(ls))

        # embeddings
        es = m[1:el](X[rand(1:N-1)]).data
        println("Embedding: ", es)
        println()
    end

    for epoch in 1:1
        info("Epoch: $epoch (Seed: $seed)")
        Flux.train!(loss, zip(X, Y), opt, cb=throttle(callback,3))
    end
end

train(rand(1:10000))

# end