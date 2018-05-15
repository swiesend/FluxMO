using Base.Test

# @testset "Deep-Autoencoder with *cross entropy* and *BetaCV* loss" begin

using Flux
using Flux.Tracker
using Flux: crossentropy, throttle

using FluxMO.Clustering: knn_clustering
using FluxMO.Validation: betacv

function train(seed::Int = rand(1:10000))
    srand(seed)

    N = 1000    # samples
    M = 100     # data size
    L = 2       # latent space size

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
    embd_layer = 3

    function supervised_betacv()
        
        # Embed all samples from X into the latent space of size L
        embds_tracked = map(x-> m[1:embd_layer](x), X)
        # 999-element Array{TrackedArray{…,Array{Float64,1}},1}:
        # param([0.63072, -0.046672])  
        # ⋮

        # untracked embedded points as matrix
        embds  = hcat(map(ta->ta.data, embds_tracked)...)
        # 2×999 Array{Float64,2}:
        
        # Cluster the current untracked embeddings to generate
        # a supervised scenario.
        # This clould also be done by DBSCAN, OPTICS, K-Means ...
        clustering, _ = knn_clustering(embds)
        # 221-element Array{Array{Int64,1},1}:
        # Array{Int64,1}[[350, 837, 466, 364, 600, 964, 271, 976, 1, 804], …

        # Obtain tracked embedded values from the clustering
        embds_clustered_tracked = map(c->map(i->embds_tracked[i],c), clustering)
        # 221-element Array{Array{TrackedArray{…,Array{Float64,1}},1},1}:
        # TrackedArray{…,Array{Float64,1}}[param([0.311645, 0.0285734]), param([0.309032, 0.019196])] 
        # ⋮

        # Validate the clustering via BetaCV measure (small is good).
        # This is done with the tracked values, as it should influence
        # the model weights and biases towards optimizing this measure.
        bcv = betacv(embds_clustered_tracked)
        # Flux.Tracker.TrackedReal{Float64}

        # untracked betacv calculates fine.
        # embds_clustered = map(c->map(i->embds[:,i],c), clustering)
        # bcv = betacv(embds_clustered)

        @show bcv
        bcv
    end

    last_bcv = 0.0

    function loss(x, y, i = -1)

        # unsupervised
        ŷ = m(x)
        ce = crossentropy(ŷ, y)

        # supervised

        # NOTE:
        # This is where I like to apply the second optimization...
        # How to backtrack this function efficently?
        #   * Do I have to treat it as second model with an supervised setting?
        #   * or can I apply the `betacv` as intendet to the existing model?

        # apply after a fraction of all seen samples
        # do not apply when called manually from the callback
        if i != -1 && i % floor(Int, (N-1)/3) == 0
            println("applying betacv() after $i samples")
            last_bcv = supervised_betacv()
        end

        # optimize both metrics
        ce + last_bcv
    end

    opt = Flux.ADAM(params(m))

    function callback()
        ns = rand(1:N-1, round(Int, sqrt(N)))
        ls = map(n->loss(X[n], Y[n]).tracker.data, ns)
        println("Training:  loss: ", sum(ls), "\tstd: ", std(ls))

        # embeddings
        es = m[1:embd_layer](X[rand(1:N-1)]).data
        println("Embedding: ", es)
        
        println()
    end

    for epoch in 1:1
        info("Epoch: $epoch (Seed: $seed)")
        Flux.train!(loss, zip(X, Y, collect(1:length(X))), opt, cb=throttle(callback,3))
    end
end

train()

# end
