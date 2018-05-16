using Base.Test

# @testset "Deep-Autoencoder with *cross entropy* and *BetaCV* loss" begin

using Flux
using Flux.Tracker
using Flux: crossentropy, throttle

using FluxMO.Clustering: knn_clustering
using FluxMO.Validation: betacv, betacv_pairwise

using Plots
gr()

function train(seed::Int = rand(1:10000); mode = :with_betacv)
    # seed = 8270
    srand(seed)

    N = 1000    # samples
    M = 100     # data size
    L = 2       # latent space size

    # generate correlated data
    D = map(_->abs.(cumsum(rand(M)./round(Int,N/2))),1:N)
    # cor(D[1],D[2])

    X = deepcopy(D[1:end-1])
    Y = deepcopy(D[2:end])

    a = sin
    m = Chain(
        Dense(M,   100, a),
        Dense(100, 10,  a),
        Dense(10,  L,   a),
        Dense(L,   10,  a),
        Dense(10,  100, a),
        Dense(100, M,   sigmoid),
    )
    embd_layer = 3

    BCV_take = 5

    epochs = 1

    kNN = 20

    randomize = false

    function supervised_betacv(i::Int)
        
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
        clustering, _ = knn_clustering(embds, k = kNN)
        # 221-element Array{Array{Int64,1},1}:
        # Array{Int64,1}[[350, 837, 466, 364, 600, 964, 271, 976, 1, 804], …

        # take only n first clusters to reduce backtracking...
        take = 1:BCV_take
        
        # take only n random clusters to reduce backtracking...
        if randomize
            cn = length(clustering)
            take = unique(rand(1:cn, min(BCV_take,cn)))
        end

        # Obtain tracked embedded values from the clustering
        embds_clustered_tracked = map(c->map(i->embds_tracked[i],c), clustering[take])
        # 221-element Array{Array{TrackedArray{…,Array{Float64,1}},1},1}:
        # TrackedArray{…,Array{Float64,1}}[param([0.311645, 0.0285734]), param([0.309032, 0.019196]), …] 
        # ⋮
        
        # Validate the clustering via BetaCV measure (small is good).
        # This is done with the tracked values, as it should influence
        # the model weights and biases towards optimizing this measure.
        bcv = @time betacv(embds_clustered_tracked)
        # Flux.Tracker.TrackedReal{Float64}

        # untracked betacv calculates fine.
        # embds_clustered = map(c->map(i->embds[:,i],c), clustering)
        # bcv = betacv(embds_clustered)

        # embds_clustered_tracked_matrix = map(c->hcat(map(i->embds_tracked[i],c)), clustering[take])
        
        # try classic betacv with for loops on tracked matrix
        # bcvm = @time betacv(embds_clustered_tracked_matrix)

        # try matrix optimized betacv for better backpropagation
        # bcvp = @time betacv_pairwise(embds_clustered_tracked_matrix)
        # ERROR: MethodError: no method matching one(::Type{TrackedArray{…,Array{Float64,1}}})
        # Closest candidates are:
        # one(::Type{Measures.Length{:mm,Float64}}) at /home/sebastian/.julia/v0.6/Plots/src/layouts.jl:13
        # one(::Type{Measures.Length{:pct,Float64}}) at /home/sebastian/.julia/v0.6/Plots/src/layouts.jl:31
        # one(::BitArray{2}) at bitarray.jl:427
        # ...
        # Stacktrace:
        # [1] result_type at /home/sebastian/.julia/v0.6/Distances/src/metrics.jl:194 [inlined]
        # [2] pairwise(::Distances.Euclidean, ::Array{TrackedArray{…,Array{Float64,1}},2}, ::Array{TrackedArray{…,Array{Float64,1}},2}) at /home/sebastian/.julia/v0.6/Distances/src/generic.jl:120
        # [3] intra_cluster_weights_pairwise(::Array{Array{TrackedArray{…,Array{Float64,1}},2},1}) at /home/sebastian/develop/julia/dev/FluxMO/src/betacv.jl:48
        # [4] betacv_pairwise(::Array{Array{TrackedArray{…,Array{Float64,1}},2},1}) at /home/sebastian/develop/julia/dev/FluxMO/src/betacv.jl:93
        # [5] macro expansion at ./util.jl:237 [inlined]
        # [6] (::#supervised_betacv#11{Array{Array{Float64,1},1},Flux.Chain,Int64})(::Int64) at ./none:63
        # [7] (::#loss#18{Symbol,Flux.Chain,#supervised_betacv#11{Array{Array{Float64,1},1},Flux.Chain,Int64}})(::Array{Float64,1}, ::Array{Float64,1}, ::Int64) at ./none:87
        # [8] #train!#130(::Flux.#throttled#14, ::Function, ::Function, ::Base.Iterators.Zip{Array{Array{Float64,1},1},Base.Iterators.Zip2{Array{Array{Float64,1},1},Array{Int64,1}}}, ::Flux.Optimise.##71#75) at /home/sebastian/.julia/v0.6/Flux/src/optimise/train.jl:39
        # [9] (::Flux.Optimise.#kw##train!)(::Array{Any,1}, ::Flux.Optimise.#train!, ::Function, ::Base.Iterators.Zip{Array{Array{Float64,1},1},Base.Iterators.Zip2{Array{Array{Float64,1},1},Array{Int64,1}}}, ::Function) at ./<missing>:0
        # [10] #train#1(::Symbol, ::Function, ::Int64) at ./none:110
        # [11] train() at ./none:3
        # [12] eval(::Module, ::Any) at ./boot.jl:235

        bcv
    end

    last_bcv = param(0.0)
    last_ce = param(0.0)

    function loss(x, y, i = -1)

        # unsupervised
        ŷ = m(x)
        prev_ce = deepcopy(last_ce.tracker.data)
        last_ce = crossentropy(ŷ, y)

        # supervised

        # NOTE:
        # This is where I like to apply the second optimization...
        # How to backtrack this function efficently?
        #   * Do I have to treat it as second model with an supervised setting?
        #   * or can I apply the `betacv` as intendet to the existing model?

        # apply after a fraction of all seen samples
        # do not apply when called manually from the callback
        # if i != -1 && i % floor(Int, (N-1)/3) == 0
        if mode == :with_betacv
            if i != -1 && i % 100 == 0
                print("applying betacv() after $i samples\t")
                prev_bcv = deepcopy(last_bcv.tracker.data)
                
                last_bcv = supervised_betacv(i)
                println("crossentropy: ", last_ce,  "\tdiff: ", last_ce.tracker.data  - prev_ce)
                println("betacv:       ", last_bcv, "\tdiff: ", last_bcv.tracker.data - prev_bcv)
            end
        end

        # NOTE: Does this even help to reduce the Stacktrace?
        Flux.truncate!(m)

        # optimize both metrics
        last_ce + last_bcv
    end

    opt = Flux.ADAM(params(m))

    function callback()
        # sqrt(N) random samples for approx. loss
        ns = rand(1:N-1, round(Int, sqrt(N)))
        ls = [loss(X[n], Y[n]) for n in ns]
        println("Train: loss:  ", sum(ls)/length(ls), "\tstd:  ", std(ls))

        # one embedding example
        embd = m[1:embd_layer](X[rand(1:N-1)]).data
        println("Embedding:    ", embd)
        
        println()
    end

    for epoch in 1:epochs
        info("Epoch: $epoch (Seed: $seed)")
        Flux.train!(loss, zip(X, Y, collect(1:length(X))), opt, cb=throttle(callback,3))
    end

    return X,Y,m,last_ce, [seed,N,M,L,embd_layer,a,BCV_take,epochs,kNN,randomize]
end

X_bcv,Y_bcv,model_bcv,ce_bcv,opts = train()
data_bcv = deepcopy(hcat(map(x->model_bcv[1:3](x).data, X_bcv)...))
clustering, _ = knn_clustering(data_bcv)
data_clustered = map(c->hcat(map(i->data_bcv[:,i],c)), clustering)
bcv_bcv = betacv(data_clustered)

ce_bcv_s = @sprintf "%1.2e" ce_bcv
bcv_bcv_s = @sprintf "%1.5f" bcv_bcv

X_ce,Y_ce,model_ce,ce_ce,_ = train(mode = :other)
data_ce = deepcopy(hcat(map(x->model_ce[1:3](x).data, X_ce)...))
clustering, _ = knn_clustering(data_ce)
data_clustered = map(c->hcat(map(i->data_ce[:,i],c)), clustering)
bcv_ce = betacv(data_clustered)
ce_ce_s = @sprintf "%1.2e" ce_ce
bcv_ce_s = @sprintf "%1.5f" bcv_ce

p = plot(
    scatter(data_bcv[1,:], data_bcv[2,:],
    label=["embedded (ce+bcv)"],
    title="crossentropy: $ce_bcv_s\nbetacv: $bcv_bcv_s"),

    scatter(data_ce[1,:], data_ce[2,:],
    label=["embedded (ce)"],
    title="crossentropy: $ce_ce_s\nbetacv: $bcv_ce_s"),
)

savefig(p, string(now(), "_", opts, "_", ".png"))
# end
