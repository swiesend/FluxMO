using Distances


function weights(S::AbstractArray, R::AbstractArray, metric::Function = Distances.euclidean)
    ws = 0.0
    for s in S
        for r in R
            ws += metric(s,r)
        end
    end
    ws
end


function weights_pairwise(S::AbstractMatrix, R::AbstractMatrix, metric::Distances.PreMetric = Distances.Euclidean())
    sum(Distances.pairwise(metric,S,R))
end


function weights_half(S::AbstractArray, R::AbstractArray, metric::Function = Distances.euclidean)
    ws = 0.0
    for (i,s) in enumerate(S)
        for (j,r) in enumerate(R)
            if i > j
                ws += metric(s,r)
            end
        end
    end
    ws
end


# https://www.coursera.org/learn/cluster-analysis/lecture/jDuBD/6-7-internal-measures-for-clustering-validation
function intra_cluster_weights(C::AbstractArray)
    W_in = 0.0
    N_in = 0
    for (i,S) in enumerate(C)
        W_in += weights_half(S,S)
        N_in += binomial(length(S),2)
    end
    W_in, N_in
end

function intra_cluster_weights_pairwise(C::AbstractArray)
    W_in = 0.0
    N_in = 0
    for (i,S) in enumerate(C)
        # NOTE: problem 1: setindex! is not differentiable
        # NOTE: problem 2: Tracked Float64 can be convert to Float64
        W_in = W_in + weights_pairwise(S,S)
        N_in += binomial(size(S)[1],2)
    end
    0.5*W_in, N_in
end


function inter_cluster_weights(C::AbstractArray)
    W_out = 0.0
    N_out = 0
    for (i,S) in enumerate(C)
        for (j,R) in enumerate(C)
            if j > i
                W_out += weights(S,R)
                N_out += size(S)[1] * size(R)[1]
            end
        end
    end
    W_out, N_out
end

function inter_cluster_weights_pairwise(C::AbstractArray)
    W_out = 0.0
    N_out = 0
    for (i,S) in enumerate(C)
        for (j,R) in enumerate(C)
            if j > i
                W_out = W_out + weights_pairwise(S,R)
                N_out += size(S)[1] * size(R)[1]
            end
        end
    end
    W_out, N_out
end


function betacv(C::AbstractArray)
    W_in, N_in = intra_cluster_weights(C)
    W_out, N_out = inter_cluster_weights(C)
    
    (W_in / N_in) / (W_out / N_out)
end


function betacv_fused(C::AbstractArray; metric::Function = Distances.euclidean)
    K = length(C)
    W_in, W_out = 0.0, 0.0
    N_in, N_out = 0,   0

    for i in 1:K
        # intra cluster weights
        for k in 1:length(C[i])
            for l in 1:length(C[i])
                if k > l
                    W_in += metric(C[i][k],C[i][l])
                end
            end
        end
        N_in += binomial(length(C[i]),2)
        # inter cluster weights
        for j in 1:K
            if j > i
                for s in C[i]
                    for r in C[j]
                        W_out += metric(s,r)
                    end
                end
                N_out += size(C[i])[1] * size(C[j])[1]
            end
        end
    end
    
    (W_in / N_in) / (W_out / N_out)
end


function betacv_pairwise(C::AbstractArray)
    W_in, N_in = intra_cluster_weights_pairwise(C)
    W_out, N_out = inter_cluster_weights_pairwise(C)
    
    (W_in / N_in) / (W_out / N_out)
end
