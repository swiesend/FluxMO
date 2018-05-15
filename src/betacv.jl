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


function weights_colwise(S::AbstractArray, R::AbstractArray, metric::Distances.PreMetric = Distances.Euclidean())
    ws = Distances.colwise(metric, S, R)
    sum(ws)
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


function inter_cluster_weights(C::AbstractArray)
    W_out = 0.0
    N_out = 0
    for (i,S) in enumerate(C)
        for (j,R) in enumerate(C)
            if j > i
                W_out += weights(S,R)
                N_out += length(S) * length(R)
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
