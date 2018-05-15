using NearestNeighbors
using Distances


function knn_clustering(data::AbstractMatrix;
    k = 10,
    mode = :hard,
    treetype::Type{T} = BallTree,
    metric::M = Euclidean()) where {T <: NearestNeighbors.NNTree, M <: Distances.PreMetric}

    N = size(data)[2]
    tree = treetype(data, metric)
    # centroids = hcat(map(r-> data[:,r], rand(1:N, N))...)
    # centroids = hcat(map(n-> data[:,n], 1:N)...)
    knnq = knn(tree, data, k)
    # @show length(unique(knnq[1]...))

    C = Vector{Array{Int64,1}}()
    used = Set{Int64}()
    if mode == :soft
        for (i,knn) in enumerate(knnq[1])
            points = Vector{Int64}()
            for nn in knn
                push!(used, nn)
                # push!(points, data[:,nn])
                push!(points, nn)
            end
            if length(points) > 0
                push!(C, points)
            end
        end
    elseif mode == :hard
        for (i, knn) in enumerate(knnq[1])
            points = Vector{Int64}()
            for nn in knn
                if !(nn in used)
                    push!(used, nn)
                    # push!(points, data[:,nn])
                    push!(points, nn)
                end
            end
            if length(points) > 0
                push!(C,points)
            end
        end
    else
        error("Unknown mode: $mode")
    end

    C, sort(collect(used))
end
