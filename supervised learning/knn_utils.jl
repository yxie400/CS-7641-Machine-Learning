import DecisionTree
import NearestNeighborModels 

include("./utils.jl");

function train_knn(data::MagicData, train::Vector)
    X, y = data.X, data.y
    KNNClassifier = @load KNNClassifier 
    knn = KNNClassifier()
    r1 = range(knn, :K, lower=3, upper=10)
    r2 = range(knn, :algorithm, values = [:kdtree, :brutetree, :balltree]) 
    tm = TunedModel(model=knn, ranges=[r1,r2], resampling=CV(nfolds=5), measure=area_under_curve)
    m = machine(tm, X, y)
    fit!(m, rows=train)
    bm = fitted_params(m).best_model
    return(bm)
end

function train_knn(data::CreditData, train::Vector)
    X, y = data.X, data.y
    KNNClassifier = @load KNNClassifier 
    knn = KNNClassifier()
    r1 = range(knn, :K, lower=3, upper=10)
    r2 = range(knn, :algorithm, values = [:kdtree, :brutetree, :balltree]) 
    tm = TunedModel(model=knn, ranges=[r1,r2], resampling=CV(nfolds=5))
    tm.measure = FScore(β=0.2)
    m = machine(tm, X, y)
    fit!(m, rows=train)
    bm = fitted_params(m).best_model
    return(bm)
end



function knn_model_performance(data::T, fraction::Float64, no_prob=false) where {T <: MLData}
    train, test = shuffle_data(data, fraction)
    dtc = train_knn(data, train)
    train_score = model_scores(dtc, data,train,no_prob)
    test_score = model_scores(dtc, data,train,test,no_prob)
    return (train_score, test_score)
end 


