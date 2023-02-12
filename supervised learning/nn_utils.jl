import MLJFlux
import Flux

include("./utils.jl");

mutable struct MyNetworkBuilder <: MLJFlux.Builder
    n1::Int #Number of cells in the first hidden layer
    n2::Int #Number of cells in the second hidden layer
end

function MLJFlux.build(model::MyNetworkBuilder, rng, n_in, n_out)
    init = Flux.glorot_uniform(rng)
    layer1 = Flux.Dense(n_in, model.n1, init=init)
    layer2 = Flux.Dense(model.n1, model.n2, init=init)
    layer3 = Flux.Dense(model.n2, n_out, init=init)
    return Flux.Chain(layer1, layer2, layer3)
end

function train_nn(data::MagicData, train::Vector, e::Integer)
    X, y = data.X, data.y
    NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg= MLJFlux
    magic_classifier = MyNetworkBuilder(16, 8)
    clf = NeuralNetworkClassifier(builder=magic_classifier, epochs = e)
    #clf.acceleration = CUDALibs()
    r1 = range(clf, :batch_size, lower=10, upper=1000, scale=:log10)
    r2 = range(clf, :lambda, lower=0, upper=5)
    tm = TunedModel(model=clf, ranges=[r1,r2], resampling=CV(nfolds=5), measure=area_under_curve)
    m = machine(tm, X, y)
    fit!(m, rows=train)
    bm = fitted_params(m).best_model
    return(bm)
end

function train_nn(data::CreditData, train::Vector, e::Integer)
    X, y = data.X, data.y
    NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg= MLJFlux
    credit_classifier = MyNetworkBuilder(16, 8)
    clf = NeuralNetworkClassifier(builder=credit_classifier, epochs = e)
    #clf.acceleration = CUDALibs()
    r1 = range(clf, :batch_size, lower=10, upper=1000, scale=:log10)
    r2 = range(clf, :lambda, lower=0, upper=5)
    tm = TunedModel(model=clf, ranges=[r1,r2], resampling=CV(nfolds=5))
    tm.measure = FScore(β=0.2)
    m = machine(tm, X, y)
    fit!(m, rows=train)
    bm = fitted_params(m).best_model
    return(bm)
end



function nn_model_performance(data::T, e::Integer, no_prob=false) where {T <: MLData}
    train, test = shuffle_data(data, 0.9)
    dtc = train_nn(data, train, e)
    train_score = model_scores(dtc, data,train,no_prob)
    test_score = model_scores(dtc, data,train,test,no_prob)
    return (train_score, test_score)
end 


