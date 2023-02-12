using  CUDA


data = magic_data
X, y = data.X, data.y

train, test = shuffle_data(data, 0.1)


NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg= MLJFlux

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

magic_classifier = MyNetworkBuilder(16, 8)
clf = NeuralNetworkClassifier(builder=magic_classifier, batch_size = 100)
clf.acceleration = CUDALibs()
r1 = range(clf, :batch_size, lower=1, upper=500)
r2 = range(clf, :lambda, lower=0, upper=10)
tm = TunedModel(model=clf, ranges=[r1,r2], resampling=CV(nfolds=5), measure=area_under_curve)
m = machine(tm, X, y)
fit!(m, rows=train)



