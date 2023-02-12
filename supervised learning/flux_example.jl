import MLJFlux
import MLJ
import DataFrames: DataFrame
import Statistics
import Flux
using Random

Random.seed!(11)


features, targets = MLJ.@load_boston
features = DataFrame(features)
@show size(features)
@show targets[1:3]
first(features, 3) |> MLJ.pretty

train, test = MLJ.partition(collect(eachindex(targets)), 0.70, rng=52)



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


myregressor = MyNetworkBuilder(20, 10)
nnregressor = MLJFlux.NeuralNetworkRegressor(builder=myregressor, epochs=100)
nnregressor.acceleration = CUDALibs()
mach = MLJ.machine(nnregressor, features, targets)
MLJ.fit!(mach, rows=train, verbosity=3, force=true)


bs = MLJ.range(nnregressor, :batch_size, lower=1, upper=5)

tm = MLJ.TunedModel(model=nnregressor, ranges=[bs, ],measure=MLJ.l2)
tm.acceleration=CPUThreads()
m = MLJ.machine(tm, features, targets)
MLJ.fit!(m)