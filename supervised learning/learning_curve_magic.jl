using  ARFFFiles, DataFrames, MLJ, StableRNGs, PrettyPrinting
using MLJBase, MLJEnsembles
using Plots

# read from arff files
df = ARFFFiles.load(DataFrame, "MagicTelescope.arff")

df = ARFFFiles.load(DataFrame, "dataset_31_credit-g.arff")

# change class label from character to 0/1
df."class:" = [c == "good" ? 1 : 0 for c in df."class"]

# remove ID column
select!(df,Not([:ID]))
# remove ":" in column names
for n in names(df)
    rename!(df, n => chop(n))
end
coerce!(df, :class => Multiclass)

MLJ.schema(df)

describe(df)

first(df, 3) |> pretty

length(df.class)

#y, X = unpack(df, ==(:class))

y = df.class 
X = select!(df,Not([:class]))
dropmissing!(X)

DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
tree_model = DecisionTreeClassifier()
tree = machine(tree_model, X, y)

rng = StableRNG(12345)

train, test = partition(eachindex(y), 0.7, shuffle=true, rng = rng);

fit!(tree, rows=train)

fitted_params(tree) |> pprint

ŷ = predict(tree, rows=test)

ȳ = predict_mode(tree, rows=test)

mce = cross_entropy(ŷ, y[test]) |> mean

dtc = DecisionTreeClassifier()
r1 = range(dtc, :max_depth, lower=1, upper=20)
tm = TunedModel(model=dtc, ranges=[r,], measure=area_under_curve)

m = machine(tm, X, y)
fit!(m, rows=train)
ny = predict(m, rows=test)
area_under_curve(ny, y[test])

fitted_params(m).best_model.max_depth
r = report(m)

using Plots
plot(m)

KNNClassifier = @load KNNClassifier
knn_model = KNNClassifier()
knn = machine(knn_model, X, y)
fit!(knn, rows=train)
ŷ = predict(knn, rows=test);
area_under_curve(ŷ, y[test])

r1 = range(knn_model, :K, lower=1, upper=20)
r2 = range(knn_model, :algorithm, values = [:kdtree, :balltree])
tm = TunedModel(model=knn_model,tuning=RandomSearch(), ranges=[r1,r2,], measure=area_under_curve)
km = machine(tm, X, y)
fit!(km, rows=train)
ny = predict(km, rows=test);
area_under_curve(ny, y[test])
report(km)


ensemble_model = EnsembleModel(model=knn_model, n=20);
ensemble = machine(ensemble_model, X, y)
estimates = evaluate!(ensemble, resampling=CV(),measure=area_under_curve)
estimates

forest = EnsembleModel(model=DecisionTreeClassifier())
forest.model.n_subfeatures = 3

rng = StableRNG(5123) # for reproducibility
m = machine(forest, X, y)
r = range(forest, :n, lower=10, upper=100)
curves = MLJ.learning_curve(m, resampling=CV(nfolds=3),
                         range=r, measure=area_under_curve, acceleration=CPUThreads());

plot(curves.parameter_values, curves.measurements)

MLJ.default_resource(CPUThreads())
