using  ARFFFiles, DataFrames, MLJ, CategoricalArrays, MLJModelInterface
using Plots
import DecisionTree
import MLJDecisionTreeInterface 

abstract type MLData end

struct MagicData <: MLData
    X::DataFrame 
    y::CategoricalVector
end

struct CreditData <: MLData
    X::DataFrame 
    y::CategoricalVector
end

function load_data()
    # load magic telescope data
    magic_df = ARFFFiles.load(DataFrame, "MagicTelescope.arff")
    # convert class tag to 0/1
    magic_df."class:" = [c == "g" ? 1 : 0 for c in magic_df."class:"]
    # remove ID column
    select!(magic_df,Not([:ID]))
    # remove ":" in column names
    for n in names(magic_df)
        rename!(magic_df, n => chop(n))
    end
    coerce!(magic_df, :class => Multiclass)
    magic_y, magic_X = unpack(magic_df, ==(:class), rng=12345)
    magic_data = MagicData(magic_X, magic_y)

    credit_df = ARFFFiles.load(DataFrame, "dataset_31_credit-g.arff")
    credit_df."class" = [c == "good" ? 1 : 0 for c in credit_df."class"]
    coerce!(credit_df, :class => OrderedFactor)
    credit_y, old_credit_X = unpack(credit_df, ==(:class), rng=12345)
    # using 1 hot encoding to transform categorical variable to numerical
    hot = OneHotEncoder(drop_last=true, ordered_factor=true)
    mach = MLJ.fit!(machine(hot, old_credit_X))
    credit_X = MLJ.transform(mach, old_credit_X)
    credit_data = CreditData(credit_X, credit_y)

    return (magic_data, credit_data)
end 

function shuffle_data(data::T, fraction::Float64) where {T <: MLData}
    train, test = partition(eachindex(data.y), fraction; rng = 123)
    return (train, test)
end

function model_scores(model::Model, data::MagicData, train::Vector, no_prob=false)
    if no_prob
        X, y = data.X, data.y
        m = machine(model, X, y)
        fit!(m, rows=train)
        ŷ = predict(m, rows=train);
        res = FScore(β=5)(ŷ, y[train])
    else 
        X, y = data.X, data.y
        m = machine(model, X, y)
        fit!(m, rows=train)
        ŷ = predict(m, rows=train);
        res = area_under_curve(ŷ, y[train])
    end
    return res
end


function model_scores(model::Model, data::CreditData, train::Vector, no_prob = false)
    if no_prob
        X, y = data.X, data.y
        m = machine(model, X, y)
        fit!(m, rows=train)
        ŷ = predict_mode(m, rows=train);
        res = FScore(β=0.2)(ŷ, y[train])
    else 
        X, y = data.X, data.y
        m = machine(model, X, y)
        fit!(m, rows=train)
        ŷ = predict(m, rows=train);
        res = FScore(β=0.2)(ŷ, y[train])
    end
    return res
end

function model_scores(model::Model, data::MagicData, train::Vector, test::Vector, no_prob=false)
    if no_prob
        X, y = data.X, data.y
        m = machine(model, X, y)
        fit!(m, rows=train)
        ŷ = predict(m, rows=test);
        res = FScore(β=5)(ŷ, y[test])
    else
        X, y = data.X, data.y
        m = machine(model, X, y)
        fit!(m, rows=train)
        ŷ = predict(m, rows=test);
        res = area_under_curve(ŷ, y[test])
    end
    return res
end

function model_scores(model::Model, data::CreditData, train::Vector, test::Vector, no_prob = false)
    if no_prob
        X, y = data.X, data.y
        m = machine(model, X, y)
        fit!(m, rows=train)
        ŷ = predict_mode(m, rows=test);
        res = FScore(β=0.2)(ŷ, y[test])
    else 
        X, y = data.X, data.y
        m = machine(model, X, y)
        fit!(m, rows=train)
        ŷ = predict(m, rows=test);
        res = FScore(β=0.2)(ŷ, y[test])
    end
    return res
end

function plot_learning_curve(train_score::Vector, test_score::Vector, fractions, name)
    plot(fractions, [train_score test_score], 
        label=["traing score" "testing score"],
        linewidth=3, marker=:o)
    plot!(legend=:outerbottom, legendcolumns=2)
    #xlims!(0,1)
    title!(name*"_training_and_testing_scores")
    xlabel!("fraction of training data")
    ylabel!("model score")
    savefig(name*".png")
end

MLJ.default_resource(CPUThreads())