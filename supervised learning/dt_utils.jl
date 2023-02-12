import DecisionTree
import MLJDecisionTreeInterface 

include("./utils.jl");

function train_decision_tree(data::MagicData, train::Vector)
    X, y = data.X, data.y
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
    magic_dtc = DecisionTreeClassifier()
    r1 = range(magic_dtc, :max_depth, lower=1, upper=20)
    tm = TunedModel(model=magic_dtc, ranges=[r1,], resampling=CV(nfolds=5), measure=area_under_curve)
    m = machine(tm, X, y)
    fit!(m, rows=train)
    bm = fitted_params(m).best_model
    return(bm)
end

function train_decision_tree(data::CreditData, train::Vector)
    X, y = data.X, data.y
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
    credit_dtc = DecisionTreeClassifier()
    r1 = range(credit_dtc, :max_depth, lower=1, upper=20)
    r2 = range(credit_dtc, :min_samples_leaf, values = [1,2,5]) 
    r3 = range(credit_dtc, :min_samples_split, values = [2,5,7,10]) 
    tm = TunedModel(model=credit_dtc, ranges=[r1,r2,r3], resampling=CV(nfolds=5))
    tm.measure = FScore(β=0.2)
    m = machine(tm, X, y)
    fit!(m, rows=train)
    bm = fitted_params(m).best_model
    return(bm)
end



function dt_model_performance(data::T, fraction::Float64) where {T <: MLData}
    train, test = shuffle_data(data, fraction)
    dtc = train_decision_tree(data, train)
    train_score = model_scores(dtc, data,train)
    test_score = model_scores(dtc, data,train,test)
    return (train_score, test_score)
end 


