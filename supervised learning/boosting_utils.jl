import MLJXGBoostInterface 

include("./utils.jl");

function train_xgb(data::MagicData, train::Vector)
    X, y = data.X, data.y
    XGBC = @load XGBoostClassifier
    magic_xgbc = XGBC()
    r1 = range(magic_xgbc, :max_depth, lower=3, upper=10)
    r2 = range(magic_xgbc, :min_child_weight, lower=0, upper=5)
    tm = TunedModel(model=magic_xgbc, ranges=[r1,r2], resampling=CV(nfolds=5), measure=area_under_curve)
    m = machine(tm, X, y)
    fit!(m, rows=train)
    bm = fitted_params(m).best_model
    return(bm)
end

function train_xgb(data::CreditData, train::Vector)
    X, y = data.X, data.y
    XGBC = @load XGBoostClassifier
    credit_xgbc = XGBC()
    r1 = range(credit_xgbc, :max_depth, lower=3, upper=10)
    r2 = range(credit_xgbc, :min_child_weight, lower=0, upper=5)
    tm = TunedModel(model=credit_xgbc, ranges=[r1,r2], resampling=CV(nfolds=5))
    tm.measure = FScore(β=0.2)
    m = machine(tm, X, y)
    fit!(m, rows=train)
    bm = fitted_params(m).best_model
    return(bm)
end



function xgb_model_performance(data::T, fraction::Float64, no_prob=false) where {T <: MLData}
    train, test = shuffle_data(data, fraction)
    xgb = train_xgb(data, train)
    train_score = model_scores(xgb, data,train,no_prob)
    test_score = model_scores(xgb, data,train,test,no_prob)
    return (train_score, test_score)
end 


