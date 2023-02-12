import MLJLIBSVMInterface

include("./utils.jl");

function train_SVM(data::MagicData, train::Vector)
    X, y = data.X, data.y
    SVC = @load SVC pkg=LIBSVM
    magic_svc = SVC()
    r1 = range(magic_svc, :cost, lower=0.1, upper=5)
    r2 = range(magic_svc, :gamma, lower=0, upper=1) 
    tm = TunedModel(model=magic_svc, ranges=[r1,r2], resampling=CV(nfolds=5), measure=FScore(β=5))
    m = machine(tm, X, y)
    fit!(m, rows=train)
    bm = fitted_params(m).best_model
    return(bm)
end

function train_SVM(data::CreditData, train::Vector)
    X, y = data.X, data.y
    SVC = @load SVC pkg=LIBSVM
    credit_svc = SVC()
    r1 = range(credit_svc, :cost, lower=0.1, upper=5)
    r2 = range(credit_svc, :gamma, lower=0, upper=1) 
    tm = TunedModel(model=credit_svc, ranges=[r1,r2], resampling=CV(nfolds=5), measure = FScore(β=0.2))
    m = machine(tm, X, y)
    fit!(m, rows=train)
    bm = fitted_params(m).best_model
    return(bm)
end



function svm_model_performance(data::T, fraction::Float64, no_prob=false) where {T <: MLData}
    train, test = shuffle_data(data, fraction)
    dtc = train_SVM(data, train)
    train_score = model_scores(dtc, data,train, no_prob)
    test_score = model_scores(dtc, data,train,test, no_prob)
    return (train_score, test_score)
end 

function svm_model_performance(data::T, fraction::Float64) where {T <: MLData}
    train, test = shuffle_data(data, fraction)
    dtc = train_SVM(data, train)
    train_score = model_scores(dtc, data, train)
    test_score = model_scores(dtc, data,train,test)
    return (train_score, test_score)
end 
