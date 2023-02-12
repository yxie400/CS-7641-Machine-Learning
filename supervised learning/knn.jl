include("./knn_utils.jl");

magic_data, credit_data = load_data();

fractions = 0.1:0.05:0.9

# magic data
magic_train_score = []
magic_test_score = []
for f in fractions
    train_score, test_score = knn_model_performance(magic_data, f);
    push!(magic_train_score, train_score)
    push!(magic_test_score, test_score)
end
plot_learning_curve(magic_train_score, magic_test_score, fractions, "magic_knn")
# credit data  
credit_train_score = []
credit_test_score = []
for f in fractions
    train_score, test_score = knn_model_performance(credit_data, f, true);
    push!(credit_train_score, train_score)
    push!(credit_test_score, test_score)
end
plot_learning_curve(credit_train_score, credit_test_score, fractions, "credit_knn")
