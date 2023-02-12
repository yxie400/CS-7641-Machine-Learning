include("./nn_utils.jl");

magic_data, credit_data = load_data();

fraction = 0.9

iterations = 10:50:500

# magic data
magic_train_score = []
magic_test_score = []
for i in iterations
    train_score, test_score = nn_model_performance(magic_data, i);
    push!(magic_train_score, train_score)
    push!(magic_test_score, test_score)
end
plot_learning_curve(magic_train_score, magic_test_score, iterations, "magic_neural_network")
# credit data
credit_train_score = []
credit_test_score = []
for i in iterations
    train_score, test_score = nn_model_performance(credit_data, i, true);
    push!(credit_train_score, train_score)
    push!(credit_test_score, test_score)
end
plot_learning_curve(credit_train_score, credit_test_score, iterations, "credit_neural_network")
