ABOUT = {
    "Historical Average": (
        "Predicts the average flow this sensor had at the same weekday and time of day during the "
        "training period. It ignores what is happening right now, so it is the simple reference that "
        "every learning model must beat."),
    "Random Forest": (
        "Many decision trees, each trained on a random part of the data, vote on the answer. "
        "Input: the last 12 readings, their 60-min and 15-min averages and variability, the latest change, "
        "and the hour and weekday."),
    "XGBoost": (
        "Gradient-boosted trees: trees are added one after another, and each new tree corrects the errors "
        "of the previous ones. It uses exactly the same features as Random Forest, so only the learning "
        "method differs."),
    "LSTM": (
        "A recurrent neural network that reads the last 12 readings (60 minutes) in order. Its gated memory "
        "cells keep useful information from earlier steps, and its final hidden state is turned into the "
        "15-minute forecast. It sees only the raw sequence, with no hand-made features."),
    "GRU": (
        "A lighter recurrent network with fewer gates than LSTM. It gets the same input and target, which "
        "tests whether the extra complexity of LSTM is needed."),
    "Hybrid LSTM-XGBoost": (
        "The proposed model. The trained LSTM is frozen and used as a feature extractor: its hidden state "
        "summarises the last hour as a vector of numbers. That vector is joined with the engineered features, "
        "and XGBoost makes the final forecast. It combines learned temporal patterns with nonlinear tree "
        "regression. It is not an average of two predictions."),
}
