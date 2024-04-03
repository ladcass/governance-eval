import numpy as np
from scipy import stats

import ast

# Initialize the dictionary to store model accuracies
model_accuracies = {}

# List of CNN filenames
filenames = [
    "models/accuracies/cnn_model_1_accuracies.txt",
    "models/accuracies/cnn_model_2_accuracies.txt",
    "models/accuracies/cnn_model_3_accuracies.txt",
    "models/accuracies/cnn_model_4_accuracies.txt",
    "models/accuracies/cnn_model_5_accuracies.txt"
]


# Function to read accuracies from a file and return as a dictionary
def read_accuracies_from_file(filename):
    with open(filename, "r") as f:
        for line in f:
            # Split the line at the colon, stripping whitespace
            model_name, accuracies_str = line.strip().split(": ", 1)
            # Convert the string representation of the list back to a list
            accuracies = ast.literal_eval(accuracies_str)
            # Store in the dictionary
            model_accuracies[model_name] = accuracies
    return model_accuracies


# For each CNN model file
for filename in filenames:
    accuracies_dict = read_accuracies_from_file(filename)
    model_accuracies.update(accuracies_dict)


# For the main accuracies file which contains 15 models
with open("models/accuracies/model_accuracies.txt", "r") as f:
    for line in f:
        model_name, accuracies_str = line.strip().split(": ", 1)
        accuracies = ast.literal_eval(accuracies_str)
        model_accuracies[model_name] = accuracies

# Baseline accuracy for random guessing
mu = 1 / 7

# Store summary statistics
model_stats = {}

# Perform t-test against baseline for each model
for model_name, accuracies in model_accuracies.items():
    x_bar = np.mean(accuracies)
    s = np.std(accuracies, ddof=1)
    n = len(accuracies)
    se = s / np.sqrt(n)
    t_stat = (x_bar - mu) / se
    df = n - 1
    p_value = stats.t.sf(np.abs(t_stat), df)  # One-tailed test

    model_stats[model_name] = {
        'mean_accuracy': x_bar,
        'std_dev': s,
        'alpha': p_value
    }

# Find the model(s) with alpha less than 0.10 (significantly better than random)
significant_models = {k: v for k, v in model_stats.items() if v['alpha'] < 0.10}

print("Significant models:")
for name, statistics in significant_models.items():
    print(f"{name}: Mean Accuracy = {statistics['mean_accuracy']:.4f}, p-value = {statistics['alpha']:.4f}")

# If only one model is significantly better, choose that one
if len(significant_models) == 1:
    best_model = list(significant_models.keys())[0]
    print(f"Best model based on alpha < 0.10: {best_model}")
elif len(significant_models) > 1:

    sorted_models = sorted(significant_models.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
    best_model = sorted_models[0][0]
    second_best_model = sorted_models[1][0]

    # Perform t-test between the best and second-best models
    best_acc = model_accuracies[best_model]
    second_best_acc = model_accuracies[second_best_model]

    t_stat, diff_p_value = stats.ttest_ind(best_acc, second_best_acc, equal_var=False, alternative='greater')
    print(diff_p_value)
    if diff_p_value < 0.10:
        print(f"Best model based on comparison t-test: {best_model}")
    else:
        # Perform ANOVA
        acc_lists = [model_accuracies[name] for name, _ in significant_models.items()]
        f_stat, anova_p_value = stats.f_oneway(*acc_lists)

        if anova_p_value < 0.10:
            print(f"Best model based on ANOVA: {best_model}")
        else:
            print("No significant differences among models; consider other criteria.")
else:
    print("No model significantly better than random chance at the 10% level.")
