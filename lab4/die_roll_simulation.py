import numpy as np
import matplotlib.pyplot as plt
import os

# Number of die rolls (from 1 to 50)
num_rolls_range = range(1, 51)
# Number of experiments for each case
num_experiments_list = [100, 500, 1000, 5000, 10000, 50000, 100000]

# Create output directory for histograms
output_dir = "die_roll_histograms"
os.makedirs(output_dir, exist_ok=True)

results = {}

for num_rolls in num_rolls_range:
    results[num_rolls] = {}
    for num_experiments in num_experiments_list:
        # Simulate die rolls
        rolls = np.random.randint(1, 7, size=(num_experiments, num_rolls))
        if num_rolls == 1:
            # For one roll, just use the value itself
            sums = rolls.flatten()
        else:
            # For more than one roll, sum the rolls
            sums = rolls.sum(axis=1)
        # Calculate mean and variance
        mean = np.mean(sums)
        variance = np.var(sums)
        results[num_rolls][num_experiments] = {'mean': mean, 'variance': variance}
        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.hist(sums, bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Histogram: {num_rolls} Die Rolls, {num_experiments} Experiments")
        plt.xlabel("Sum of Die Rolls" if num_rolls > 1 else "Die Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        # Save histogram
        filename = f"hist_{num_rolls}rolls_{num_experiments}exp.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

# Print mean and variance for each case
for num_rolls in num_rolls_range:
    for num_experiments in num_experiments_list:
        mean = results[num_rolls][num_experiments]['mean']
        variance = results[num_rolls][num_experiments]['variance']
        print(f"Rolls: {num_rolls}, Experiments: {num_experiments} -> Mean: {mean:.2f}, Variance: {variance:.2f}")
