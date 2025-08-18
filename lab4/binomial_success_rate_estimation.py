import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_trials = 100  # Number of reviews
n_simulations = 10000  # Number of simulations per probability
observed_successes = 97  # Actual positive reviews

# Probabilities to test
probabilities = [0.90, 0.95, 0.97, 0.99]

plt.figure(figsize=(12, 8))
for i, p in enumerate(probabilities, 1):
    # Simulate binomial outcomes
    samples = np.random.binomial(n_trials, p, n_simulations)
    plt.subplot(2, 2, i)
    plt.hist(samples, bins=range(n_trials+2), alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(observed_successes, color='red', linestyle='dashed', linewidth=2, label=f'Observed: {observed_successes}')
    plt.title(f'Binomial Distribution (p={p})')
    plt.xlabel('Number of Positive Reviews')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()

plt.suptitle('Binomial Distributions for Different Success Rates')
plt.show()
