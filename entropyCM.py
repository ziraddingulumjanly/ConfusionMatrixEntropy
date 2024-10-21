import numpy as np

def calculate_entropy(CM):
    N = CM.shape[0]
    entropy = 0
    for i in range(N):
        for j in range(i + 1, N):
            Cij = CM[i, j]
            Cji = CM[j, i]
            if (Cij + Cji) > 0:
                entropy += (Cij / (Cij + Cji)) * np.log(Cij / (Cij + Cji)) + (Cji / (Cij + Cji)) * np.log(Cji / (Cij + Cji))
    entropy = -entropy / (N * (N - 1) / 2)
    return entropy

# Confusion matrix
CM = np.array([[0.65, 0.15, 0.20],
               [0.10, 0.70, 0.20],
               [0.15, 0.10, 0.75]])

# Calculate entropy for pairs (A,B), (A,C), (B,C)
entropy_A_B = calculate_entropy(CM[:2, :2])
entropy_A_C = calculate_entropy(CM[np.ix_([0, 2], [0, 2])])
entropy_B_C = calculate_entropy(CM[1:, 1:])

# Print the results 
print(f"Entropy values for the given confusion matrix:")
print(f"- Entropy for pair (A, B): {entropy_A_B:.4f}")
print(f"- Entropy for pair (A, C): {entropy_A_C:.4f}")
print(f"- Entropy for pair (B, C): {entropy_B_C:.4f}")

# Highlight which pair has the highest entropy
highest_entropy_pair = max(
    ("(A, B)", entropy_A_B),
    ("(A, C)", entropy_A_C),
    ("(B, C)", entropy_B_C),
    key=lambda x: x[1]
)
print(f"\nThe pair with the highest entropy is {highest_entropy_pair[0]} with an entropy of {highest_entropy_pair[1]:.4f}, "
      "indicating it is more likely to have sufficient training data.")
