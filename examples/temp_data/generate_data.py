import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for linear regression
n_samples = 1000
n_features = 5

# Generate feature matrix X
X = np.random.randn(n_samples, n_features)

# Generate true coefficients
true_coefficients = np.array([2.0, -1.5, 0.8, 1.2, -0.5])

# Generate target variable y with some noise
noise = np.random.randn(n_samples) * 0.1
y = np.dot(X, true_coefficients) + noise

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
df['target'] = y

# Save to CSV
df.to_csv('sample_data.csv', index=False)
print("Data generated and saved to sample_data.csv") 