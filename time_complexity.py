import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# User-defined number of points
n_points = 100

# Define log-spaced matrix sizes
sizes = np.logspace(np.log10(10), np.log10(10000), n_points, dtype=int)
matrix_exp_times = []
matrix_mult_times = []

# Warmup phase
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warmup_matrix = torch.randn(10, 10).to(device)
torch.matrix_exp(warmup_matrix)  # Warmup for matrix exponential
warmup_matrix @ warmup_matrix  # Warmup for matrix multiplication

# Perform timing tests
for size in sizes:
    matrix = torch.randn(size, size).to(device)
    t_matrix = torch.randn(size, size).to(device)

    # Time torch.matrix_exp
    start_time = time.time()
    torch.matrix_exp(matrix)
    torch.cuda.synchronize() if device.type == "cuda" else None
    matrix_exp_time = time.time() - start_time

    # Time matrix multiplication
    start_time = time.time()
    matrix @ t_matrix
    torch.cuda.synchronize() if device.type == "cuda" else None
    matrix_mult_time = time.time() - start_time

    matrix_exp_times.append(matrix_exp_time)
    matrix_mult_times.append(matrix_mult_time)

# Convert to log-log scale for regression
log_sizes = np.log10(sizes).reshape(-1, 1)
log_exp_times = np.log10(matrix_exp_times)
log_mult_times = np.log10(matrix_mult_times)

# Split into regimes
split_index_small = np.searchsorted(sizes, 500)
split_index_large = np.searchsorted(sizes, 1000)

small_log_sizes = log_sizes[:split_index_small]
medium_log_sizes = log_sizes[split_index_large:]
small_log_exp_times = log_exp_times[:split_index_small]
medium_log_exp_times = log_exp_times[split_index_large:]
small_log_mult_times = log_mult_times[:split_index_small]
medium_log_mult_times = log_mult_times[split_index_large:]

# Perform linear regression in both regimes
reg_exp_small = LinearRegression().fit(small_log_sizes, small_log_exp_times)
reg_exp_large = LinearRegression().fit(medium_log_sizes, medium_log_exp_times)

reg_mult_small = LinearRegression().fit(small_log_sizes, small_log_mult_times)
reg_mult_large = LinearRegression().fit(medium_log_sizes, medium_log_mult_times)

# Get slopes
exp_slope_small = reg_exp_small.coef_[0]
exp_slope_large = reg_exp_large.coef_[0]

mult_slope_small = reg_mult_small.coef_[0]
mult_slope_large = reg_mult_large.coef_[0]

# Plot results on log-log scale
plt.figure(figsize=(12, 8))
plt.loglog(sizes, matrix_exp_times, label='torch.matrix_exp', marker='o', linestyle='-', color='cyan')
plt.loglog(sizes, matrix_mult_times, label='Matrix multiplication', marker='s', linestyle='-', color='orange')

# Add regression lines
# plt.plot(
#     10**small_log_sizes.flatten(), 10**reg_exp_small.predict(small_log_sizes),
#     linestyle='--', color='blue', label=f'matrix_exp small regime (slope={exp_slope_small:.2f})'
# )
plt.plot(
    10**medium_log_sizes.flatten(), 10**reg_exp_large.predict(medium_log_sizes),
    linestyle='--', color='blue', label=f'matrix_exp large regime (slope={exp_slope_large:.2f})'
)
# plt.plot(
#     10**small_log_sizes.flatten(), 10**reg_mult_small.predict(small_log_sizes),
#     linestyle='--', color='red', label=f'matrix_mult small regime (slope={mult_slope_small:.2f})'
# )
plt.plot(
    10**medium_log_sizes.flatten(), 10**reg_mult_large.predict(medium_log_sizes),
    linestyle='--', color='red', label=f'matrix_mult large regime (slope={mult_slope_large:.2f})'
)

# Plot a vertical green line to indicate the boundary between the two regimes
# at MNIST size of 784
plt.axvline(x=784, color='green', linestyle='--', label=r'MNIST ($N=784$)')

plt.xlabel('Square Matrix Size (N)')
plt.ylabel('Time (seconds)')
plt.title('Scaling Behavior: torch.matrix_exp vs Matrix Multiplication')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()





