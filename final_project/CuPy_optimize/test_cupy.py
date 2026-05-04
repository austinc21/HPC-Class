import cupy as cp
import numpy as np

# Test basic array creation
x = cp.array([1, 2, 3, 4, 5])
print("CuPy array:", x)

# Test computation
y = x ** 2
print("Squared:", y)

# Compare with NumPy
x_np = np.array([1, 2, 3, 4, 5])
y_np = x_np ** 2
print("NumPy squared:", y_np)

# Check if results match
print("Results match:", cp.allclose(y, cp.asarray(y_np)))