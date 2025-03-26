import cupy as cp
import time

# Define matrix size
N = 1000

# Create two random matrices on the GPU
a = cp.random.rand(N, N)
b = cp.random.rand(N, N)

# Perform matrix multiplication on the GPU
start_time = time.time()
c = cp.dot(a, b)
cp.cuda.Device(0).synchronize()  # Ensure computation is complete
end_time = time.time()

print(
    f"Matrix multiplication of size {N}x{N} completed on GPU in {end_time - start_time:.4f} seconds."
)