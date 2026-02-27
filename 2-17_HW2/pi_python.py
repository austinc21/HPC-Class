import time
from numpy.random import rand
import sys
def calc_pi_loop(n):
    h = 0 # Number of hits inside the circle
    for _ in range(n):
        x, y = rand(), rand() # Random points in [0, 1)
    if x*x + y*y < 1.:
        h += 1 # Successful hit
    return 4. * float(h) / float(n) # Estimate pi

if __name__ == "__main__":
    start_time = time.perf_counter()
    if len(sys.argv) == 1:
        n = 10000000
    elif len(sys.argv) == 2:
        n = int(sys.argv[1]) # Command-line argument
    else:
        print("Usage: python pi_loop.py <n>")
        sys.exit(1)
    pi_est = calc_pi_loop(n)
    end_time = time.perf_counter()
    print(f"n={n}, pi={pi_est}")
    print(f"Execution time: {end_time - start_time:.6f} seconds")