from numpy import sum
from numpy.random import rand
import sys
def calc_pi_numpy(n):
    h = sum(rand(n)**2 + rand(n)**2 < 1.)
    return 4. * float(h) / float(n) # Estimate pi
if __name__ == "__main__":
    if len(sys.argv) == 1:
        n = 1000000
    elif len(sys.argv) == 2:
        n = int(sys.argv[1]) # Command-line argument
    else:
        print("Usage: python pi_loop.py <n>")
        sys.exit(1)
    pi_est = calc_pi_numpy(n)
    print(f"n={n}, pi={pi_est}")