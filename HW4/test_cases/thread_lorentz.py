import numpy as np
def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Sample n random points from the Lorentzian distribution
    using inverse transform sampling. Make a histogram with
    the specified bin count and range. Returns counts.
    """
    u = np.random.random(n)  # Uniform(0,1)
    x = 1. / np.tan(np.pi * u)  # x = 1/tan(pi*u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts  # No need to return bin edges for uniform bins

# thread_lorentz.py
import threading
def add_chunk(n, counts, lock, bins=100, xmin=-10, xmax=10):
    """
    Generate n samples and add to global counts.
    """
    local_counts = lorentzian_histogram(n, bins, xmin, xmax)
    # Acquire lock to merge partial counts into global
    with lock:
        counts += local_counts
def run_threaded(n, n_threads=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using threads.
    """
    # Split n samples among processes
    chunks = (n // n_threads) * np.ones(n_threads, dtype=int)
    chunks[:n % n_threads] += 1 # Distribute remainder
    threads = [None] * n_threads # Thread list
    counts = np.zeros(bins) # Global counts
    lock = threading.Lock() # Lock for global data
    for i in range(n_threads):
        t = threading.Thread(target=add_chunk, args=(chunks[i], counts, lock, bins, xmin, xmax))
        t.start() # Start thread
        threads[i] = t
    for t in threads:
        t.join() # Wait for all threads to finish
    return counts


N = 100