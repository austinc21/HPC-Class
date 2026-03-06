import numpy as np
import matplotlib.pyplot as plt

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

def lorentzian_samples(n):
    u = np.random.random(n)
    x = 1. / np.tan(np.pi * u)
    return x

N = 1000
counts = lorentzian_histogram(N)

x = np.linspace(-10, 10, 100)
bin_width = 20 / 100

density = counts / (N * bin_width)
pdf = 1 / (np.pi * (1 + x**2))

plt.bar(x, density, width=bin_width)
plt.plot(x, pdf, color='red', label='Theoretical PDF')
plt.show()

# Comparing Quantiles
x = lorentzian_samples(N)

sample_q1 = np.percentile(x, 25)
sample_median = np.percentile(x, 50)
sample_q3 = np.percentile(x, 75)

theory_q1 = np.tan(np.pi*(0.25 - 0.5))
theory_median = np.tan(np.pi*(0.50 - 0.5))
theory_q3 = np.tan(np.pi*(0.75 - 0.5))

print("Sample Q1:", sample_q1)
print("Theoretical Q1:", theory_q1)

print("Sample median:", sample_median)
print("Theoretical median:", theory_median)

print("Sample Q3:", sample_q3)
print("Theoretical Q3:", theory_q3)

# Verifying CDF yields Uniform(0,1)
x = lorentzian_samples(N)

u_from_x = 0.5 + np.arctan(x)/np.pi

plt.hist(u_from_x, bins=20, range=(0,1))
plt.show()