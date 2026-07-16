import numpy as np
import matplotlib.pyplot as plt

r_min, r_max = 2.5, 4.0
step = 0.001
R = np.arange(r_min, r_max, step)

X = np.full_like(R, 0.5)

total_iterations = 1000
transient = 500

plt.figure(figsize=(10, 6))

for i in range(total_iterations):
    X = R * X * (1 - X)
    
    if i >= transient:
        plt.plot(R, X, ',k', alpha=0.1)

plt.title('Bifurcation Diagram - Logistic Map')
plt.xlabel('Control Parameter (r)')
plt.ylabel('Retained $x_n$')
plt.xlim(r_min, r_max)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()