import numpy as np
import matplotlib.pyplot as plt

a_min, a_max = 1.0, 1.4
step = 0.001
A = np.arange(a_min, a_max, step)

B = 0.3

X = np.zeros_like(A)
Y = np.zeros_like(A)

total_iterations = 1000
transient = 500

plt.figure(figsize=(10, 6))

for i in range(total_iterations):
    X_next = 1 - A * (X ** 2) + Y
    Y_next = B * X
    
    X, Y = X_next, Y_next
    
    if i >= transient:
        plt.plot(A, X, ',k', alpha=0.1)

plt.title('Bifurcation Diagram - Hénon Map')
plt.xlabel('Control Parameter (a)')
plt.ylabel('Retained $x_n$')
plt.xlim(a_min, a_max)
plt.tight_layout()
plt.show()