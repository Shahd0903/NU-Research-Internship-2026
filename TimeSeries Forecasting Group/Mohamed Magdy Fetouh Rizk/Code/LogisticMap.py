import numpy as np
import matplotlib.pyplot as plt

def plot_time_series(r=3.9, x0=0.5, total_iterations=1000, transient=200):
    n_values = []
    x_values = []
    x = x0
    
    for n in range(total_iterations):
        if n >= transient:
            n_values.append(n)
            x_values.append(x)
        x = r * x * (1 - x)
        
    plt.figure(figsize=(10, 5))
    plt.plot(n_values, x_values, marker='o', markersize=2, linestyle='-', linewidth=0.5, color='b')
    plt.title(f'Time-Series Plot for Logistic Map (r = {r})')
    plt.xlabel('Iteration (n)')
    plt.ylabel('State Variable (x_n)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_time_series(r=3.9, x0=0.5)