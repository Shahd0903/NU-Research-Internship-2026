import matplotlib.pyplot as plt

def plot_henon_attractor(a=1.4, b=0.3, x0=0.0, y0=0.0, total_iterations=10000, transient=500):
    x_vals = []
    y_vals = []
    
    x, y = x0, y0
    
    for n in range(total_iterations):
        x_next = 1 - a * (x ** 2) + y
        y_next = b * x
        
        if n >= transient:
            x_vals.append(x_next)
            y_vals.append(y_next)
            
        x, y = x_next, y_next

    plt.figure(figsize=(8, 6))
    
    plt.scatter(x_vals, y_vals, s=0.5, color='black', marker='.')
    
    plt.title(f'Hénon Map Attractor (a={a}, b={b})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_henon_attractor(a=1.4, b=0.3, x0=0.0, y0=0.0, total_iterations=10000, transient=500)