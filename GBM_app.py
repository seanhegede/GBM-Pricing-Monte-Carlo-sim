import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Button, VBox, interactive_output, HBox
from IPython.display import Image, display, clear_output
import io
import time

# Global seed
seed = int(time.time() * 1000)

def seeded_random(s, size):
    rng = s
    result = []
    for _ in range(size):
        rng = (rng * 9301 + 49297) % 233280
        result.append(rng / 233280)
    return np.array(result)

def plot_gbm(mu=0.15, sigma=0.25, num_paths=5):
    global seed
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#fafafa')
    
    t_min, t_max = 0, 1
    S_min, S_max = 50, 200
    
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(S_min, S_max)
    ax.set_xlabel('Time (t)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Stock Price (S)', fontsize=14, fontweight='bold')
    ax.set_title('GBM Slope Field: dS = μS dt + σS dW', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, color='#e0e0e0')
    ax.set_facecolor('#fafafa')
    
    # Draw slope field
    grid_t, grid_s = 25, 20
    
    for i in range(1, grid_t):
        for j in range(1, grid_s):
            t = t_min + (i / grid_t) * (t_max - t_min)
            S = S_min + (j / grid_s) * (S_max - S_min)
            
            # Slope from deterministic part: dS/dt = μS
            slope = mu * S
            
            # Direction vector - normalized
            dt_dir = 1.0
            dS_dir = slope
            
            # Fixed arrow length for uniform appearance
            arrow_length = 0.015
            
            # Calculate direction (not normalized by magnitude)
            # This preserves the relative steepness at different S values
            dt_dir = 0.024
            dS_dir = slope * dt_dir
            
            # Draw simple line segment (no arrowhead)
            ax.plot([t - dt_dir/2, t + dt_dir/2], [S - dS_dir/2, S + dS_dir/2],
                   color='#2563eb', linewidth=1.5, alpha=0.7)
    
    # Generate paths
    colors = ['#dc2626', '#16a34a', '#2563eb', '#ea580c', '#9333ea', '#ca8a04']
    steps = 250
    dt = (t_max - t_min) / steps
    
    total_randoms = steps * num_paths
    randoms = seeded_random(seed, total_randoms)
    random_idx = 0
    
    for p in range(num_paths):
        S = 100
        t_vals = [0]
        S_vals = [S]
        
        for i in range(1, steps + 1):
            t = i * dt
            dW = np.sqrt(dt) * (randoms[random_idx] * 2 - 1) * np.sqrt(3)
            random_idx += 1
            S = S * (1 + mu * dt + sigma * dW)
            S = max(S_min + 5, min(S_max - 5, S))
            
            t_vals.append(t)
            S_vals.append(S)
        
        ax.plot(t_vals, S_vals, color=colors[p % len(colors)],
                linewidth=2.5, alpha=0.8)
    
    # Info box
    info_text = (
        f'Parameters:\n'
        f'μ (drift) = {mu:.3f}\n'
        f'σ (volatility) = {sigma:.3f}\n\n'
        f'Blue arrows: Drift field (μS)\n'
        f'Colored lines: Sample paths'
    )
    ax.text(0.98, 0.97, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='#ccc'))
    
    plt.tight_layout()
    
    # Save to buffer and display as image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    # Display the image
    display(Image(buf.read()))

# Create sliders
slider_mu = FloatSlider(value=0.15, min=-0.2, max=0.5, step=0.01, 
                       description='Drift (μ):', style={'description_width': '120px'},
                       layout={'width': '500px'})
slider_sigma = FloatSlider(value=0.25, min=0.05, max=0.6, step=0.01, 
                          description='Volatility (σ):', style={'description_width': '120px'},
                          layout={'width': '500px'})
slider_paths = IntSlider(value=5, min=1, max=6, step=1, 
                        description='# Paths:', style={'description_width': '120px'},
                        layout={'width': '500px'})

# Create button
button = Button(description='🔄 Generate New Paths', button_style='success',
               layout={'width': '200px', 'height': '40px'})

# Output widget
output = interactive_output(plot_gbm, {'mu': slider_mu, 'sigma': slider_sigma, 'num_paths': slider_paths})

def on_button_click(b):
    global seed
    seed = int(time.time() * 1000)
    # Trigger update by changing slider slightly and back
    old_val = slider_mu.value
    slider_mu.value = old_val + 0.001
    slider_mu.value = old_val

button.on_click(on_button_click)

# Layout
controls = VBox([
    slider_mu,
    slider_sigma, 
    slider_paths,
    button
])

# Display
print("Interactive GBM Slope Field - Use the sliders and button below!")
print("-" * 70)
display(VBox([controls, output]))
print("\n" + "=" * 70)
print("Deterministic Slope Field Equation: dy/dx = cy")
print("Steeper lines at higher S values show that price changes are proportional to current price")
print("=" * 70)