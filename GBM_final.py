import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="GBM Slope Field Visualizer", layout="wide")

st.title("Interactive GBM Slope Field Visualizer")
st.markdown("Explore Geometric Brownian Motion with adjustable parameters")

# Sidebar controls
st.sidebar.header("Parameters")
mu = st.sidebar.slider("Drift (μ)", -0.2, 0.5, 0.15, 0.01)
sigma = st.sidebar.slider("Volatility (σ)", 0.05, 0.6, 0.25, 0.01)
num_paths = st.sidebar.slider("Number of Paths", 1, 6, 5, 1)

# Initialize seed in session state
if 'seed' not in st.session_state:
    st.session_state.seed = int(time.time() * 1000)

if st.sidebar.button("🔄 Generate New Paths"):
    st.session_state.seed = int(time.time() * 1000)

def seeded_random(s, size):
    rng = s
    result = []
    for _ in range(size):
        rng = (rng * 9301 + 49297) % 233280
        result.append(rng / 233280)
    return np.array(result)

def plot_gbm(mu, sigma, num_paths, seed):
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
            
            slope = mu * S
            dt_dir = 0.024
            dS_dir = slope * dt_dir
            
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
    return fig

# Generate and display plot
fig = plot_gbm(mu, sigma, num_paths, st.session_state.seed)
st.pyplot(fig)

# Add explanation
st.markdown("""
---
### Understanding the Visualization
- **Blue arrows**: Show the deterministic drift field (μS)  
- **Colored lines**: Sample paths showing possible price trajectories  
- **Steeper slopes at higher prices**: Price changes are proportional to current price level

### Equation
The Geometric Brownian Motion follows:  
**dS = μS dt + σS dW**

Where:
- **μ** = drift (expected return rate)
- **σ** = volatility (standard deviation of returns)
- **dW** = Wiener process (random component)
""")