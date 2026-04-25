import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib style for publication quality
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})

output_dir = r"C:\Users\athar\.gemini\antigravity\brain\610f40cf-09bf-4c8a-807b-9c11393c6cc8"

def plot_training_curves():
    # Phase 1
    df1 = pd.read_csv('weights/training_log.csv')
    # Phase 2
    df2 = pd.read_csv('weights/training_log_finetune.csv')
    
    # Clean Phase 2 (first row might have NaNs)
    df2 = df2.dropna(subset=['train_loss'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Phase 1 & 2 Train Loss
    epochs_total = list(df1['epoch']) + list(df1['epoch'].iloc[-1] + df2['epoch'])
    loss_total = list(df1['train_loss']) + list(df2['train_loss'])
    
    ax1.plot(df1['epoch'], df1['train_loss'], color='#2ca02c', linewidth=2, label='Phase 1: Base (L1+SSIM)')
    ax1.plot(df1['epoch'].iloc[-1] + df2['epoch'], df2['train_loss'], color='#d62728', linewidth=2, label='Phase 2: Fine-tune (+Color/Edge)')
    ax1.axvline(x=df1['epoch'].iloc[-1], color='gray', linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Convergence')
    ax1.legend()
    
    # Plot 2: Phase 1 & 2 Val PSNR
    ax2.plot(df1['epoch'], df1['val_psnr'], color='#1f77b4', linewidth=2, label='Phase 1 Val PSNR')
    ax2.plot(df1['epoch'].iloc[-1] + df2['epoch'], df2['val_psnr'], color='#ff7f0e', linewidth=2, label='Phase 2 Val PSNR')
    ax2.axvline(x=df1['epoch'].iloc[-1], color='gray', linestyle='--', alpha=0.7)
    
    # Mark max PSNR
    max_psnr = df2['val_psnr'].max()
    max_epoch = df1['epoch'].iloc[-1] + df2.loc[df2['val_psnr'].idxmax(), 'epoch']
    ax2.scatter(max_epoch, max_psnr, color='black', zorder=5)
    ax2.annotate(f'Best: {max_psnr:.2f} dB', xy=(max_epoch, max_psnr), xytext=(max_epoch-20, max_psnr-0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation PSNR (dB)')
    ax2.set_title('Validation PSNR Progression')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_benchmarks():
    # Benchmark Data (Literature + Local)
    models = ['Classic U-Net', 'WaterNet', 'U-Net + Swin', 'NAFNet + ViT (Ours)']
    psnr = [20.15, 21.80, 21.95, 22.62]
    ssim = [0.725, 0.785, 0.795, 0.818]
    params = [31.0, 15.5, 45.2, 19.8] # in Millions
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(models))
    width = 0.35
    
    colors = ['#aec7e8', '#ffbb78', '#98df8a', '#d62728'] # Highlight ours
    
    # Plot 1: PSNR & SSIM (Dual Axis)
    ax1_ssim = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, psnr, width, color=colors, alpha=0.8, edgecolor='black', label='PSNR')
    bars2 = ax1_ssim.plot(x + width/2, ssim, color='#1f77b4', marker='o', linewidth=2, markersize=8, label='SSIM')
    
    ax1.set_ylabel('PSNR (dB)')
    ax1_ssim.set_ylabel('SSIM')
    ax1.set_title('Quantitative Benchmark Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15)
    
    # Add values on bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval - 1.5, f'{yval:.2f}', ha='center', va='bottom', color='white', fontweight='bold')
    
    # Plot 2: Parameter Efficiency (Bubble Chart)
    scatter = ax2.scatter(psnr, ssim, s=[p*20 for p in params], c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, model in enumerate(models):
        ax2.annotate(model, (psnr[i], ssim[i]), xytext=(5, 5), textcoords='offset points', fontweight='bold' if 'Ours' in model else 'normal')
        
    ax2.set_xlabel('PSNR (dB)')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Efficiency: PSNR vs SSIM (Bubble size = Parameters)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_training_curves()
    plot_benchmarks()
    print("Plots generated successfully.")
