"""
plotter.py
A separate module for handling all plotting to keep the main script clean.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def plot_loss_history(history_df, save_path):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    
    loss_df = history_df.melt(
        id_vars=['epoch'], 
        value_vars=['train_loss', 'val_loss'], 
        var_name='Loss Type', 
        value_name='Loss'
    )
    
    sns.lineplot(data=loss_df, x='epoch', y='Loss', hue='Loss Type', marker='o')
    plt.title(f'Loss vs. Epoch (Classes: {save_path.split("_")[-1].split(".")[0]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(save_path) 
    plt.close(fig) 
    print(f"Saved training loss plot to '{save_path}'")

def plot_accuracy_history(history_df, save_path):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=history_df, x='epoch', y='val_accuracy', marker='o', color='green')
    plt.title(f'Val. Accuracy vs. Epoch (Classes: {save_path.split("_")[-1].split(".")[0]})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(save_path) 
    plt.close(fig)
    print(f"Saved validation accuracy plot to '{save_path}'")

def plot_confusion_matrix(y_true_class, y_pred_class, num_classes, save_path):
    if num_classes > 72: # Plotting 120 or 720 classes is unreadable
        print(f"Skipping confusion matrix: {num_classes} classes is too large to plot.")
        return
        
    cm = confusion_matrix(y_true_class, y_pred_class)
    fig = plt.figure(figsize=(max(14, num_classes // 2), max(10, num_classes // 3)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    if num_classes == 24:
        labels = [f"{h}:{m:02d}" for h in range(12) for m in (0, 30)]
    elif num_classes == 72: # Example: 10-minute intervals
        labels = [f"{h}:{m:02d}" for h in range(12) for m in range(0, 60, 10)]
    else: # Fallback for other small class counts
        labels = [str(i) for i in range(num_classes)]

    plt.title(f'Confusion Matrix ({num_classes} Classes)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(ticks=np.arange(num_classes) + 0.5, labels=labels, rotation=90)
    plt.yticks(ticks=np.arange(num_classes) + 0.5, labels=labels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved confusion matrix plot to '{save_path}'")

def plot_summary_results(summary_df, save_dir):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Accuracy Plot
    acc_df = summary_df.melt(
        id_vars=['num_classes'], 
        value_vars=['train_acc', 'test_acc'], 
        var_name='Accuracy Type', 
        value_name='Accuracy (%)'
    )
    sns.lineplot(data=acc_df, x='num_classes', y='Accuracy (%)', hue='Accuracy Type', marker='o', ax=ax1)
    ax1.set_title('Final Accuracy vs. Number of Classes')
    ax1.set_xlabel('Number of Classes')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xscale('log') # Use log scale for x-axis as classes are not linear

    # Common Sense Error Plot
    sns.lineplot(data=summary_df, x='num_classes', y='common_sense_error', marker='o', color='red', ax=ax2)
    ax2.set_title('Final "Common Sense" Error vs. Number of Classes')
    ax2.set_xlabel('Number of Classes')
    ax2.set_ylabel('Mean Error (minutes)')
    ax2.set_xscale('log') # Use log scale

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ablation_summary_plot.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved summary ablation plot to '{save_path}'")