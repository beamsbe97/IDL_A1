"""
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

def plot_summary_accuracy(summary_df, save_dir):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10, 7))
    
    # Create a categorical x-axis
    plot_df = summary_df.copy()
    plot_df['num_classes_str'] = plot_df['num_classes'].astype(str)
    plot_df = plot_df.sort_values(by='num_classes')

    # Melt data for plotting
    acc_df = plot_df.melt(
        id_vars=['num_classes_str'], 
        value_vars=['train_acc', 'test_acc'], 
        var_name='Accuracy Type', 
        value_name='Accuracy (%)'
    )
    
    # Use pointplot for a clearer categorical visualization
    sns.pointplot(data=acc_df, x='num_classes_str', y='Accuracy (%)', hue='Accuracy Type', dodge=True)
    ax = plt.gca()
    ax.set_title('Final Accuracy vs. Number of Classes')
    ax.set_xlabel('Number of Classes (Categorical)')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(title='Accuracy Type')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ablation_summary_accuracy.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved summary accuracy plot to '{save_path}'")

def plot_summary_error(summary_df, save_dir):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10, 7))
    
    # Create a categorical x-axis
    plot_df = summary_df.copy()
    plot_df['num_classes_str'] = plot_df['num_classes'].astype(str)
    plot_df = plot_df.sort_values(by='num_classes')
    
    # Use pointplot here as well
    sns.pointplot(data=plot_df, x='num_classes_str', y='common_sense_error', color='red')
    ax = plt.gca()
    ax.set_title('Final "Common Sense" Error vs. Number of Classes')
    ax.set_xlabel('Number of Classes (Categorical)')
    ax.set_ylabel('Mean Error (minutes)')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ablation_summary_error.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved summary error plot to '{save_path}'")