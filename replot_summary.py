import pandas as pd
import os
import glob # Used to find files
import re   # Used to extract numbers from filenames
import plotter  # Your existing plotter.py file

# --- 1. Define Directories ---
BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'Results')
PLOTS_DIR = os.path.join(BASE_DIR, 'Plots')

# --- 2. Re-plot all INDIVIDUAL class results ---
print("--- Re-plotting individual class histories ---")

# Find all individual history files
history_files = glob.glob(os.path.join(RESULTS_DIR, 'history_cls_*.csv'))

if not history_files:
    print("No individual 'history_cls_*.csv' files found in /Results.")
else:
    for csv_path in history_files:
        # Extract the number of classes from the filename
        # e.g., "history_cls_24.csv" -> "24"
        match = re.search(r'history_cls_(\d+)', csv_path)
        if not match:
            print(f"Skipping file with unexpected name: {csv_path}")
            continue
            
        num_classes = int(match.group(1))
        print(f"Processing results for {num_classes} classes...")
        
        # Load the individual history data
        history_df = pd.read_csv(csv_path)
        
        # Call the loss plotter
        plotter.plot_loss_history(
            history_df, 
            os.path.join(PLOTS_DIR, f'plot_loss_cls_{num_classes}.png')
        )
        
        # Call the accuracy plotter
        plotter.plot_accuracy_history(
            history_df,
            os.path.join(PLOTS_DIR, f'plot_acc_cls_{num_classes}.png')
        )

# --- 3. Re-plot SUMMARY results ---
print("\n--- Re-plotting summary ablation graphs ---")

# Define the summary file path
SUMMARY_CSV_FILENAME = 'ablation_summary.csv' 
SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, SUMMARY_CSV_FILENAME)

# Load the summary data
try:
    summary_df = pd.read_csv(SUMMARY_CSV_PATH)
except FileNotFoundError:
    print(f"Error: Summary file not found at {SUMMARY_CSV_PATH}")
    print("Cannot generate summary plots. Exiting.")
    exit()

# Generate the summary plots
print("Generating summary plots...")

# Call the two summary functions from plotter
plotter.plot_summary_accuracy(summary_df, PLOTS_DIR)
plotter.plot_summary_error(summary_df, PLOTS_DIR)

print(f"\nAll plots saved to {PLOTS_DIR}")
print("Done.")