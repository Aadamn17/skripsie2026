#!/usr/bin/env python3
"""
plot_losses.py

Reads a per-epoch loss log (default: logs/per_epoch_loss.txt),
groups epochs into runs, and saves train/dev/test loss curves as PNG files.

Usage (from project root):
    python src/plot_losses.py                  # saves loss_plot_run1.png
    python src/plot_losses.py 3                # saves loss_plot_run3.png
    python src/plot_losses.py all              # saves all_loss_plots.png

Usage (from src/):
    python plot_losses.py                      # uses ../logs/per_epoch_loss.txt
"""

import sys
import re
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless saving
import matplotlib.pyplot as plt

def find_default_log():
    """Return the first existing log file among common locations."""
    candidates = [
        "logs/per_epoch_loss.txt",
        "../logs/per_epoch_loss.txt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]

def parse_log(file_path):
    """Parse log file and return a list of runs (each is a list of epoch dicts)."""
    runs = []
    current_run = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pattern = (
                r"Epoch (\d+)/(\d+), Train Loss: ([\d.]+), Dev Loss: ([\d.]+), "
                r"Test Loss: ([\d.]+), Dev Acc: ([\d.]+), Dev AUC: ([\d.]+), "
                r"Test Acc: ([\d.]+), Test AUC: ([\d.]+)"
            )
            m = re.match(pattern, line)
            if not m:
                continue
            epoch = int(m.group(1))
            train_loss = float(m.group(3))
            dev_loss = float(m.group(4))
            test_loss = float(m.group(5))
            if epoch == 1:
                current_run = []
                runs.append(current_run)
            if current_run is not None:
                current_run.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'dev_loss': dev_loss,
                    'test_loss': test_loss,
                })
    return runs

def plot_single_run(run, run_number, ax=None):
    """Plot losses for a single run on the given axes (or create new)."""
    epochs = [e['epoch'] for e in run]
    train_losses = [e['train_loss'] for e in run]
    dev_losses = [e['dev_loss'] for e in run]
    test_losses = [e['test_loss'] for e in run]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=4)
    ax.plot(epochs, dev_losses, label='Dev Loss', marker='s', markersize=4)
    ax.plot(epochs, test_losses, label='Test Loss', marker='^', markersize=4)

    ax.set_title(f'Run #{run_number}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

def main():
    # Parse arguments: [log_path] [run_number_or_all]
    args = sys.argv[1:]
    log_file = find_default_log()
    run_spec = "1"  # default

    if len(args) >= 1:
        # If the first arg exists as a file, treat it as log path; otherwise treat as run spec
        if os.path.exists(args[0]):
            log_file = args[0]
            if len(args) >= 2:
                run_spec = args[1]
        else:
            run_spec = args[0]

    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.", file=sys.stderr)
        sys.exit(1)

    runs = parse_log(log_file)
    if not runs:
        print("No valid epochs found in the log file.")
        sys.exit(0)

    if run_spec == "all":
        # Plot all runs in subplots and save to a single file
        n_runs = len(runs)
        cols = 3
        rows = (n_runs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten()
        for i, run in enumerate(runs):
            plot_single_run(run, i+1, ax=axes[i])
        # Hide unused axes
        for j in range(n_runs, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        output_file = "all_loss_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        plt.close(fig)
    else:
        try:
            run_num = int(run_spec)
        except ValueError:
            print(f"Invalid run specifier: '{run_spec}'. Use a number or 'all'.", file=sys.stderr)
            sys.exit(1)
        if run_num < 1 or run_num > len(runs):
            print(f"Error: Run number must be between 1 and {len(runs)}.")
            sys.exit(1)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_single_run(runs[run_num - 1], run_num, ax)
        output_file = f"loss_plot_run{run_num}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        plt.close(fig)

if __name__ == "__main__":
    main()