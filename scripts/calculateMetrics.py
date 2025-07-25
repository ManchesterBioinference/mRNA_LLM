#!/usr/bin/env python3
"""
Calculate metrics for SVM predictions and create scatter plot.

This script reads SVM predictions file (with true and predicted values),
calculates correlation metrics, and generates a scatter plot with 
correlation information and x=y line.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import os


def read_predictions_file(predictions_file):
    """
    Read SVM predictions file containing true and predicted values.
    
    Args:
        predictions_file (str): Path to predictions file (.tsv)
        
    Returns:
        tuple: (true_values, predicted_values) as numpy arrays
    """
    # Read the TSV file with no header
    # First column: true values, Second column: predicted values
    try:
        data = pd.read_csv(predictions_file, sep='\t', header=None, names=['true', 'predicted'])
        true_values = data['true'].values
        predicted_values = data['predicted'].values
        
        print(f"Loaded {len(true_values)} samples from {predictions_file}")
        
        return true_values, predicted_values
        
    except Exception as e:
        print(f"Error reading predictions file: {e}")
        
        # Fallback: try reading as space-separated or other formats
        try:
            data = pd.read_csv(predictions_file, sep=None, header=None, names=['true', 'predicted'], engine='python')
            true_values = data['true'].values
            predicted_values = data['predicted'].values
            
            print(f"Loaded {len(true_values)} samples from {predictions_file} (fallback format)")
            
            return true_values, predicted_values
            
        except Exception as e2:
            print(f"Failed to read file with fallback method: {e2}")
            raise


def calculate_metrics(predictions, true_labels):
    """
    Calculate Pearson and Spearman correlations.
    
    Args:
        predictions (numpy.array): Predicted values
        true_labels (numpy.array): True values
        
    Returns:
        dict: Dictionary containing correlation metrics
    """
    # Calculate correlations
    pearson_corr, pearson_pval = pearsonr(predictions, true_labels)
    spearman_corr, spearman_pval = spearmanr(predictions, true_labels)
    
    # Calculate R-squared
    correlation_matrix = np.corrcoef(predictions, true_labels)
    r_squared = correlation_matrix[0, 1] ** 2
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions - true_labels) ** 2))
    
    # Calculate MAE
    mae = np.mean(np.abs(predictions - true_labels))
    
    metrics = {
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_pval,
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_pval,
        'r_squared': r_squared,
        'rmse': rmse,
        'mae': mae,
        'n_samples': len(predictions)
    }
    
    return metrics


def create_scatter_plot(predictions, true_labels, metrics, output_path):
    """
    Create scatter plot with correlation metrics and x=y line.
    
    Args:
        predictions (numpy.array): Predicted values
        true_labels (numpy.array): True values
        metrics (dict): Dictionary containing correlation metrics
        output_path (str): Path to save the plot
    """
    # Set up the plot with a larger figure size
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(true_labels, predictions, alpha=0.6, s=20, color='steelblue', edgecolors='none')
    
    # Add x=y line (perfect prediction line) in red
    min_val = min(np.min(true_labels), np.min(predictions))
    max_val = max(np.max(true_labels), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Perfect prediction (x=y)')
    
    # Add regression line
    z = np.polyfit(true_labels, predictions, 1)
    p = np.poly1d(z)
    plt.plot(true_labels, p(true_labels), "--", alpha=0.8, color='orange', linewidth=1.5, label='Regression line')
    
    # Set labels and title
    plt.xlabel('True Decay Rate', fontsize=12)
    plt.ylabel('Predicted Decay Rate', fontsize=12)
    plt.title('SVM Predictions vs True Decay Rates', fontsize=14, fontweight='bold')
    
    # Add correlation metrics as text box
    textstr = '\n'.join([
        f'Pearson r = {metrics["pearson_correlation"]:.4f} (p = {metrics["pearson_pvalue"]:.2e})',
        f'Spearman ρ = {metrics["spearman_correlation"]:.4f} (p = {metrics["spearman_pvalue"]:.2e})',
        f'R² = {metrics["r_squared"]:.4f}',
        f'RMSE = {metrics["rmse"]:.4f}',
        f'MAE = {metrics["mae"]:.4f}',
        f'N = {metrics["n_samples"]}'
    ])
    
    # Position text box in upper left
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Make sure the plot is square and axis limits are equal
    plt.axis('equal')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter plot saved to: {output_path}")


def write_metrics_file(metrics, output_path):
    """
    Write metrics to a text file.
    
    Args:
        metrics (dict): Dictionary containing correlation metrics
        output_path (str): Path to save the metrics file
    """
    with open(output_path, 'w') as f:
        f.write("SVM Prediction Metrics\n")
        f.write("=====================\n\n")
        f.write(f"Number of samples: {metrics['n_samples']}\n\n")
        f.write("Correlation Metrics:\n")
        f.write(f"  Pearson correlation: {metrics['pearson_correlation']:.6f}\n")
        f.write(f"  Pearson p-value: {metrics['pearson_pvalue']:.6e}\n")
        f.write(f"  Spearman correlation: {metrics['spearman_correlation']:.6f}\n")
        f.write(f"  Spearman p-value: {metrics['spearman_pvalue']:.6e}\n\n")
        f.write("Other Metrics:\n")
        f.write(f"  R-squared: {metrics['r_squared']:.6f}\n")
        f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"  MAE: {metrics['mae']:.6f}\n")
    
    print(f"Metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for SVM predictions')
    parser.add_argument('predictions_file', help='Path to SVM predictions file (.tsv)')
    parser.add_argument('metrics_output', help='Path to output metrics file (.txt)')
    parser.add_argument('--plot_output', help='Path to output scatter plot (.png)', 
                       default=None)
    
    args = parser.parse_args()
    
    # Generate plot output path if not provided
    if args.plot_output is None:
        base_path = args.predictions_file.replace('_predictions.tsv', '')
        args.plot_output = f"{base_path}_scatter_plot.png"
    
    print(f"Reading predictions from: {args.predictions_file}")
    
    # Check if files exist
    if not os.path.exists(args.predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {args.predictions_file}")
    
    # Read data
    true_labels, predictions = read_predictions_file(args.predictions_file)
    
    print(f"Loaded {len(predictions)} predictions and {len(true_labels)} true labels")
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, true_labels)
    
    # Print metrics to console
    print("\nMetrics:")
    print(f"  Pearson correlation: {metrics['pearson_correlation']:.4f} (p = {metrics['pearson_pvalue']:.2e})")
    print(f"  Spearman correlation: {metrics['spearman_correlation']:.4f} (p = {metrics['spearman_pvalue']:.2e})")
    print(f"  R-squared: {metrics['r_squared']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    
    # Write metrics file
    write_metrics_file(metrics, args.metrics_output)
    
    # Create scatter plot
    create_scatter_plot(predictions, true_labels, metrics, args.plot_output)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
