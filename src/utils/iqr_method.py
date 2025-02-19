import numpy as np


def filter_by_quartiles(data, multiplier=1.5):
    """
    Filter data based on quartiles (IQR method)

    Args:
        data: numpy array of values
        multiplier: IQR multiplier (default 1.5 for standard outlier detection)
                   use 3.0 for extreme outlier detection
    Returns:
        filtered_data: data with outliers removed
        mask: boolean mask indicating valid values
    """
    # Calculate quartiles
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)  # 100th percentile is the max value

    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Define bounds
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Create mask for valid values
    mask = (data >= lower_bound) & (data <= upper_bound)

    # Filter data
    filtered_data = data[mask]

    return filtered_data, mask


# Example usage in your context:
def calculate_filtered_mean(metrics, print_stats=True):
    """
    Calculate filtered mean of metrics removing outliers

    Args:
        metrics: numpy array of metrics (e.g., sti_hats, alcons_hats)
        print_stats: whether to print statistics
    Returns:
        filtered_mean: mean after removing outliers
    """
    # Filter outliers
    filtered_metrics, mask = filter_by_quartiles(metrics)

    if print_stats:
        print(f"Removed {np.sum(~mask)} outliers")
        print(f"Original mean: {metrics.mean():.4f}")
        print(f"Filtered mean: {filtered_metrics.mean():.4f}")

    return np.mean(filtered_metrics).round(4)


# More detailed example with statistics:
def analyze_metrics_distribution(metric_values, metric_name):
    """
    Analyze metric distribution and filter outliers
    """
    # Original statistics
    print(f"\nAnalyzing {metric_name}:")
    print("Original Statistics:")
    print(f"Mean: {metric_values.mean():.4f}")
    print(f"Std: {metric_values.std():.4f}")
    print(f"Min: {metric_values.min():.4f}")
    print(f"Max: {metric_values.max():.4f}")

    # Calculate quartiles
    Q1 = np.percentile(metric_values, 25)
    Q2 = np.percentile(metric_values, 50)  # median
    Q3 = np.percentile(metric_values, 75)
    IQR = Q3 - Q1

    print("\nQuartile Analysis:")
    print(f"Q1 (25th percentile): {Q1:.4f}")
    print(f"Q2 (median): {Q2:.4f}")
    print(f"Q3 (75th percentile): {Q3:.4f}")
    print(f"IQR: {IQR:.4f}")

    # Filter with different thresholds
    for multiplier in [1.5, 2.0, 3.0]:
        filtered_data, mask = filter_by_quartiles(metric_values, multiplier)
        print(f"\nFiltering with {multiplier}×IQR:")
        print(f"Removed {np.sum(~mask)} values")
        print(f"Original mean: {metric_values.mean():.4f}")
        print(f"Filtered mean: {filtered_data.mean():.4f}")

    # Return filtered data with standard 1.5×IQR
    return filter_by_quartiles(metric_values, 1.5)[0].mean()
