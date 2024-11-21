import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Read data from CSV file
df = pd.read_csv('data/execution_times_SoA.csv')

# Create directory for saving plots
plot_dir = 'data/plots'
os.makedirs(plot_dir, exist_ok=True)

# Function to calculate speedup
def calculate_speedup(data):
    # Filter out the sequential data (where numThreads == 0)
    single_thread_data = data[data['numThreads'] == 0][['numPoints', 'numClusters', 'maxIterations', 'executionTime']]
    single_thread_data = single_thread_data.rename(columns={'executionTime': 'single_thread_time'})
    
    # Merge the sequential data with the parallel data based on numPoints, numClusters, and maxIterations
    merged_data = pd.merge(data, single_thread_data, on=['numPoints', 'numClusters', 'maxIterations'], how='left')
    
    # Calculate speedup: single_thread_time / execution_time (for parallel runs)
    merged_data['speedup'] = merged_data['single_thread_time'] / merged_data['executionTime']
    
    return merged_data

# Calculate speedup
data_with_speedup = calculate_speedup(df)

# Function to plot speedup for each numPoints
def plot_speedup_by_numPoints(data):
    # Group by numThreads, numPoints, numClusters, and maxIterations
    grouped = data.groupby(['numThreads', 'numPoints', 'numClusters', 'maxIterations'])['speedup'].mean().reset_index()
    
    # Generate a color palette for unique numClusters
    unique_clusters = grouped['numClusters'].unique()
    color_palette = sns.color_palette("tab10", len(unique_clusters))  # Using seaborn's color palette
    cluster_colors = {cluster: color_palette[i] for i, cluster in enumerate(unique_clusters)}

    # Define marker styles for each maxIterations
    unique_max_iterations = grouped['maxIterations'].unique()
    markers = ['o', 's', '^', 'D', 'p', '*', 'H', 'X', '+', 'v']  # Example markers
    max_iterations_markers = {maxIter: markers[i % len(markers)] for i, maxIter in enumerate(unique_max_iterations)}
    
    # Iterate over different numPoints and create a separate plot for each
    for numPoints in grouped['numPoints'].unique():
        plt.figure(figsize=(10, 6))

        # Filter data for the current numPoints
        plot_data = grouped[grouped['numPoints'] == numPoints]

        # Plot the speedup curves for different configurations
        for (numClusters, maxIterations), group in plot_data.groupby(['numClusters', 'maxIterations']):
            color = cluster_colors[numClusters]  # Get color for this numClusters
            marker = max_iterations_markers[maxIterations]  # Get marker for this maxIterations
            plt.plot(group['numThreads'], group['speedup'], label=f'Clusters: {numClusters}, Iterations: {maxIterations}', color=color, marker=marker, markersize=8)
        
        # Set plot labels and title
        plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
        plt.ylabel('Speedup', fontsize=12, fontweight='bold')
        plt.title(f'K-Means Speedup vs Number of Threads (Points: {numPoints})', fontsize=14, fontweight='bold')

        # Make the legend fancier
        plt.legend(
            title='Configurations', 
            bbox_to_anchor=(1, 0), 
            loc='lower right', 
            fontsize=10, 
            title_fontsize=12,
            shadow=True,  # Add shadow to the legend
            fancybox=True,  # Use a fancy box for the legend
            framealpha=0.8,  # Transparency of the legend box
            borderpad=1,  # Padding between legend entries and the box
            borderaxespad=1  # Padding between legend and axes
        )
        
        # Add a grid with a lighter style
        plt.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the plot for this numPoints
        plot_filename = os.path.join(plot_dir, f'kmeans_speedup_{numPoints}_points.png')
        plt.savefig(plot_filename, dpi=300)  # Save with high resolution
        plt.show()

# Call the function to plot speedup for different numPoints
plot_speedup_by_numPoints(data_with_speedup)
