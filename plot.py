import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_kmeans_results(dataset_file, labels_file):
    # Load the dataset and labels
    dataset = pd.read_csv(dataset_file, header=None, names=['x', 'y'])
    labels = pd.read_csv(labels_file, header=None, names=['label'])

    # Ensure that the number of points and labels match
    num_points = len(labels)
    print(num_points)

    # Use only the first 'num_points' from the dataset
    dataset = dataset.iloc[:num_points]

    # Ensure labels are in a numpy array
    labels = labels.to_numpy().flatten()

    # Plot the points with different colors based on the labels
    plt.figure(figsize=(8, 6))

    # Get the unique labels (clusters)
    unique_labels = np.unique(labels)

    # Plot each cluster with a different color
    for label in unique_labels:
        cluster_points = dataset[labels == label]
        plt.scatter(cluster_points['x'], cluster_points['y'], label=f'Cluster {label}', s=10)

    # Add title and labels
    plt.title('K-Means Clustering Results')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Show legend
    # plt.legend()

    # Show the plot
    plt.show()

# Call the function with appropriate file paths
plot_kmeans_results('data/dataset.csv', 'data/labels.csv')
