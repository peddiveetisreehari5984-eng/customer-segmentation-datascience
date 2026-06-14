import matplotlib.pyplot as plt


def plot_elbow_method(k_range, inertia_values):
    """
    Elbow curve visualization.
    """

    start, end = k_range
    ks = list(range(start, end + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertia_values, marker='o', linewidth=2)

    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_clusters(data, labels):
    """
    Cluster visualization (2D projection).
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        data[:, 0],
        data[:, 1],
        c=labels,
        cmap="viridis",
        s=50,
        alpha=0.8
    )

    plt.title("Customer Segmentation Clusters")
    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()