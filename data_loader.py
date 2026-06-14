from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np


def load_data(n_samples=800, random_state=42):
    """
    Generates synthetic customer dataset.
    Simulates real-world clustering data with noise.
    """

    X, _ = make_blobs(
        n_samples=n_samples,
        centers=4,
        cluster_std=[1.0, 1.5, 1.2, 2.0],
        n_features=2,
        random_state=random_state
    )

    df = pd.DataFrame(X, columns=["Annual Income", "Spending Score"])

    # Adding realistic noise features
    np.random.seed(random_state)
    df["Age"] = np.random.randint(18, 70, n_samples)
    df["Savings Score"] = np.random.uniform(0, 100, n_samples)

    # Adding derived feature
    df["Income_Spending_Ratio"] = df["Annual Income"] / (df["Spending Score"] + 1e-5)

    return df