import os

from src.data_loader import load_data
from src.preprocessing import (
    handle_missing_values,
    select_features,
    FeatureScaler
)
from src.clustering_model import KMeansModel
from src.visualization import plot_elbow_method, plot_clusters


def main():

    print("\n--- Customer Segmentation Pipeline Started ---\n")

    # 1. Load data
    df = load_data()
    print(f"Data Shape: {df.shape}")
    print(df.head(), "\n")

    # 2. Preprocessing
    df = handle_missing_values(df)

    feature_df = select_features(df)

    scaler = FeatureScaler()
    scaled_data = scaler.fit_transform(feature_df)

    print("Preprocessing completed.\n")

    # 3. Elbow Method
    k_range = (2, 10)

    model = KMeansModel()

    inertia_values = model.compute_inertia_values(scaled_data, k_range)

    plot_elbow_method(k_range, inertia_values)

    print("Elbow method completed.\n")

    # 4. Train final model
    k = 4  # chosen after elbow analysis
    labels = model.train(scaled_data, k)

    df["Cluster"] = labels

    print(f"Model trained with k={k}\n")

    # 5. Visualization
    plot_clusters(scaled_data, labels)

    # 6. Save results
    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/clustered_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")
    print("\n--- Pipeline Completed ---\n")


if __name__ == "__main__":
    main()