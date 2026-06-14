from sklearn.preprocessing import StandardScaler


def handle_missing_values(df):
    """
    Basic missing value handling.
    """
    return df.dropna()


def select_features(df):
    """
    Selects features for clustering.
    """
    features = [
        "Annual Income",
        "Spending Score",
        "Age",
        "Savings Score",
        "Income_Spending_Ratio"
    ]

    return df[features]


class FeatureScaler:
    """
    Encapsulated scaler for reuse
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def transform(self, data):
        return self.scaler.transform(data)