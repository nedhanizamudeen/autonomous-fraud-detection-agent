import pandas as pd
from sklearn.model_selection import train_test_split


class PreprocessingAgent:

    def __init__(self, data, sample_size=100000):
        if len(data) > sample_size:
            print(f"\nSampling {sample_size} rows for processing...")
            self.data = data.sample(sample_size, random_state=42)
        else:
            self.data = data

    def clean_data(self):
        print("\nStarting preprocessing...")

        if "TransactionID" in self.data.columns:
            self.data = self.data.drop(columns=["TransactionID"])

        X = self.data.drop(columns=["isFraud"])
        y = self.data["isFraud"]

        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = X.select_dtypes(include=["object"]).columns

        print("Numeric columns:", len(numeric_cols))
        print("Categorical columns:", len(categorical_cols))

        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        X[categorical_cols] = X[categorical_cols].fillna("Missing")

        high_cardinality_cols = [
            col for col in categorical_cols if X[col].nunique() > 50
        ]

        if high_cardinality_cols:
            print("\nDropping high-cardinality columns:")
            print(high_cardinality_cols)
            X = X.drop(columns=high_cardinality_cols)
            categorical_cols = X.select_dtypes(include=["object"]).columns

        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        print("Shape after encoding:", X.shape)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training set shape:", self.X_train.shape)
        print("Test set shape:", self.X_test.shape)

        return self.X_train, self.X_test, self.y_train, self.y_test