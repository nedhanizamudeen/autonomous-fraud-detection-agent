import pandas as pd


class DataUnderstandingAgent:

    def __init__(self, transaction_path, identity_path):
        self.transaction_path = transaction_path
        self.identity_path = identity_path

    def load_data(self):
        print("Loading datasets...\n")

        self.transactions = pd.read_csv(self.transaction_path)
        self.identity = pd.read_csv(self.identity_path)

        print("Transaction Dataset Shape:", self.transactions.shape)
        print("Identity Dataset Shape:", self.identity.shape)

    def merge_data(self):
        print("\nMerging transaction and identity datasets...")

        self.data = self.transactions.merge(
            self.identity,
            on="TransactionID",
            how="left"
        )

        print("Merged Dataset Shape:", self.data.shape)
        return self.data

    def analyze_missing_values(self):
        print("\nAnalyzing missing values...")

        missing_percent = (self.data.isnull().sum() / len(self.data)) * 100
        high_missing = missing_percent[missing_percent > 50]

        print("\nColumns with >50% missing values:")
        print(high_missing.sort_values(ascending=False))

    def analyze_class_imbalance(self):
        print("\nAnalyzing class distribution...")

        fraud_counts = self.data['isFraud'].value_counts()
        print(fraud_counts)

        fraud_ratio = fraud_counts[1] / fraud_counts.sum()
        print("\nFraud Ratio:", round(fraud_ratio, 4))

        if fraud_ratio < 0.1:
            print("\nDecision: Dataset is highly imbalanced.")
            print("Primary Metric: Recall")
            print("Secondary Metric: ROC-AUC")
            self.primary_metric = "Recall"
        else:
            print("\nDataset is relatively balanced.")
            self.primary_metric = "ROC-AUC"

    def run(self):
        self.load_data()
        merged_data = self.merge_data()
        self.analyze_missing_values()
        self.analyze_class_imbalance()
        return merged_data, self.primary_metric