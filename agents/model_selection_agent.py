class ModelSelectionAgent:

    def __init__(self, results):
        self.results = results

    def run(self):

        print("\n--- Model Selection Started ---")

        best_model = None
        best_score = (-1, -1)  # (F1, ROC-AUC)

        for model_name, metrics in self.results.items():

            f1 = metrics["F1 Score"]
            roc_auc = metrics["ROC-AUC"]

            score = (f1, roc_auc)

            if score > best_score:
                best_score = score
                best_model = model_name

        print(f"\nSelected Best Model: {best_model}")
        print("Reason: Best balance of Precision and Recall (F1 Score)")

        return best_model