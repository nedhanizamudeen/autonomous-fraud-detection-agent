from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score


class ModelEvaluationAgent:

    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}

    def evaluate_model(self, model, model_name, scaler):

        print(f"\nEvaluating {model_name}...")

        # Apply scaling if needed
        if scaler is not None:
            X_test = scaler.transform(self.X_test)
        else:
            X_test = self.X_test

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_prob)
        f1 = f1_score(self.y_test, y_pred)

        # Store results
        self.results[model_name] = {
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "ROC-AUC": roc_auc,
            "F1": f1
        }

        # Print results
        print(f"{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")

    def run(self):

        print("\n--- Model Evaluation Started ---")

        for model_name, (model, scaler) in self.models.items():
            self.evaluate_model(model, model_name, scaler)

        print("\n--- Evaluation Completed ---")

        return self.results