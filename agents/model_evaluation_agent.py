from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, confusion_matrix


class ModelEvaluationAgent:

    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}

    def evaluate_model(self, model, model_name, scaler=None):

        print(f"\nEvaluating {model_name}...")

        # Apply scaling only if needed (for Logistic Regression)
        if scaler is not None:
            X_test = scaler.transform(self.X_test)
        else:
            X_test = self.X_test

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(self.y_test, y_pred)

        print("\nConfusion Matrix:")
        print(cm)

        tn, fp, fn, tp = cm.ravel()

        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives (TP): {tp}")

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
            "F1 Score": f1
        }

        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")

    def run(self):

        print("\n--- Model Evaluation Started ---")

        for model_name, model_data in self.models.items():

            # Handle models with scaler (Logistic Regression)
            if isinstance(model_data, tuple):
                model, scaler = model_data
            else:
                model = model_data
                scaler = None

            self.evaluate_model(model, model_name, scaler)

        print("\n--- Evaluation Completed ---")

        return self.results