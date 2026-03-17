from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class ModelTrainingAgent:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.models = {}

    def run(self):

        print("\n--- Model Training Started ---")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)

        print("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        lr.fit(X_train_scaled, self.y_train)
        self.models["Logistic Regression"] = (lr, scaler)

        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(self.X_train, self.y_train)
        self.models["Random Forest"] = (rf, None)

        print("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(self.X_train, self.y_train)
        self.models["Gradient Boosting"] = (gb, None)

        print("\nAll models trained successfully.")
        return self.models