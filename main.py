import warnings
warnings.filterwarnings("ignore")

from agents.data_understanding_agent import DataUnderstandingAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.model_training_agent import ModelTrainingAgent
from agents.model_evaluation_agent import ModelEvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent


if __name__ == "__main__":

    print("\n=== Autonomous Fraud Detection Pipeline Started ===\n")

    data_agent = DataUnderstandingAgent(
        transaction_path="data/raw/train_transaction.csv",
        identity_path="data/raw/train_identity.csv"
    )

    merged_data, primary_metric = data_agent.run()

    preprocessing_agent = PreprocessingAgent(merged_data)
    X_train, X_test, y_train, y_test = preprocessing_agent.clean_data()

    training_agent = ModelTrainingAgent(X_train, y_train)
    models = training_agent.run()

    evaluation_agent = ModelEvaluationAgent(models, X_test, y_test)
    results = evaluation_agent.run()

    selection_agent = ModelSelectionAgent(results)
    best_model = selection_agent.run()

    print("\n=== Pipeline Completed Successfully ===\n")