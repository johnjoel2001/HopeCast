import pandas as pd
from lifelines.utils import concordance_index
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split


class NaiveSurvivalEvaluator:
    """
    Class to evaluate a naive baseline model using concordance index
    
    """

    def __init__(self, name, file_path):
        self.name = name
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.e_train = None
        self.e_test = None

    def load_and_prepare_data(self):

        """
        Loading dataset and prepares train/test split

        """
        df = pd.read_csv(self.file_path)
        X = df.drop(columns=["patient_id", "overall_survival", "overall_survival_months"], errors='ignore')
        y = df["overall_survival_months"]
        events = df["overall_survival"]
        self.X_train, self.X_test, self.y_train, self.y_test, self.e_train, self.e_test = train_test_split(
            X, y, events, test_size=0.2, random_state=42
        )

    def evaluate_model(self):

        """
        Training a naive model and computes the C-index

        """
        model = DummyRegressor(strategy="mean")
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        c_index = concordance_index(self.y_test, -y_pred, self.e_test)
        print(f"{self.name} C-index (naive): {c_index:.4f}")


def main():

    """
    Running naive model evaluation for all datasets using NaiveSurvivalEvaluator class
    
    """

    datasets = {
        "Clinical": "data/clinical_data.csv",
        "Genetic": "data/genetic_data.csv",
        "Mutation": "data/mutation_data.csv"
    }

    for name, path in datasets.items():
        evaluator = NaiveSurvivalEvaluator(name, path)
        evaluator.load_and_prepare_data()
        evaluator.evaluate_model()


if __name__ == "__main__":
    main()