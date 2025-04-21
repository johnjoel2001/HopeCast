import pandas as pd
import os  # Added for folder creation
if not hasattr(pd.Series, 'is_monotonic'):
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from pycox.evaluation import EvalSurv

class SurvivalModel:
    def __init__(self, file_path, model_name):
        self.file_path = file_path
        self.model_name = model_name
        self.model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)

    def load_data(self):
        df = pd.read_csv(self.file_path)
        df = df.dropna(subset=['overall_survival_months', 'overall_survival'])
        df = df[df['overall_survival_months'] > 0]
        df = df.drop(columns=['patient_id'], errors='ignore')

        X = df.drop(columns=['overall_survival_months', 'overall_survival'])
        y = df[['overall_survival_months', 'overall_survival']]

        # One-hot encode categorical columns for Clinical only
        if self.model_name == 'Clinical':
            categorical_cols = ['er_status', 'her2_status', 'pam50_+_claudin-low_subtype', 'pr_status', 'type_of_breast_surgery', 'cellularity']
            available_cols = [col for col in categorical_cols if col in X.columns]
            if available_cols:
                X = pd.get_dummies(X, columns=available_cols, dummy_na=True)
        
        X = X.select_dtypes(include=[np.number])
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = self.load_data()
        self.model.fit(X_train, y_train['overall_survival_months'])
        y_pred = self.model.predict(X_test)

        # Save the trained model to models/non_deep_learning folder
        model_dir = "models/non_deep_learning"
        os.makedirs(model_dir, exist_ok=True)  # Create the folder if it doesn't exist
        model_path = os.path.join(model_dir, f"{self.model_name.lower()}_xgb.json")
        self.model.save_model(model_path)
        print(f"💾 Saved {self.model_name} model to {model_path}")

        y_pred = np.maximum(y_pred, 1e-10)
        times = np.array([1.0])
        surv = pd.DataFrame(np.exp(-1/y_pred[:, None]), columns=times, index=y_test.index)

        ev = EvalSurv(surv.T, y_test['overall_survival_months'].values, y_test['overall_survival'].values, censor_surv='km')
        c_index = ev.concordance_td('antolini')
        print(f"{'📊' if self.model_name == 'Clinical' else '🧬' if self.model_name == 'Genetic' else '🧪'} {self.model_name} C-index: {c_index:.4f}")

def run_all_models():
    datasets = [
        ('Clinical', 'data/clinical_data.csv'),
        ('Genetic', 'data/genetic_data.csv'),
        ('Mutation', 'data/mutation_data.csv')
    ]
    for name, path in datasets:
        model = SurvivalModel(path, name)
        model.train_and_evaluate()

if __name__ == "__main__":
    run_all_models()