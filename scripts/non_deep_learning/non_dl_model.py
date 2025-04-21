# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# import joblib

# class SurvivalMonthsMLModel:
#     def __init__(self, file_path='data/gene_expression.csv'):
#         self.file_path = file_path
#         self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
#         self.df = None
#         self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

#     def load_and_prepare_data(self):
#         print("📥 Loading dataset...")
#         self.df = pd.read_csv(self.file_path)
#         print(f"✅ Loaded data shape: {self.df.shape}")

#         # Drop missing target
#         self.df = self.df.dropna(subset=['overall_survival_months'])

#         # Drop unnecessary or non-numeric fields
#         drop_cols = ['patient_id', 'type_of_breast_surgery', 'cancer_type', 'overall_survival']
#         self.df = self.df.drop(columns=[col for col in drop_cols if col in self.df.columns], errors='ignore')

#         X = self.df.drop(columns=['overall_survival_months'])
#         y = self.df['overall_survival_months']
#         X = X.select_dtypes(include=[np.number])  # keep numeric features only

#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#     def train_model(self):
#         print("🧠 Training Random Forest model...")
#         self.model.fit(self.X_train, self.y_train)

#     def evaluate_model(self):
#         y_pred = self.model.predict(self.X_test)
#         mae = mean_absolute_error(self.y_test, y_pred)
#         r2 = r2_score(self.y_test, y_pred)
#         print(f"📊 MAE: {mae:.2f}")
#         print(f"📈 R² Score: {r2:.2f}")
#         return mae, r2

#     def save_model(self, path="models/survival_rf_model.pkl"):
#         os.makedirs("models", exist_ok=True)
#         joblib.dump(self.model, path)
#         print(f"💾 Model saved to {path}")

#     def run(self):
#         self.load_and_prepare_data()
#         self.train_model()
#         return self.evaluate_model()

# if __name__ == "__main__":
#     ml_model = SurvivalMonthsMLModel()
#     ml_model.run()


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# from xgboost import XGBRegressor

# class SurvivalPredictionModel:
#     def __init__(self, file_path="data/gene_expression.csv"):
#         self.file_path = file_path
#         self.model = XGBRegressor(
#             n_estimators=300,
#             learning_rate=0.05,
#             max_depth=6,
#             random_state=42,
#             n_jobs=-1
#         )

#     def load_and_preprocess(self):
#         print("🔹 Loading and cleaning data...")
#         df = pd.read_csv(self.file_path)
#         df = df.dropna(subset=["overall_survival_months"])

#         # Drop unused or identifier columns
#         drop_cols = ["patient_id", "type_of_breast_surgery", "cancer_type", "overall_survival"]
#         df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

#         # Features and Target
#         X = df.drop(columns=["overall_survival_months"]).select_dtypes(include=[np.number])
#         y = df["overall_survival_months"]

#         return train_test_split(X, y, test_size=0.2, random_state=42)

#     def train_and_evaluate(self):
#         X_train, X_test, y_train, y_test = self.load_and_preprocess()
#         print("🚀 Training model...")
#         self.model.fit(X_train, y_train)

#         print("📈 Evaluating model...")
#         y_pred = self.model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         print(f"📊 MAE on overall survival months: {mae:.2f}")

#         return mae

# if __name__ == "__main__":
#     model = SurvivalPredictionModel()
#     model.train_and_evaluate()



# import pandas as pd
# if not hasattr(pd.Series, 'is_monotonic'):
#     pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)

# import numpy as np
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
# from pycox.evaluation import EvalSurv

# class SurvivalModel:
#     def __init__(self, file_path, model_name):
#         self.file_path = file_path
#         self.model_name = model_name
#         self.model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)

#     def load_data(self):
#         df = pd.read_csv(self.file_path)
#         df = df.dropna(subset=['overall_survival_months', 'overall_survival'])
#         df = df[df['overall_survival_months'] > 0]
#         df = df.drop(columns=['patient_id'], errors='ignore')

#         X = df.drop(columns=['overall_survival_months', 'overall_survival'])
#         y = df[['overall_survival_months', 'overall_survival']]

#         # One-hot encode categorical columns for Clinical only
#         if self.model_name == 'Clinical':
#             categorical_cols = ['er_status', 'her2_status', 'pam50_+_claudin-low_subtype', 'pr_status', 'type_of_breast_surgery', 'cellularity']
#             categorical_cols = [col for col in categorical_cols if col in X.columns]
#             if categorical_cols:
#                 X = pd.get_dummies(X, columns=categorical_cols, dummy_na=True)
        
#         X = X.select_dtypes(include=[np.number])
#         for col in X.columns:
#             X[col] = X[col].fillna(X[col].median())

#         print(f"{self.model_name} features: {X.shape[1]}")  # Debug feature count
#         return train_test_split(X, y, test_size=0.2, random_state=42)

#     def train_and_evaluate(self):
#         X_train, X_test, y_train, y_test = self.load_data()
#         self.model.fit(X_train, y_train['overall_survival_months'])
#         y_pred = self.model.predict(X_test)

#         y_pred = np.maximum(y_pred, 1e-10)
#         print(f"{self.model_name} sample predictions: {y_pred[:5]}")
#         print(f"{self.model_name} sample actual: {y_test['overall_survival_months'].values[:5]}")

#         times = np.array([1.0])
#         surv = pd.DataFrame(np.exp(-1/y_pred[:, None]), columns=times, index=y_test.index)

#         ev = EvalSurv(surv.T, y_test['overall_survival_months'].values, y_test['overall_survival'].values, censor_surv='km')
#         c_index = ev.concordance_td('antolini')
#         print(f"{'📊' if self.model_name == 'Clinical' else '🧬' if self.model_name == 'Genetic' else '🧪'} {self.model_name} C-index: {c_index:.4f}")

# def run_all_models():
#     datasets = [
#         ('Clinical', 'data/clinical_data.csv'),
#         ('Genetic', 'data/genetic_data.csv'),
#         ('Mutation', 'data/mutation_data.csv')
#     ]
#     for name, path in datasets:
#         model = SurvivalModel(path, name)
#         model.train_and_evaluate()

# if __name__ == "__main__":
#     run_all_models()

import pandas as pd
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