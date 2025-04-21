import pandas as pd
if not hasattr(pd.Series, 'is_monotonic'):
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)

import numpy as np
import torch
import torchtuples as tt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import os
import pickle

# Ensure the model folder exists
MODEL_DIR = "models/deep_learning"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class DeepSurvRunner:
    def __init__(self, dataset_path, label_duration='overall_survival_months', label_event='overall_survival'):
        self.dataset_path = dataset_path
        self.label_duration = label_duration
        self.label_event = label_event
        self.scaler = StandardScaler()
        # Initialize attributes to None for safety
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.dataset_path)
        df = df.dropna(subset=[self.label_duration, self.label_event])
        df = df[df[self.label_duration] > 0]
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if self.label_duration in numerical_cols:
            numerical_cols.remove(self.label_duration)
        if self.label_event in numerical_cols:
            numerical_cols.remove(self.label_event)
        
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        for col in categorical_cols:
            df[col] = df[col].fillna('missing')
        
        if categorical_cols:
            encode_cols = [col for col in categorical_cols if col != 'patient_id']
            df = pd.get_dummies(df, columns=encode_cols, prefix=encode_cols, dummy_na=True)
        
        if 'patient_id' in df.columns:
            df = df.drop('patient_id', axis=1)
        
        df = df.select_dtypes(include=[np.number])
        
        if df.shape[1] < 3 or df.shape[0] < 10 or self.label_duration not in df.columns or self.label_event not in df.columns or not df[self.label_event].isin([0, 1]).all():
            print(f"⚠️ {self.dataset_path}: Dataset does not meet requirements (too few features/rows, missing labels, or invalid event values).")
            return None
        return df

    def prepare_data(self):
        df = self.load_data()
        if df is None:
            return False
        X = df.drop([self.label_duration, self.label_event], axis=1)
        y = df[[self.label_duration, self.label_event]]
        if X.shape[1] == 0:
            print(f"⚠️ {self.dataset_path}: No features available after preprocessing.")
            return False
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_test = self.scaler.transform(X_test).astype(np.float32)
        self.x_train, self.x_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.feature_names = X.columns.tolist()
        return True

    def build_model(self):
        if self.x_train is None:
            print(f"⚠️ {self.dataset_path}: Cannot build model, training data not prepared.")
            return False
        net = torch.nn.Sequential(
            torch.nn.Linear(self.x_train.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.model = CoxPH(net, tt.optim.Adam)
        return True

    def train(self):
        if self.x_train is None or self.y_train is None:
            print(f"⚠️ {self.dataset_path}: Cannot train model, training data not prepared.")
            return False
        train_data = (self.x_train, (self.y_train[self.label_duration].values, self.y_train[self.label_event].values))
        self.model.fit(*train_data, batch_size=64, epochs=50, verbose=False)
        self.model.compute_baseline_hazards()
        return True

    def save_model(self):
        if self.model is None:
            print(f"⚠️ {self.dataset_path}: Cannot save model, model not trained.")
            return False
        model_name = self.dataset_path.split('/')[1].split('_')[0].capitalize()
        try:
            # Save the CoxPH model's state dictionary
            model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pth")
            torch.save(self.model.net.state_dict(), model_path)
            print(f"✅ {model_name}: Model saved to {model_path}")

            # Save the scaler
            scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ {model_name}: Scaler saved to {scaler_path}")

            # Save the feature names
            features_path = os.path.join(MODEL_DIR, f"{model_name}_features.pkl")
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_names, f)
            print(f"✅ {model_name}: Feature names saved to {features_path}")
            return True
        except Exception as e:
            print(f"⚠️ {model_name}: Error saving model: {str(e)}")
            return False

    def evaluate(self):
        if self.x_test is None or self.y_test is None:
            print(f"⚠️ {self.dataset_path}: Cannot evaluate model, test data not prepared.")
            return False
        surv = self.model.predict_surv_df(self.x_test)
        surv = surv.sort_index()
        surv = surv.reindex(sorted(surv.index), axis=0)
        durations_test = self.y_test[self.label_duration].values
        events_test = self.y_test[self.label_event].values
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        c_index = ev.concordance_td('antolini')
        model_name = self.dataset_path.split('/')[1].split('_')[0].capitalize()
        print(f"{model_name}: {c_index:.4f}")
        return True

def run_all_models():
    for name, path in {'Clinical': 'data/clinical_data.csv', 'Genetic': 'data/genetic_data.csv', 'Mutation': 'data/mutation_data.csv'}.items():
        print(f"\n=== Processing {name} Model ===")
        runner = DeepSurvRunner(dataset_path=path)
        if runner.prepare_data() and runner.build_model() and runner.train():
            runner.save_model()  # Save the model after training
            runner.evaluate()
        else:
            print(f"⚠️ {name}: Failed to train model.")

if __name__ == "__main__":
    run_all_models()