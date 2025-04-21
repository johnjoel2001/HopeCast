import pandas as pd
from lifelines.utils import concordance_index
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split


def evaluate_naive_c_index(name, file_path):
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    X = df.drop(columns=["patient_id", "overall_survival", "overall_survival_months"])
    y = df["overall_survival_months"]
    events = df["overall_survival"]

    # Train/test split
    X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
        X, y, events, test_size=0.2, random_state=42
    )

    # Naive model: predict mean survival months
    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate concordance index
    c_index = concordance_index(y_test, -y_pred, e_test)  # negative for correct ranking
    print(f"✅ {name} C-index (naive): {c_index:.4f}")


def main():
    datasets = {
        "Clinical": "data/clinical_data.csv",
        "Genetic": "data/genetic_data.csv",
        "Mutation": "data/mutation_data.csv"
    }

    for name, path in datasets.items():
        evaluate_naive_c_index(name, path)


if __name__ == "__main__":
    main()
