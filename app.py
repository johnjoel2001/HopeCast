import os
import pickle
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from sklearn.utils import resample
from scripts.rag.rag import RAGSystem  # Import RAGSystem from scripts/rag/rag.py
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_label(column: str) -> str:
    """
    Format column name by replacing underscores with spaces and capitalizing words.

    Args:
        column (str): Original column name (e.g., 'type_of_breast_surgery').

    Returns:
        str: Formatted label (e.g., 'Type of Breast Surgery').
    """
    return ' '.join(word.capitalize() for word in column.replace('_', ' ').split())

class DatasetLoader:
    """Handles loading and validation of clinical, genetic, and mutation datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DatasetLoader with directory path.

        Args:
            data_dir (str): Directory containing dataset CSV files.
        """
        self.data_dir = data_dir
        self.target_cols = ['patient_id', 'overall_survival', 'overall_survival_months']
        self.datasets = self._load_datasets()

    def _load_datasets(self) -> dict:
        """
        Load datasets from CSV files.

        Returns:
            dict: Dictionary of dataset names to DataFrames.

        Raises:
            FileNotFoundError: If a dataset file is missing.
        """
        try:
            datasets = {
                'Clinical': pd.read_csv(os.path.join(self.data_dir, 'clinical_data.csv')),
                'Genetic': pd.read_csv(os.path.join(self.data_dir, 'genetic_data.csv')),
                'Mutation': pd.read_csv(os.path.join(self.data_dir, 'mutation_data.csv'))
            }
            for name, df in datasets.items():
                logger.info(f"{name} dataset shape: {df.shape}")
                if 'tumor_stage' in df.columns:
                    logger.info(f"{name} tumor stage distribution: {df['tumor_stage'].value_counts().to_dict()}")
                if 'tumor_size' in df.columns:
                    logger.info(f"{name} tumor size stats: min={df['tumor_size'].min()}, max={df['tumor_size'].max()}")
            return datasets
        except FileNotFoundError as e:
            logger.error(f"Dataset not found: {str(e)}")
            st.error(f"Dataset not found: {str(e)}")
            raise

    def get_dataset(self, dataset_type: str) -> pd.DataFrame:
        """
        Retrieve a specific dataset.

        Args:
            dataset_type (str): Name of the dataset ('Clinical', 'Genetic', 'Mutation').

        Returns:
            pd.DataFrame: Requested dataset.
        """
        return self.datasets[dataset_type]

    def get_feature_cols(self, dataset_type: str) -> list:
        """
        Get non-target columns for a dataset.

        Args:
            dataset_type (str): Name of the dataset.

        Returns:
            list: List of feature column names.
        """
        return [col for col in self.datasets[dataset_type].columns if col not in self.target_cols]

class SurvivalModel:
    """Manages CoxPH model loading, preprocessing, and survival prediction."""
    
    def __init__(self, model_dir: str = "models/deep_learning"):
        """
        Initialize SurvivalModel with model directory.

        Args:
            model_dir (str): Directory containing model files.
        """
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            st.error(f"Model directory '{model_dir}' not found. Please ensure model files are present.")
            st.markdown("### Instructions\n1. Ensure model files are in the 'models/deep_learning' directory.\n2. Check file permissions.")
            raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    def _load_model(self, dataset_type: str) -> tuple:
        """
        Load pre-trained CoxPH model, scaler, and feature names.

        Args:
            dataset_type (str): Dataset type ('Clinical', 'Genetic', 'Mutation').

        Returns:
            tuple: (model, scaler, feature_names).

        Raises:
            Exception: If model files are missing or invalid.
        """
        logger.info(f"Loading model for {dataset_type}...")
        try:
            # Load feature names
            with open(os.path.join(self.model_dir, f"{dataset_type}_features.pkl"), 'rb') as f:
                feature_names = pickle.load(f)
            logger.info(f"Loaded feature names: {feature_names}")

            # Load scaler
            with open(os.path.join(self.model_dir, f"{dataset_type}_scaler.pkl"), 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Scaler loaded successfully.")

            # Import PyTorch-related modules here to avoid Streamlit file watcher issues
            import torch
            import torchtuples as tt
            from pycox.models import CoxPH

            # Initialize neural network
            net = torch.nn.Sequential(
                torch.nn.Linear(len(feature_names), 64),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(64),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )
            # Set learning rate in optimizer to avoid 'lr' error
            optimizer = tt.optim.Adam(lr=0.001)
            model = CoxPH(net, optimizer)

            # Load pre-trained weights
            model.net.load_state_dict(torch.load(os.path.join(self.model_dir, f"{dataset_type}_model.pth")))
            model.net.eval()
            logger.info(f"Model loaded from {dataset_type}_model.pth")

            return model, scaler, feature_names
        except Exception as e:
            logger.error(f"Error loading model for {dataset_type}: {str(e)}")
            raise

    def _preprocess_data(self, df: pd.DataFrame, user_input: dict, feature_names: list) -> tuple:
        """
        Preprocess dataset and user input for prediction.

        Args:
            df (pd.DataFrame): Input dataset.
            user_input (dict): User-provided feature values.
            feature_names (list): Expected feature names from model.

        Returns:
            tuple: (df_encoded, input_df, y_duration, y_event).

        Raises:
            ValueError: If preprocessing results in empty or NaN-filled data.
        """
        # Import here to avoid file watcher issues
        import numpy as np

        # Handle data imbalance by upsampling severe cases
        if 'tumor_stage' in df.columns:
            high_risk = df[df['tumor_stage'].isin(['3', '4'])]
            low_risk = df[~df['tumor_stage'].isin(['3', '4'])]
            if not high_risk.empty and not low_risk.empty:
                high_risk_upsampled = resample(high_risk, replace=True, n_samples=len(low_risk), random_state=42)
                df = pd.concat([low_risk, high_risk_upsampled])
                logger.info(f"Upsampled high-risk cases. New dataset shape: {df.shape}")

        # Clean dataset
        df = df.dropna(subset=['overall_survival_months', 'overall_survival'])
        df = df[df['overall_survival_months'] > 0]
        logger.info(f"Dataset for baseline hazards: {df.shape}")
        if 'tumor_stage' in df.columns:
            logger.info(f"Tumor stage distribution: {df['tumor_stage'].value_counts()}")
        if 'tumor_size' in df.columns:
            logger.info(f"Tumor size stats: min={df['tumor_size'].min()}, max={df['tumor_size'].max()}, mean={df['tumor_size'].mean()}")

        # Extract features
        target_cols = ['patient_id', 'overall_survival', 'overall_survival_months']
        df_features = df.drop(columns=target_cols, errors='ignore')
        numerical_cols = df_features.select_dtypes(include=[np.number]).columns
        categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()

        # Fill missing numerical values with median
        for col in numerical_cols:
            df_features[col] = df_features[col].fillna(df_features[col].median())
        logger.info("Filled NaN values in numerical columns with median.")

        # Encode categorical variables
        df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=False, dummy_na=True)
        logger.info(f"Features after encoding: {df_encoded.columns.tolist()}")

        if df_encoded.empty:
            raise ValueError("No features available after preprocessing.")

        # Align encoded dataset with model feature names
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_names]
        if df_encoded.isna().any().any():
            raise ValueError(f"Encoded dataset contains NaN: {df_encoded.columns[df_encoded.isna().any()].tolist()}")

        # Process user input
        input_df = pd.DataFrame([user_input])

        # Map chemotherapy to binary
        if 'chemotherapy' in input_df.columns:
            input_df['chemotherapy'] = input_df['chemotherapy'].map({'yes': 1, 'no': 0})

        # Encode tumor_stage as categorical
        if 'tumor_stage' in input_df.columns:
            input_df['tumor_stage'] = input_df['tumor_stage'].astype(str)
            dummy_cols = pd.get_dummies(input_df['tumor_stage'], prefix='tumor_stage', dummy_na=True)
            input_df = pd.concat([input_df.drop('tumor_stage', axis=1), dummy_cols], axis=1)

        # Clip tumor_size to non-negative
        if 'tumor_size' in input_df.columns:
            input_df['tumor_size'] = input_df['tumor_size'].astype(float).clip(lower=0)

        # Fill missing numerical values in user input
        for col in numerical_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].fillna(df_features[col].median())

        # Encode other categorical variables
        for col in categorical_cols:
            if col in input_df.columns and col != 'tumor_stage':
                dummy_cols = [c for c in feature_names if c.startswith(f"{col}_")]
                for dummy in dummy_cols:
                    dummy_value = dummy.split(f"{col}_")[-1]
                    input_df[dummy] = 1 if input_df[col].iloc[0] == dummy_value else 0
                input_df = input_df.drop(columns=col)

        # Align user input with model feature names
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]
        if input_df.isna().any().any():
            raise ValueError(f"User input contains NaN: {input_df.columns[input_df.isna().any()].tolist()}")

        # Extract survival outcomes
        y_duration = df['overall_survival_months'].values
        y_event = df['overall_survival'].values
        logger.info(f"y_duration: {len(y_duration)}, y_event: {len(y_event)}")
        logger.info(f"Survival time stats: mean={np.mean(y_duration):.2f}, std={np.std(y_duration):.2f}, events={np.sum(y_event)}/{len(y_event)}")

        return df_encoded, input_df, y_duration, y_event

    def predict(self, df: pd.DataFrame, user_input: dict, dataset_type: str) -> tuple:
        """
        Predict median survival time using CoxPH model and return survival curve.

        Args:
            df (pd.DataFrame): Input dataset.
            user_input (dict): User-provided feature values.
            dataset_type (str): Dataset type ('Clinical', 'Genetic', 'Mutation').

        Returns:
            tuple: (median survival time in months, survival curve DataFrame).

        Raises:
            ValueError: If prediction fails due to invalid data or NaN values.
        """
        logger.info(f"Starting prediction for {dataset_type}...")

        # Load model, scaler, and feature names
        model, scaler, feature_names = self._load_model(dataset_type)

        # Preprocess data
        df_encoded, input_df, y_duration, y_event = self._preprocess_data(df, user_input, feature_names)

        # Scale features
        df_encoded_scaled = scaler.transform(df_encoded).astype(np.float32)
        x_input = scaler.transform(input_df).astype(np.float32)
        logger.info(f"Scaled input shape: {x_input.shape}")

        # Fine-tune model on current data
        model.fit(df_encoded_scaled, (y_duration, y_event), batch_size=64, epochs=10, verbose=True)
        model.compute_baseline_hazards()

        # Predict survival curve
        surv_input = model.predict_surv_df(x_input)
        logger.info(f"Survival curve sample: {surv_input.iloc[:5, :].to_dict()}")

        # Extract survival probabilities and times
        surv_probs = surv_input.iloc[:, 0].values
        if np.any(np.isnan(surv_probs)):
            raise ValueError("Survival probabilities contain NaN values.")
        times = surv_input.index

        # Calculate median survival time
        median_idx = np.where(surv_probs <= 0.5)[0]
        median_time = times[median_idx[0]] if median_idx.size > 0 else times[-1]
        logger.info(f"Median survival time: {median_time:.1f} months")

        return median_time, surv_input

class StreamlitApp:
    """Manages the Streamlit UI and application flow."""
    
    def __init__(self):
        """Initialize StreamlitApp with dataset loader, model, and RAG system."""
        self.dataset_loader = DatasetLoader()
        self.model = SurvivalModel()
        self.rag = RAGSystem()

        # Apply custom CSS for UI styling
        st.markdown("""
            <style>
            .main {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stSelectbox, .stNumberInput {
                background-color: white;
                border-radius: 5px;
                padding: 5px;
                margin-bottom: 10px;
            }
            .stExpander {
                background-color: #ffffff;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .stSuccess, .stInfo, .stError {
                border-radius: 5px;
                padding: 10px;
                margin-top: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

    def _create_input_fields(self, dataset_type: str) -> dict:
        """
        Create Streamlit input fields for dataset features in a grid layout.

        Args:
            dataset_type (str): Dataset type ('Clinical', 'Genetic', 'Mutation').

        Returns:
            dict: User-provided feature values.
        """
        df = self.dataset_loader.get_dataset(dataset_type)
        feature_cols = self.dataset_loader.get_feature_cols(dataset_type)
        
        # Group key fields in an expander
        with st.expander("Key Patient Information", expanded=True):
            st.markdown("**Enter patient details below**")
            # Create a 2-column grid for input fields
            cols = st.columns(2)
            user_input = {}
            for i, col in enumerate(feature_cols):
                dtype = df[col].dtype
                # Use formatted label
                label = format_label(col)
                # Assign field to left or right column
                with cols[i % 2]:
                    if col.lower() == 'chemotherapy':
                        user_input[col] = st.selectbox(
                            label,
                            ["yes", "no"],
                            help="Indicate if the patient received chemotherapy."
                        )
                    elif col.lower() == 'tumor_stage':
                        user_input[col] = st.selectbox(
                            label,
                            ["1", "2", "3", "4"],
                            help="Select the tumor stage (1-4)."
                        )
                    elif col.lower() == 'type_of_breast_surgery':
                        unique_vals = df[col].dropna().unique()
                        user_input[col] = st.selectbox(
                            label,
                            unique_vals if len(unique_vals) > 0 else ['Unknown'],
                            help="Specify the type of breast surgery (e.g., lumpectomy, mastectomy)."
                        )
                    elif dtype == object:
                        unique_vals = df[col].dropna().unique()
                        user_input[col] = st.selectbox(
                            label,
                            unique_vals if len(unique_vals) > 0 else ['Unknown'],
                            help=f"Select a value for {label}."
                        )
                    else:
                        min_val = float(df[col].min()) if not df[col].dropna().empty else 0
                        max_val = float(df[col].max()) if not df[col].dropna().empty else 100
                        user_input[col] = st.number_input(
                            label,
                            min_value=min_val,
                            max_value=max_val,
                            value=float(df[col].dropna().median() if not df[col].dropna().empty else 0),
                            help=f"Enter a value for {label} (range: {min_val} to {max_val})."
                        )
        return user_input

    def run(self):
        """Run the Streamlit application with enhanced UI."""
        st.title("HopeCast")

        # Dataset selection
        dataset_type = st.selectbox(
            "Choose Dataset Type:",
            ["Clinical", "Genetic", "Mutation"],
            help="Select the type of data to use for prediction."
        )

        # Create input fields
        user_input = self._create_input_fields(dataset_type)

        if st.button("📃 Predict & Explain"):
            with st.spinner("Processing patient data..."):
                try:
                    # Predict survival time and get survival curve
                    df = self.dataset_loader.get_dataset(dataset_type)
                    expected_time, surv_input = self.model.predict(df, user_input, dataset_type)

                    # Display survival curve at specific intervals (1, 5, 10, 15, 20 months)
                    st.markdown("### 🌟 Survival Probability Over Time")
                    st.markdown("Here’s a glimpse of your predicted survival probability at different time points:")
                    desired_times = [1, 5, 10,20,40,80,160,320]  # Desired time points in months
                    times = np.array(surv_input.index)  # Available time points in the survival curve
                    for desired_time in desired_times:
                        # Find the closest time point in the survival curve to the desired time
                        closest_idx = (np.abs(times - desired_time)).argmin()
                        closest_time = times[closest_idx]
                        prob = surv_input.iloc[closest_idx, 0] * 100  # Convert to percentage
                        st.markdown(f"- At **{closest_time:.1f} months**, there is a **{prob:.1f}% chance** of survival.")
                    st.markdown("*These points from your survival curve show your likelihood of survival at key intervals.*")

                    # Display median survival time with nice wording
                    years = expected_time // 12
                    months = expected_time % 12
                    time_str = f"{int(years)} years" if years > 0 else ""
                    if months > 0:
                        time_str += f" and {int(months)} months" if years > 0 else f"{int(months)} months"
                    st.markdown(f"###  Median Survival Time")
                    st.markdown(f"Based on the analysis, you have a **50% chance** of living beyond **{time_str}** ({expected_time:.1f} months). This means that half of the patients with similar characteristics are expected to survive at least this long.")

                    # Generate treatment advice
                    explanation = self.rag.explain(user_input, expected_time)
                    st.markdown("### 🧾 Treatment Advice")
                    st.write(explanation)

                    # Validate with severe case
                    severe_input = user_input.copy()
                    if 'tumor_stage' in severe_input:
                        severe_input['tumor_stage'] = '4'
                    if 'tumor_size' in severe_input:
                        severe_input['tumor_size'] = 30.0
                    severe_time, _ = self.model.predict(df, severe_input, dataset_type)
                    severe_years = severe_time // 12
                    severe_months = severe_time % 12
                    severe_time_str = f"{int(severe_years)} years" if severe_years > 0 else ""
                    if severe_months > 0:
                        severe_time_str += f" and {int(severe_months)} months" if severe_years > 0 else f"{int(severe_months)} months"
                    st.markdown(f"### 📊 Severe Case Validation")
                    st.markdown(f"For a severe case (Stage 4, Tumor Size 30), the median survival time is **{severe_time_str}** ({severe_time:.1f} months).")

                except Exception as e:
                    logger.error(f"Prediction or explanation error: {str(e)}", exc_info=True)
                    st.error(f"Error: {str(e)}\nPlease check your inputs or contact support at https://x.ai/api.")

def main():
    """Entry point for the Streamlit application."""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()