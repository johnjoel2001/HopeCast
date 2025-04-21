import pandas as pd
import numpy as np
import os

def load_data(file_path='data/gene_expression.csv'):
    print("🔹 Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded data with shape: {df.shape}")
    return df

def clean_main_df(df):
    print("🔹 Cleaning mutation columns...")
    mutation_cols = df.columns[520:]
    df[mutation_cols] = df[mutation_cols].apply(pd.to_numeric, errors='coerce').fillna(1).astype(int)
    return df

def extract_clinical_df(df):
    print("🔹 Extracting clinical attributes...")
    clinical_df = df.drop(columns=df.columns[31:], axis=1)
    clinical_df['overall_survival'] = df['overall_survival']
    clinical_df['overall_survival_months'] = df['overall_survival_months']
    clinical_df['patient_id'] = df['patient_id']
    print(f"✅ Clinical data shape: {clinical_df.shape}")
    return clinical_df

def extract_genetic_df(df):
    print("🔹 Extracting gene expression (mRNA Z-score) attributes...")
    genetic_df = df.copy()
    genetic_df = genetic_df.drop(columns=df.columns[520:], axis=1)  # drop mutation data
    genetic_df = genetic_df.drop(columns=genetic_df.columns[4:35])  # drop clinical data
    genetic_df = genetic_df.drop(columns=['age_at_diagnosis','type_of_breast_surgery', 'cancer_type'])
    genetic_df = genetic_df.iloc[:, :-174]  # drop trailing mutations if any
    genetic_df['overall_survival'] = df['overall_survival']
    genetic_df['overall_survival_months'] = df['overall_survival_months']
    genetic_df['patient_id'] = df['patient_id']
    print(f"✅ Genetic data shape: {genetic_df.shape}")
    return genetic_df

def extract_mutation_df(df):
    print("🔹 Extracting mutation attributes...")
    mutation_df = df.iloc[:, 520:].copy()
    mutation_df.insert(0, 'overall_survival', df['overall_survival'])
    mutation_df.insert(1, 'overall_survival_months', df['overall_survival_months'])
    mutation_df.insert(0, 'patient_id', df['patient_id'])
    print(f"✅ Mutation data shape: {mutation_df.shape}")
    return mutation_df

def save_cleaned_data(clinical_df, genetic_df, mutation_df):
    print("💾 Saving cleaned datasets...")
    clinical_df.to_csv('data/clinical_data.csv', index=False)
    genetic_df.to_csv('data/genetic_data.csv', index=False)
    mutation_df.to_csv('data/mutation_data.csv', index=False)
    print("✅ All files saved in /data folder.")

def main():
    os.makedirs('data', exist_ok=True)
    df = load_data()
    df = clean_main_df(df)
    clinical_df = extract_clinical_df(df)
    genetic_df = extract_genetic_df(df)
    mutation_df = extract_mutation_df(df)
    save_cleaned_data(clinical_df, genetic_df, mutation_df)

if __name__ == "__main__":
    main()
