import pandas as pd
import numpy as np
import os

class BreastCancerDataProcessor:

    """
    Processes and splits gene expression data into clinical, genetic, and mutation subsets
    
    """

    def __init__(self, file_path='data/gene_expression.csv'):
        self.file_path = file_path
        self.df = None

    def load_data(self):

        """
        Loading the raw dataset
        """
        print("Loading dataset")
        self.df = pd.read_csv(self.file_path)
        print(f"Loaded data with shape: {self.df.shape}")

    def clean_mutation_columns(self):
        """
        Cleaning mutation columns by converting to numeric and filling missing values
        
        """
        print("Cleaning mutation columns")
        mutation_cols = self.df.columns[520:]
        self.df[mutation_cols] = self.df[mutation_cols].apply(pd.to_numeric, errors='coerce').fillna(1).astype(int)

    def extract_clinical_data(self):
        """
        Extracing clinical attributes from the dataset
        
        """
        print("Extracting clinical attributes")
        clinical_df = self.df.drop(columns=self.df.columns[31:], axis=1)
        clinical_df['overall_survival'] = self.df['overall_survival']
        clinical_df['overall_survival_months'] = self.df['overall_survival_months']
        clinical_df['patient_id'] = self.df['patient_id']
        print(f"Clinical data shape: {clinical_df.shape}")
        return clinical_df

    def extract_genetic_data(self):
        """
        Extracting gene expression data (mRNA Z-scores)
        
        """
        print("Extracting gene expression attributes")
        genetic_df = self.df.copy()
        genetic_df = genetic_df.drop(columns=self.df.columns[520:], axis=1)
        genetic_df = genetic_df.drop(columns=genetic_df.columns[4:35])
        genetic_df = genetic_df.drop(columns=['age_at_diagnosis','type_of_breast_surgery', 'cancer_type'])
        genetic_df = genetic_df.iloc[:, :-174]
        genetic_df['overall_survival'] = self.df['overall_survival']
        genetic_df['overall_survival_months'] = self.df['overall_survival_months']
        genetic_df['patient_id'] = self.df['patient_id']
        print(f"Genetic data shape: {genetic_df.shape}")
        return genetic_df

    def extract_mutation_data(self):
        """
        Extracting mutation data from the dataset
        
        """
        print("Extracting mutation attributes")
        mutation_df = self.df.iloc[:, 520:].copy()
        mutation_df.insert(0, 'overall_survival', self.df['overall_survival'])
        mutation_df.insert(1, 'overall_survival_months', self.df['overall_survival_months'])
        mutation_df.insert(0, 'patient_id', self.df['patient_id'])
        print(f"Mutation data shape: {mutation_df.shape}")
        return mutation_df

    def save_datasets(self, clinical_df, genetic_df, mutation_df):
        """
        Saving the processed datasets to CSV files
        
        """
        print("Saving cleaned datasets")
        clinical_df.to_csv('data/clinical_data.csv', index=False)
        genetic_df.to_csv('data/genetic_data.csv', index=False)
        mutation_df.to_csv('data/mutation_data.csv', index=False)
        print("All files saved in /data folder")


def main():
    os.makedirs('data', exist_ok=True)
    processor = BreastCancerDataProcessor()
    processor.load_data()
    processor.clean_mutation_columns()
    clinical_df = processor.extract_clinical_data()
    genetic_df = processor.extract_genetic_data()
    mutation_df = processor.extract_mutation_data()
    processor.save_datasets(clinical_df, genetic_df, mutation_df)


if __name__ == "__main__":
    main()