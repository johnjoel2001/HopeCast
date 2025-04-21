import os
import pandas as pd
import kagglehub

def main():
    os.makedirs("data", exist_ok=True)

    print("⬇️ Downloading METABRIC gene expression dataset from KaggleHub...")

    # Download the dataset via KaggleHub
    dataset_path = kagglehub.dataset_download('raghadalharbi/breast-cancer-gene-expression-profiles-metabric')
    print(f"✅ Dataset downloaded to: {dataset_path}")

    # Load only the gene expression file
    gene_expr_file = os.path.join(dataset_path, 'METABRIC_RNA_Mutation.csv')

    # Copy it to the data folder
    os.system(f"cp '{gene_expr_file}' data/gene_expression.csv")

    # Load to confirm
    gene_df = pd.read_csv("data/gene_expression.csv", delimiter=',')

    print("📊 Gene Expression shape:", gene_df.shape)
    print("✅ Gene expression data saved in data/gene_expression.csv")

if __name__ == "__main__":
    main()
