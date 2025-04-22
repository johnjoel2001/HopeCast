import os
import pandas as pd
import kagglehub

class MetabricDownloader:
    """
    Handles downloading and saving the METABRIC gene expression dataset
    
    """

    def __init__(self, save_dir='data'):
        self.save_dir = save_dir
        self.local_file = os.path.join(self.save_dir, 'gene_expression.csv')

    def download_dataset(self):
        """
        Downloading the dataset from KaggleHub and saves the gene expression file
        
        """
        os.makedirs(self.save_dir, exist_ok=True)
        print("Downloading METABRIC gene expression dataset from KaggleHub...")

        # Using KaggleHub to download dataset
        dataset_path = kagglehub.dataset_download('raghadalharbi/breast-cancer-gene-expression-profiles-metabric')
        print(f"Dataset downloaded to: {dataset_path}")

        # Path to the gene expression file within the dataset
        gene_expr_file = os.path.join(dataset_path, 'METABRIC_RNA_Mutation.csv')

        # Copying the gene expression file to the target directory
        os.system(f"cp '{gene_expr_file}' '{self.local_file}'")
        print("Gene expression data saved in:", self.local_file)

    def verify_download(self):

        """
        Loading the gene expression file to confirm it was saved correctly
        
        """
        df = pd.read_csv(self.local_file)
        print("Gene expression shape:", df.shape)
        return df


def main():
    downloader = MetabricDownloader()
    downloader.download_dataset()
    downloader.verify_download()


if __name__ == "__main__":
    main()
