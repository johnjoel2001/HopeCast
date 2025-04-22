import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import faiss
from Bio import Entrez
import pickle
import time
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:

    """
    Retrieving PubMed articles and generates treatment advice using Grok API
    
    """
    
    def __init__(self, max_articles: int = 50, cache_file: str = "pubmed_cache.pkl"):

        """
        Initializing RAGSystem for article retrieval and explanation

        Args:
            max_articles (int): Maximum number of PubMed articles to fetch.
            cache_file (str): File to cache PubMed articles (relative to project root)

        """
        # Loading environment variables from the project root 

        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        load_dotenv(env_path)
        if not os.getenv("XAI_API_KEY"):
            raise ValueError("XAI_API_KEY not found in environment variables. Please set it in .env.")
        if not os.getenv("ENTREZ_EMAIL") or "@" not in os.getenv("ENTREZ_EMAIL"):
            raise ValueError("ENTREZ_EMAIL not configured or invalid in environment variables. Please set it in .env.")
        Entrez.email = os.getenv("ENTREZ_EMAIL")
        logger.info("Environment variables for RAGSystem loaded successfully.")

        self.max_articles = max_articles

        # Storing cache file in the project root 

        self.cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), cache_file)
      
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.documents = self._load_or_fetch_pubmed_articles()
        self.index = self._build_index()

    def _load_or_fetch_pubmed_articles(self) -> list:
        """
        Load cached PubMed articles or fetch new ones.

        Returns:
            list: List of article strings (title: abstract)

        """
        try:
            with open(self.cache_file, 'rb') as f:
                logger.info(f"Loaded {self.cache_file} from cache.")
                return pickle.load(f)
        except FileNotFoundError:
            documents = self._fetch_pubmed_articles()
            with open(self.cache_file, 'wb') as f:
                pickle.dump(documents, f)
            logger.info(f"Saved {self.cache_file} to cache.")
            return documents

    def _fetch_pubmed_articles(self) -> list:
        """
        Fetch PubMed articles using Entrez API.

        Returns:
            list: List of article strings

        """
        logger.info("Fetching PubMed articles...")
        query = "breast cancer treatment (HER2-positive OR ER-positive OR triple-negative)[Title/Abstract] AND (2020/01/01:2025/12/31[Date - Publication])"
        time.sleep(0.5)
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=self.max_articles, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.error(f"PubMed esearch error: {str(e)}")
            return ["No relevant PubMed articles found."]
        
        pmids = record['IdList']
        if not pmids:
            logger.warning("No PubMed articles found.")
            return ["No relevant PubMed articles found."]
        
        documents = []
        for pmid in pmids:
            time.sleep(0.5)
            try:
                handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
                record = Entrez.read(handle)
                handle.close()
                article = record['PubmedArticle'][0]
                title = article['MedlineCitation']['Article']['ArticleTitle']
                abstract = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', '')
                if isinstance(abstract, list):
                    abstract = ' '.join(abstract)
                if abstract:
                    documents.append(f"{title}: {abstract}")
            except (KeyError, IndexError, Exception) as e:
                logger.warning(f"Error fetching article {pmid}: {str(e)}")
                continue
        logger.info(f"Fetched {len(documents)} PubMed articles.")
        return documents if documents else ["No relevant PubMed articles found."]

    def _build_index(self) -> faiss.IndexFlatL2:
        """
        Build FAISS index for document retrieval.

        Returns:
            faiss.IndexFlatL2: FAISS index for article embeddings

        """
        logger.info("Building FAISS index...")
        embeddings = self.embedding_model.encode(self.documents)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        logger.info("FAISS index built.")
        return index

    def explain(self, patient_data: dict, expected_time: float) -> str:
        """
        Generate treatment advice using Grok API and PubMed articles

        Args:
            patient_data (dict): Patient feature values.
            expected_time (float): Predicted survival time in months.

        Returns:
            str: Treatment advice or error message.
        """
        logger.info("Generating explanation for patient data (anonymized).")
        query = f"Patient with {patient_data}, expected survival time: {expected_time:.1f} months."
        embedding = self.embedding_model.encode([query]).astype(np.float32)
        _, indices = self.index.search(embedding, 2)
        context = [self.documents[i] for i in indices[0]]
        
        prompt = f"""Based on the following articles:\n{chr(10).join(context)}\n\nExplain treatment options and provide advice for this case: {query}"""
        logger.info(f"Calling Grok API with prompt: {prompt[:100]}...")
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
        }
        data = {
            "messages": [
                {"role": "system", "content": "You are a medical assistant. Provide a clear, concise explanation of treatment options and advice based on the provided PubMed articles, patient data, and expected survival time."},
                {"role": "user", "content": prompt}
            ],
            "model": "grok-beta",
            "stream": False,
            "temperature": 0.5,
            "max_tokens": 1000
        }
        try:
            time.sleep(1.0)
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            explanation = response_data['choices'][0]['message']['content'].strip()
            if response_data['choices'][0].get('finish_reason') == 'length':
                explanation += "\n\n**Warning**: The explanation may be incomplete due to length limits. Please refine the input or contact support."
            logger.info("Grok API call successful.")
            return explanation
        except Exception as e:
            logger.error(f"Grok API error: {str(e)}")
            return f"Error calling Grok API: {str(e)}"