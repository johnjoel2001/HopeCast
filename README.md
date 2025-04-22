# HopeCast: Breast Cancer Survival Prediction & Personalized Treatment Guidance

## Overview

- HopeCast is an AI-powered web application that predicts breast cancer survival time using clinical, genetic, and mutation data. 
- It also provides personalized treatment explanations by retrieving information from real medical literature.
- The project integrates classical ML (XGBoost), deep learning (DeepSurv), and a Retrieval-Augmented Generation (RAG) system for explainability.



##  Features

- Predicting survival using **clinical**, **genetic** and **mutation** data
- Comparing performance between **Naive**, **XGBoost**, and **DeepSurv** models
- Understanding predictions with **RAG-powered explanations** using real **PubMed** research
- User-friendly interface built with **Streamlit**

## How It Works

1. **Choose a dataset** (Clinical, Genetic, or Mutation)
2. **Enter patient details** like age, tumor stage, and treatment history
3. The model predicts:
   - **Median survival time**
   - **Survival probability at different months**
4. A **Retrieval-Augmented Generation (RAG)** system is used to:
   - Search for **relevant articles from PubMed**
   - Extract meaningful medical content
   - Generate **clear, personalized treatment explanations** using a language model (via Grok API)


##  Model Results

| Dataset  | Naive Baseline | XGBoost | DeepSurv |
|----------|----------------|---------|----------|
| Clinical | 0.5000         | 0.6001  | **0.7135** |
| Genetic  | 0.5000         | 0.6513  | **0.6730** |
| Mutation | 0.5000         | 0.5687  | **0.5611** |

## Try the HopeCast app here:  

 [https://hopecast-service-1004867270011.us-central1.run.app/](https://hopecast-service-1004867270011.us-central1.run.app/)

##  Setup Instructions to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/johnjoel2001/HopeCast.git
cd HopeCast
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Add Environment Variables ( Required if running on local machine )

Create a .env file in the project root with:

```bash
XAI_API_KEY=your_openai_or_grok_api_key
ENTREZ_EMAIL=your_email@example.com
```

## Steps to Run all the Models Seperately

### 1. Download and Prepare Data

```bash
python scripts/make_dataset.py
python scripts/clean_data.py
```
### 2: Train Models

Naive Baseline

```bash
python scripts/naive_model.py
```

XGBoost

```bash
python scripts/non_deep_learning.py
```

DeepSurv

```bash
python scripts/deep_learning.py
```
## Steps to Run the App

```bash
streamlit run app.py
```

## Ethics & Responsibility

- HopeCast is a research tool, not a clinical decision system

- All data is anonymized and sourced from METABRIC (public)

- Explanations are generated from open-access PubMed literature

- Built to support, not replace medical professionals

## Acknowledgements

- METABRIC Dataset (via Kaggle)

- DeepSurv (Katzman et al., 2018)

- xAI Grok API

- PubMed/NCBI Entrez

- FAISS, Lifelines, PyCox, Streamlit, and other open-source libraries






