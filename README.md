# Financial Regulation Article Clustering

A Python-based tool for parsing, filtering, categorizing, and clustering financial regulatory articles from multiple sources (BRRD, EBA, FIVA_MOK/EN). The system uses TF-IDF pre-filtering combined with LLM-based similarity scoring to identify overlapping and contradictory requirements across different regulatory frameworks.

## 🎯 Project Overview

This project automates the analysis of financial regulation documents by:

-   **Parsing** regulatory documents into structured JSON format
-   **Filtering** relevant articles from unrelated content
-   **Categorizing** articles by risk type (credit, liquidity, market, operational, compliance)
-   **Clustering** similar articles using AI-powered similarity analysis

## 📋 Table of Contents

-   [Installation](#installation)
-   [Pipeline Overview](#pipeline-overview)
-   [Usage](#usage)
-   [Project Structure](#project-structure)
-   [Configuration](#configuration)
-   [Output Format](#output-format)

## 🚀 Installation

### Prerequisites

-   Python 3.10+
-   [Ollama](https://ollama.ai/) with the `gemma3:1b` model installed
-   Conda (recommended) or pip

### Setup with Conda

```bash
# Create environment from env.yml
conda env create -f env.yml
conda activate paragraph-classifier

# Install additional dependencies
pip install scikit-learn
```

### Install Ollama Model

```bash
# Pull the required model
ollama pull gemma3:1b
```

## 🔄 Pipeline Overview

The processing pipeline consists of 4 main stages:

```
Parse → Filter → Split by Risk → Cluster
```

### Stage 1: Parse

Convert PDF regulatory documents into structured JSON format with articles, paragraphs, and metadata.

### Stage 2: Filter

Use AI to classify articles as relevant or unrelated to financial institution risk management.

### Stage 3: Split by Risk Category

Categorize articles into 5 risk types:

-   Credit Risk
-   Liquidity Risk
-   Market Risk
-   Operational Risk
-   Compliance Risk

### Stage 4: Cluster

Identify similar/contradictory articles across different regulatory frameworks using TF-IDF + LLM analysis.

## 📖 Usage

### 1. Parse Regulatory Documents

```bash
# Parse EBA documents (run inside EBA folder)
cd EBA
python ../parse_EBA.py

# Parse FIVA_MOK documents (run inside FIVA_MOK folder)
cd FIVA_MOK
python ../parse_fiva_mok.py
```

**Output:** `eba_parsed.json`, `fiva_mok_parsed.json`

### 2. Filter Relevant Articles

```bash
python select_relevant.py
```

**Configuration:** Edit the input file path in `select_relevant.py`

**Output:**

-   `credit_related_*.json` (relevant articles)
-   `unrelated_*.json` (filtered out articles)

### 3. Split by Risk Category

```bash
python split_by_risk_category.py
```

**Configuration:** Edit file names in the code

**Output:** Creates 5 files in `categorized_cleaned_data/`:

-   `credit_risk_*.json`
-   `liquidity_risk_*.json`
-   `market_risk_*.json`
-   `operational_risk_*.json`
-   `compliance_risk_*.json`

### 4. Cluster Similar Articles

```bash
python binitys2.py
```

**Parameters (configurable in code):**

-   `tfidf_threshold`: TF-IDF similarity threshold for pre-filtering (default: 0.11)
-   `llm_threshold`: LLM similarity threshold for clustering (default: 0.86)
-   `early_exit`: Similarity score for immediate matching (default: 0.91)

**Output:** `clustered_articles.json`

## 📁 Project Structure

```
junction2025/
├── parse_EBA.py              # Parse EBA regulatory documents
├── parse_fiva_mok.py          # Parse FIVA_MOK documents
├── select_relevant.py         # Filter relevant articles using AI
├── split_by_risk_category.py # Categorize by risk type
├── binitys2.py               # Main clustering algorithm
├── visualize.ipynb           # Data visualization notebook
├── env.yml                   # Conda environment specification
├── categorized_cleaned_data/ # Processed articles by risk category
├── clustered_data/           # Clustering results
└── Results/                  # Analysis outputs
```

## ⚙️ Configuration

### Clustering Parameters

Edit `binitys2.py` main section:

```python
clusters = cluster_articles_with_tfidf(
    articles,
    tfidf_threshold=0.11,   # Lower = more candidates for LLM
    llm_threshold=0.86,     # Higher = stricter clustering
    early_exit=0.91,        # Stop comparing if very similar
)
```

### Input Files

The clustering script currently processes:

-   `categorized_cleaned_data/compliance_risk_eba.json`
-   `categorized_cleaned_data/compliance_risk_fiva_mok.json`

Modify the file paths in the `__main__` section to process other risk categories.

## 📊 Output Format

### Clustered Articles JSON

```json
[
  {
    "reference_label": "Document Name Article ID",
    "articles": [
      {
        "article": {
          "document name": "...",
          "article id": "...",
          "article name": "...",
          "article paragraphs": ["..."]
        },
        "reference_article": true
      },
      {
        "article": {...},
        "reference_article": false
      }
    ]
  }
]
```

-   **reference_label**: The cluster identifier (based on the first/reference article)
-   **reference_article**: `true` for cluster heads, `false` for matched articles

## 🤖 How Clustering Works

1. **TF-IDF Pre-filtering**: Computes text similarity using TF-IDF vectorization to reduce LLM calls
2. **Same-document filtering**: Skips articles from the same document to focus on cross-regulation overlap
3. **LLM Similarity Scoring**: Uses Gemma 3:1b via Ollama to assess semantic similarity of requirements
4. **Cluster Assignment**: Groups articles with similarity above threshold or creates new reference clusters
5. **Progressive Reference Building**: Matched articles become references for future comparisons

## 🔧 Troubleshooting

**KeyError on cluster labels**: Ensure you're using the latest version of `binitys2.py` with proper `reference_to_label` mapping.

**Ollama errors**: Verify Ollama is running and the model is installed:

```bash
ollama list
ollama pull gemma3:1b
```

**Poor clustering results**: Adjust thresholds based on your needs:

-   Lower `llm_threshold` for broader clusters
-   Raise `tfidf_threshold` to reduce LLM calls

## 📝 License

[Add your license here]

## 👥 Contributors

Junction 2025 Project Team
