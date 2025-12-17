# Financial Regulation Article Clustering

## Contributors

-   osma.ovaskainen@aalto.fi
-   aleksanteri.hamalainen@aalto.fi
-   aku.karhinen@aalto.fi
-   niko.laukkanen@outlook.com

## 🎯 Project Overview

This project automates the analysis of financial regulation documents by:

-   **Parsing** provided JSON articles
-   **Filtering** relevant articles from unrelated content
-   **Categorizing** articles by risk type (credit, liquidity, market, operational, compliance)
-   **Clustering** similar articles with TF-IDF and LLM
-   **Comparing** clustered articles with LLM for contradictions or overlap

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
-   [Ollama](https://ollama.ai/) with the `gemma3:1b` & `gemma3:4b` model installed
-   Conda (recommended) or pip

### Setup with Conda

```bash
# Create environment from env.yml
conda env create -f env.yml
conda activate bureaucracy-buster

# Install additional dependencies
pip install scikit-learn
```

### Install Ollama Model

```bash
# Pull the required models
ollama pull gemma3:1b
ollama pull gemma3:4b
```

## Pipeline Overview

The processing pipeline consists of 5 main stages:

```
Parse → Filter → Split by Risk → Cluster → Compare
```

## Usage

### Run the Complete Pipeline

The easiest way to run the entire pipeline is using `run_all.py`:

```bash
cd src
python run_all.py
```

This will execute all 5 stages sequentially for all sources (EBA, FIVA_MOK) and all risk categories.

---

### Individual Pipeline Steps

You can also run each step individually. Edit the source/category variables in the file as needed, then run:

### 1. Parse Documents (`data_parse.py`)

Parses regulatory documents from the `data/gold/` folder into structured JSON.

```bash
# Edit SOURCE variable in data_parse.py as needed (EBA or FIVA_MOK)
python src/data_parse.py
```

**Input:** Documents in `data/gold/{source}/`  
**Output:** `data/intermediate/parsed/all_{source}_parsed.json`

### 2. Filter Relevant Articles (`data_filter.py`)

Uses an LLM to classify articles as relevant or unrelated to credit-giving organizations.

```bash
# Edit SOURCE variable in data_filter.py as needed
python src/data_filter.py
```

**Input:** Parsed documents from Step 1  
**Output:**

-   `data/intermediate/filtered/credit_related_{source}.json`
-   `data/intermediate/filtered/unrelated_{source}.json`

### 3. Categorize by Risk Type (`data_categorize.py`)

Uses an LLM to classify articles into 5 risk categories.

```bash
# Edit SOURCE variable in data_categorize.py as needed
python src/data_categorize.py
```

**Input:** Filtered articles from Step 2  
**Output:** `data/intermediate/categorized/{category}_{source}.json` for each category:

-   `credit_risk_{source}.json`
-   `liquidity_risk_{source}.json`
-   `market_risk_{source}.json`
-   `operational_risk_{source}.json`
-   `compliance_risk_{source}.json`

### 4. Cluster Similar Articles (`data_cluster.py`)

Clusters articles across sources using TF-IDF pre-filtering + LLM similarity scoring.

```bash
# Edit category variable in data_cluster.py as needed
python src/data_cluster.py
```

**Parameters (configurable in file):**

-   `tfidf_threshold`: TF-IDF pre-filter threshold (default: 0.2)
-   `llm_threshold`: LLM similarity threshold for clustering (default: 0.84)
-   `early_exit`: Stop comparing if very high similarity (default: 0.91)

**Input:** Categorized articles from Step 3  
**Output:** `data/intermediate/clustered/{category}.json`

### 5. Compare Clustered Articles (`data_compare.py`)

Compares articles within clusters to identify overlaps and contradictions.

```bash
# Edit category variable in data_compare.py as needed
python src/data_compare.py
```

**Input:** Clustered articles from Step 4  
**Output:** `results/{category}.json` containing:

-   `overlap`: Articles with same regulatory requirements
-   `contradiction`: Articles with conflicting requirements
-   `bloat`: Generic similarities without concrete overlap

## How Clustering Works

1. **TF-IDF Pre-filtering**: Computes text similarity using TF-IDF vectorization to reduce LLM calls
2. **Same-document filtering**: Skips articles from the same document to focus on cross-regulation overlap
3. **LLM Similarity Scoring**: Uses Gemma 3:1b via Ollama to assess semantic similarity of requirements
4. **Cluster Assignment**: Groups articles with similarity above threshold or creates new reference clusters
5. **Progressive Reference Building**: Unmatched articles become references for future comparisons

**Poor clustering results**: Adjust thresholds based on your needs:

-   Lower `llm_threshold` for broader clusters
-   Raise `tfidf_threshold` to reduce LLM calls

## License

MIT
