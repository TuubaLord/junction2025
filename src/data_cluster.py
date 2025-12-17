import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from helpers import query_llm


# -----------------------------
# Convert article dict to text
# -----------------------------
def build_article_text(article):
    """
    Convert an article dictionary to a formatted text string.

    Args:
        article (dict): Article dictionary containing document info and paragraphs

    Returns:
        str: Formatted text combining document name, article ID, name, and paragraphs
    """
    text = f"{article.get('document name', '')}\n"
    text += f"{article.get('article id', '')}\n"
    text += f"{article.get('article name', '')}\n"
    for p in article.get("article paragraphs", []):
        text += f"{p}\n"
    return text


# -----------------------------
# Compute LLM similarity score
# -----------------------------
def compare_articles_score(article_a, article_b):
    """
    Compare two articles using LLM and return a similarity score.

    Args:
        article_a (Dict): First article dictionary
        article_b (Dict): Second article dictionary

    Returns:
        float: Similarity score between 0.0 and 1.0, or -1.0 if LLM fails
    """
    text_a = build_article_text(article_a)
    text_b = build_article_text(article_b)

    prompt = f"""
    You are a compliance analyst AI for financial regulation. Compare the following two regulatory articles.

    Output a single decimal number between 0 and 1 representing their similarity:
    - 1.0 = the articles contain overlapping or contradictory requirements/obligations
    - 0.5 = the articles have similar topics, but the requirement/obligations differ
    - 0.0 = completely unrelated articles
    - Round to two decimal places

    Article A:
    {text_a}

    Article B:
    {text_b}

    Answer with only the number:
    """.strip()

    try:
        response = query_llm(prompt)
        score = float(response.strip())
        score = max(0.0, min(1.0, score))
    except (ValueError, Exception) as e:
        print(f"Warning: LLM comparison failed: {e}")
        score = -1.0
    return score


def cluster_articles(
    sources,
    category,
    tfidf_threshold=0.2,
    llm_threshold=0.84,
    early_exit=0.91,
    verbose=True,
):
    """
    Main function to cluster articles from a specified source and category.

    Args:
        source (str): Source to process ("eba" or "fiva_mok")
        category (str): Risk category (e.g., "compliance_risk", "credit_risk", etc.)
        tfidf_threshold (float): Minimum TF-IDF similarity to consider candidates
        llm_threshold (float): Minimum LLM similarity score for clustering
        early_exit (float): LLM score threshold for early termination
        verbose (bool): If True, print progress information

    Returns:
        dict: Clusters with reference article labels as keys and lists of articles as values
    """

    # Load articles
    articles = []
    for source in sources:
        with open(
            Path(__file__).parent.parent.resolve()
            / "data"
            / "intermediate"
            / "categorized"
            / f"{category}_{source}.json",
            "r",
            encoding="utf-8",
        ) as f:
            articles.extend(json.load(f))

    if not articles:
        print(f"No articles found.")
        return {}

    # Cluster articles
    if verbose:
        print(f"Found {len(articles)} articles")
        print("Clustering articles...")

    reference_indices = []  # Track indices of reference articles
    reference_to_label = {}  # Map article index to cluster label
    clusters = {}

    # Precompute TF-IDF for all articles
    corpus = [build_article_text(a) for a in articles]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    for idx, article in enumerate(articles):
        article_label = (
            f"{article.get('document name', '')} {article.get('article id', '')}"
        )
        if verbose:
            print(
                f"\n\033[96m[{idx + 1}/{len(articles)}]\033[0m Processing article {article_label}"
            )

        if idx == 0:
            # first article becomes reference
            reference_indices.append(idx)
            reference_to_label[idx] = article_label
            clusters[article_label] = [{"article": article, "reference_article": True}]
            if verbose:
                print("→ Added as first reference")
            continue

        # Compute TF-IDF cosine similarity to all references
        candidate_refs = []
        article_vec = tfidf_matrix[idx]

        for ref_idx in reference_indices:
            ref_vec = tfidf_matrix[ref_idx]
            sim = cosine_similarity(article_vec, ref_vec)[0][0]

            # skip same-document references
            if article.get("document name") == articles[ref_idx].get("document name"):
                continue

            if sim >= tfidf_threshold:
                candidate_refs.append(ref_idx)

        if not candidate_refs:
            # No candidate references -> add as new reference
            reference_indices.append(idx)
            reference_to_label[idx] = article_label
            clusters[article_label] = [{"article": article, "reference_article": True}]
            if verbose:
                print("→ No TF-IDF candidates, added as new reference")
            continue

        # Compare with candidate references using LLM
        best_score = -5.0
        prelim_matches = []

        for ref_idx in candidate_refs:
            ref_label = reference_to_label[ref_idx]
            score = compare_articles_score(article, articles[ref_idx])
            if verbose:
                print(
                    f"   Compared {article_label} -> {ref_label}: LLM similarity {score:.2f}"
                )

            if score > llm_threshold:
                prelim_matches.append((ref_label, score, ref_idx))

            if score > best_score:
                best_score = score

            if score >= early_exit:
                # early exit if very high similarity
                if verbose:
                    print(f"     → Early exit: similarity {score:.2f} >= {early_exit}")
                prelim_matches = [(ref_label, score, ref_idx)]
                break

        if prelim_matches:
            chosen_ref_label = max(prelim_matches, key=lambda x: x[1])[0]
            clusters[chosen_ref_label].append(
                {"article": article, "reference_article": False}
            )
            # Add matched article as a candidate reference for future articles
            # Store the cluster label (not article_label) so future lookups work
            reference_indices.append(idx)
            reference_to_label[idx] = chosen_ref_label
            if verbose:
                print(f"→ {article_label} assigned to {chosen_ref_label}")
        else:
            reference_indices.append(idx)
            reference_to_label[idx] = article_label
            clusters[article_label] = [{"article": article, "reference_article": True}]
            if verbose:
                print(
                    f"→ {article_label} did not match any reference, added as new reference"
                )

    if verbose:
        print(f"Clustered into {len(clusters)} groups")

    return clusters


# -----------------------------
# Save clustered JSON
# -----------------------------
def save_clusters_to_json(clusters, filename="clustered_articles.json"):
    """
    Save clustered articles to a JSON file.

    Args:
        clusters (dict): Dictionary of clusters with reference labels as keys
        filename (str): Output filename for the JSON file

    Returns:
        None
    """
    output = []
    for ref_label, articles_list in clusters.items():
        cluster_data = {"reference_label": ref_label, "articles": articles_list}
        output.append(cluster_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"\n Clustered articles saved to {filename}")


if __name__ == "__main__":
    # Example usage: Cluster articles from specified source
    sources = ["EBA", "FIVA_MOK"]

    category = "compliance_risk"
    # category = "credit_risk"
    # category = "liquidity_risk"
    # category = "market_risk"
    # category = "operational_risk"

    # Cluster articles
    clusters = cluster_articles(
        sources,
        category,
        tfidf_threshold=0.2,
        llm_threshold=0.84,
        early_exit=0.91,
        verbose=True,
    )

    # Save to file
    print("Saving results")
    save_dir = (
        Path(__file__).parent.parent.resolve() / "data" / "intermediate" / "clustered"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    if clusters:
        save_clusters_to_json(clusters, save_dir / f"{category}.json")
        print(f"Successfully clustered articles into {len(clusters)} groups")
    else:
        print(f"No clusters created for category: {category}")
