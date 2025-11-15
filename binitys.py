import json
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# LLM CALL (Gemma 3:4 via Ollama)
# -----------------------------
def call_ollama(prompt, model="gemma3:1b"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8").strip()


# -----------------------------
# Convert article dict to text
# -----------------------------
def build_article_text(article):
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
    text_a = build_article_text(article_a)
    text_b = build_article_text(article_b)

    #     prompt = f"""
    # Do these two financial regulation articles have the same or a conflicting rule? Answer with a number.

    # Score 1.0 if the articles have any shared or conflicting rule. Score 0.0 if they do not.

    # Output: A single number: 0.0 or 1.0.

    # Article A: {text_a}

    # Article B: {text_b}

    # Answer:
    # """.strip()

    #     prompt = f"""
    # I am categorising the key risks of financial corporations granting credit.

    # You are a compliance analyst AI in the context of finance regulation. Compare two regulatory articles.

    # Instructions:

    # 1. Consider the articles **similar** if:
    #    - Both contain at least one paragraph or section that defines the **same requirement or obligation**, even if the wording is slightly different.
    #    - Both contain at least one paragraph or section that defines requirements for the same situation, even if the requirements **contradict** or differ in strictness.

    # 2. Consider the articles **not similar** if:
    #    - They only discuss the same topic or situation, but **no requirement or obligation overlaps or contradicts**.
    #    - They are about entirely different topics.

    # 3. Your task is to output a **single decimal number between 0 and 1** representing similarity:
    #    - 1.0 = articles contain at least one overlapping or contradicting requirement
    #    - 0.0 = no overlapping or contradicting requirements
    #    - You may use intermediate values (e.g., 0.5) if there is partial overlap, but only score highly if there is **actual requirement overlap or contradiction**.

    # Output: A single number between 0.00 and 1.00.

    # Article A:
    # {text_a}

    # Article B:
    # {text_b}

    # Answer:

    # """.strip()

    #     prompt = f"""
    # You are a compliance analyst AI. Compare the following two regulatory articles.

    # Output a single decimal number between 0 and 1 representing their similarity:
    # - 1.0 = very similar obligations and requirements imposed
    # - 0.0 = completely unrelated articles
    # - Round to two decimal places

    # Article A:
    # {text_a}

    # Article B:
    # {text_b}

    # Answer with only the number:
    # """.strip()
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

    response = call_ollama(prompt)
    try:
        score = float(response.strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = -1.0
    return score


# -----------------------------
# Cluster articles with TF-IDF pre-filter
# -----------------------------
def cluster_articles_with_tfidf(
    articles, tfidf_threshold=0.2, llm_threshold=0.8, early_exit=0.95
):
    """
    Cluster articles using TF-IDF to pre-select candidate references for LLM comparison.
    """
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
        print(
            f"\n\033[96m[{idx+1}/{len(articles)}]\033[0m Processing article {article_label}"
        )

        if idx == 0:
            # first article becomes reference
            reference_indices.append(idx)
            reference_to_label[idx] = article_label
            clusters[article_label] = [{"article": article, "reference_article": True}]
            print(f"→ Added as first reference")
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
            print(f"→ No TF-IDF candidates, added as new reference")
            continue

        # Compare with candidate references using LLM
        best_score = -5.0
        best_ref_label = None
        best_ref_idx = None
        prelim_matches = []

        for ref_idx in candidate_refs:
            ref_label = reference_to_label[ref_idx]
            score = compare_articles_score(article, articles[ref_idx])
            print(
                f"   Compared {article_label} -> {ref_label}: LLM similarity {score:.2f}"
            )

            if score > llm_threshold:
                prelim_matches.append((ref_label, score, ref_idx))

            if score > best_score:
                best_score = score
                best_ref_label = ref_label
                best_ref_idx = ref_idx

            if score >= early_exit:
                # early exit if very high similarity
                print(f"     → Early exit: similarity {score:.2f} >= {early_exit}")
                prelim_matches = [(ref_label, score, ref_idx)]
                best_score = score
                best_ref_label = ref_label
                best_ref_idx = ref_idx
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
            print(f"→ {article_label} assigned to {chosen_ref_label}")
        else:
            reference_indices.append(idx)
            reference_to_label[idx] = article_label
            clusters[article_label] = [{"article": article, "reference_article": True}]
            print(
                f"→ {article_label} did not match any reference, added as new reference"
            )

    return clusters


# -----------------------------
# Save clustered JSON
# -----------------------------
def save_clusters_to_json(clusters, filename="clustered_articles.json"):
    output = []
    for ref_label, articles_list in clusters.items():
        cluster_data = {"reference_label": ref_label, "articles": articles_list}
        output.append(cluster_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"\n Clustered articles saved to {filename}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Load articles from JSON
    with open(
        "categorized_cleaned_data/compliance_risk_eba_sanitized.json",
        "r",
        encoding="utf-8",
    ) as f:
        data1 = json.load(f)
    with open(
        "categorized_cleaned_data/compliance_risk_fiva_mok.json", "r", encoding="utf-8"
    ) as f:
        data2 = json.load(f)

    articles = data1 + data2

    clusters = cluster_articles_with_tfidf(
        articles,
        # tfidf_threshold=0.05,  # pre-filter
        tfidf_threshold=0.2,  # pre-filter
        # tfidf_threshold=0.175,  # pre-filter
        # llm_threshold=0.7,  # preliminary LLM match
        llm_threshold=0.84,  # preliminary LLM match
        early_exit=0.91,  # stop LLM comparisons if very high similarity
    )

    save_clusters_to_json(clusters)
