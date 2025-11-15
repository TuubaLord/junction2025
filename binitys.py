import json
import subprocess


# -----------------------------
# LLM CALL (Gemma 3:4 via Ollama)
# -----------------------------
def call_ollama(prompt, model="gemma3:4b"):
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
    text = f"Document Name: {article.get('document name', '')}\n"
    text += f"Article ID: {article.get('article id', '')}\n"
    text += f"Article Title: {article.get('article name', '')}\n"
    text += "Paragraphs:\n"
    for p in article.get("article paragraphs", []):
        text += f"- {p}\n"
    return text


# -----------------------------
# Compute similarity score 0-1
# -----------------------------
def compare_articles_score(article_a, article_b):
    text_a = build_article_text(article_a)
    text_b = build_article_text(article_b)

    prompt = f"""
You are a compliance analyst AI. Compare the following two regulatory articles.

Output a single decimal number between 0 and 1 representing their similarity:
- 1.0 = very similar obligations
- 0.0 = completely unrelated
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
        score = 0.0
    return score


# -----------------------------
# Cluster articles by reference
# -----------------------------
def cluster_articles_by_similarity(articles, threshold=0.6):
    """
    Cluster articles by similarity:
    - First article becomes first reference
    - Compare each next article to all reference articles
    - If similarity > threshold, record preliminary match
    - Assign article to reference with highest similarity
    - If no match above threshold, create new reference article
    """

    total_articles = len(articles)

    reference_articles = []  # list of reference articles
    clusters = {}  # key: reference label, value: list of articles

    for idx, article in enumerate(articles):
        article_label = (
            f"{article.get('document name', '')} {article.get('article id', '')}"
        )

        # Probably section title or something
        if len(build_article_text(article)) < 100:
            continue

        progress_pct = (idx + 1) / total_articles * 100
        print(
            f"\n[\033[1;32m{idx + 1}/{total_articles}\033[0m] Processing article {article_label} ({progress_pct:.1f}%)"
        )

        if idx == 0:
            # first article becomes reference
            reference_articles.append(article)
            clusters[article_label] = [{"article": article, "reference_article": True}]
            print(f"→ Article {article_label} added as first reference\n")
            continue

        # compare to all reference articles
        prelim_matches = []
        best_score = -1.0
        best_ref_label = None

        for ref in reference_articles:
            # Skip if reference article is from the same document
            if article.get("document name") == ref.get("document name"):
                continue

            ref_label = f"{ref.get('document name', '')} {ref.get('article id', '')}"
            score = compare_articles_score(article, ref)
            print(f"   Compared {article_label} -> {ref_label}: similarity {score:.2f}")

            if score > threshold:
                prelim_matches.append((ref_label, score))

            if score > best_score:
                best_score = score
                best_ref_label = ref_label

            if score >= 0.95:
                # skip further comparisons for this article
                best_score = score
                best_ref_label = ref_label
                prelim_matches = [(ref_label, score)]  # override prelim matches
                print(
                    f"     → High similarity {score:.2f} reached, skipping further comparisons"
                )
                break

        # Decide assignment
        if prelim_matches:
            # assign to reference with highest score
            chosen_ref_label = max(prelim_matches, key=lambda x: x[1])[0]
            clusters[chosen_ref_label].append(
                {"article": article, "reference_article": False}
            )
            print(
                f"   → {article_label} matched with {chosen_ref_label} (preliminary match)\n"
            )
        else:
            # no match above threshold → new reference
            reference_articles.append(article)
            clusters[article_label] = [{"article": article, "reference_article": True}]
            print(
                f"   → {article_label} did not match any reference, added as new reference\n"
            )

    return clusters


# -----------------------------
# Save clustered JSON
# -----------------------------
def save_clusters_to_json(clusters, filename="clustered_articles.json"):
    # Convert to list format
    output = []
    for ref_label, articles_list in clusters.items():
        cluster_data = {"reference_label": ref_label, "articles": articles_list}
        output.append(cluster_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Clustered articles saved to {filename}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Load articles from JSON
    with open(
        "categorized_cleaned_data/compliance_risk_eba.json", "r", encoding="utf-8"
    ) as f:
        data1 = json.load(f)
    with open(
        "categorized_cleaned_data/compliance_risk_fiva_mok.json", "r", encoding="utf-8"
    ) as f:
        data2 = json.load(f)

    articles = data1 + data2
    # Cluster articles
    clusters = cluster_articles_by_similarity(articles, threshold=0.6)

    # Save to JSON
    save_clusters_to_json(clusters)
