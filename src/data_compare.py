import itertools
import json
from pathlib import Path

from helpers import query_llm
import logging
from tqdm import tqdm

CLASSIFICATION_INSTRUCTIONS = """
Compare Paragraph A and Paragraph B. Output one label: contradiction, overlap, or bloat.

contradiction
Use only if:

Their obligations/permissions/prohibitions cannot be followed at the same time, and

One directly negates or forbids what the other requires.
If simultaneous compliance is possible, it is not a contradiction.

overlap

Use only if both paragraphs regulate the same narrowly defined requirement, addressing the same mechanism, same scope, and same regulatory intent, such that compliance with one would materially satisfy the other.

They are not a contradiction.

bloat
Use when:

Any similarity is generic (e.g., “risk”, “capital”, “liquidity”), and

They do not regulate the same concrete behaviour/process.

Decision order:

contradiction

overlap

bloat

Output: one lowercase word, nothing else.
"""


def classify_paragraph_pair(p_a, p_b):
    """
    Classify the relationship between two regulatory text paragraphs using an LLM.

    Args:
        p_a (str): First paragraph or article text to compare
        p_b (str): Second paragraph or article text to compare

    Returns:
        str: One of 'contradiction', 'overlap', or 'bloat'
            - 'contradiction': Requirements cannot be followed simultaneously
            - 'overlap': Same narrowly defined requirement with same regulatory intent
            - 'bloat': Generic similarity without regulating same concrete behaviour

    Notes:
        - Uses gemma3:4b model via Ollama
        - Returns 'bloat' as default if LLM call fails or response is unclear
        - Logs warnings for invalid responses and errors for failures
    """

    prompt = f"""
{CLASSIFICATION_INSTRUCTIONS}

Paragraph A:
\"\"\"{p_a}\"\"\"

Paragraph B:
\"\"\"{p_b}\"\"\"
"""

    # Execute prompt with error handling
    try:
        raw_response = query_llm(prompt, model="gemma3:4b").lower()
    except Exception as e:
        logging.error(f"LLM query failed: {e}")
        return "bloat"  # Default to bloat on error

    # Normalise to one of the three labels
    if "contradiction" in raw_response:
        return "contradiction"
    if "overlap" in raw_response:
        return "overlap"
    if "bloat" in raw_response:
        return "bloat"

    # Fallbacks if it only gives first letters or similar
    if raw_response.startswith("c"):
        return "contradiction"
    if raw_response.startswith("o"):
        return "overlap"
    if raw_response.startswith("b"):
        return "bloat"

    logging.warning(f"Model did not answer with a valid label: {raw_response}")
    return "bloat"  # Default to bloat if unclear


def summarize_relations(results):
    """
    Count overlaps, contradictions, and bloat in a list of relation records.

    Args:
        results (list): List of dictionaries, each containing a 'relation' key
                       with value 'overlap', 'contradiction', or 'bloat'

    Returns:
        dict: Summary with keys:
            - 'overlap' (int): Count of overlap relations
            - 'contradiction' (int): Count of contradiction relations
            - 'bloat' (int): Count of bloat relations
            - 'total_metric' (int): overlap + contradiction (excludes bloat)
            - 'total_all' (int): Total pairs including bloat

    Notes:
        Metrics 'overlap' and 'contradiction' EXCLUDE bloat from 'total_metric'
        so you can evaluate only on substantive pairs.
    """

    counts = {"overlap": 0, "contradiction": 0, "bloat": 0}
    for r in results:
        rel = r.get("relation")
        if rel in counts:
            counts[rel] += 1

    # Total number of pairs used for core metrics (exclude bloat)
    counts["total_metric"] = counts["overlap"] + counts["contradiction"]
    # Total pairs including bloat (for sanity check)
    counts["total_all"] = counts["total_metric"] + counts["bloat"]

    return counts


def analyse_clustered_file(category):
    """
    Run analysis on a single clustered file to find overlaps and contradictions.

    Loads a clustered JSON file and compares all pairs of articles within each
    cluster to identify regulatory overlaps, contradictions, or generic bloat.

    Args:
        category (str): Risk category name (e.g., 'compliance_risk', 'credit_risk')
                       Used to construct file path:
                       data/intermediate/clustered/{category}.json

    Returns:
        list: List of dictionaries, each representing a compared article pair:
            {
                'subcategory': str,        # Cluster reference label
                'section_a_id': str,        # Article A ID
                'section_b_id': str,        # Article B ID
                'paragraph_a_index': None,  # Always None (article-level comparison)
                'paragraph_b_index': None,  # Always None (article-level comparison)
                'paragraph_a_text': str,    # Full article A text (all paragraphs)
                'paragraph_b_text': str,    # Full article B text (all paragraphs)
                'relation': str             # 'overlap', 'contradiction', or 'bloat'
            }

    Input JSON format:
        Top level: list of clusters. Each element has:
        {
          "reference_label": "...",
          "articles": [
             {
               "article": {
                  "article id": "...",
                  "article paragraphs": [ "...", ... ]
               },
               "reference_article": true/false
             },
             ...
          ]
        }

    Notes:
        - Compares WHOLE articles (all paragraphs concatenated)
        - Only compares different articles within the same cluster
        - Skips empty articles (no paragraphs)
        - Uses classify_paragraph_pair() for each comparison
    """

    with open(
        Path(__file__).parent.parent.resolve()
        / "data"
        / "intermediate"
        / "clustered"
        / f"{category}.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    all_results = []

    # Calculate total number of comparisons for progress bar
    total_pairs = sum(
        len(list(itertools.combinations(range(len(subcat.get("articles", []))), 2)))
        for subcat in data
    )

    with tqdm(total=total_pairs, desc="Comparing article pairs", unit="pair") as pbar:
        for subcat in data:
            subcat_label = subcat.get("reference_label", "UNKNOWN_SUBCATEGORY")
            articles = subcat.get("articles", [])

            # All pairs of DIFFERENT articles: i != j
            for i, j in itertools.combinations(range(len(articles)), 2):
                art_a = articles[i].get("article", {})
                art_b = articles[j].get("article", {})

                art_a_id = art_a.get("article id", f"article_{i}")
                art_b_id = art_b.get("article id", f"article_{j}")

                paras_a = art_a.get("article paragraphs", []) or []
                paras_b = art_b.get("article paragraphs", []) or []

                # Skip empty articles
                if not paras_a or not paras_b:
                    pbar.update(1)
                    continue

                # Compare full article texts (concatenate all paragraphs)
                text_a = "\n\n".join(paras_a)
                text_b = "\n\n".join(paras_b)

                relation = classify_paragraph_pair(text_a, text_b)

                all_results.append(
                    {
                        "subcategory": subcat_label,
                        "section_a_id": art_a_id,
                        "section_b_id": art_b_id,
                        # Indices are not meaningful at article level; set to None
                        "paragraph_a_index": None,
                        "paragraph_b_index": None,
                        # These now contain the FULL article texts
                        "paragraph_a_text": text_a,
                        "paragraph_b_text": text_b,
                        "relation": relation,  # 'overlap', 'contradiction', or 'bloat'
                    }
                )

                pbar.update(1)

    return all_results


if __name__ == "__main__":
    # Example usage: Find overlap/contradiction/bloat in clustered articles

    # category = "compliance_risk"
    # category = "credit_risk"
    # category = "liquidity_risk"
    # category = "market_risk"
    category = "operational_risk"

    results = analyse_clustered_file(category)
    summary = summarize_relations(results)

    # save all article-level relations
    with open(
        Path(__file__).parent.parent.resolve() / "results" / f"{category}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(len(results), "article pairs analysed (including bloat)")
    print("Overlaps (used in metrics):", summary["overlap"])
    print("Contradictions (used in metrics):", summary["contradiction"])
    print("Bloat (ignored in metrics):", summary["bloat"])
    print("Total used for metrics (overlap + contradiction):", summary["total_metric"])
    print("Total pairs including bloat:", summary["total_all"])
