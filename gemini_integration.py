import os
import json
import itertools
from google import genai

# Gemini API settings
GEMINI_MODEL = "gemini-2.5-flash"  # or another Gemini model name

# Expect the API key in the environment for safety
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Create an API key in Google AI Studio and export GEMINI_API_KEY."
    )

# Create a single client to reuse
client = genai.Client(api_key=GEMINI_API_KEY)

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


def call_gemini(prompt: str, temperature: float = 0.1) -> str:
    """
    Call the Gemini SDK and normalise the output
    to one of: 'contradiction', 'overlap', 'bloat'.
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": temperature,
        },
    )

    # response.text is the concatenated text output
    raw = (response.text or "").strip().lower()

    # Normalise to one of the three labels
    if "contradiction" in raw:
        return "contradiction"
    if "overlap" in raw:
        return "overlap"
    if "bloat" in raw:
        return "bloat"

    # Fallbacks if it only gives first letters or similar
    if raw.startswith("c"):
        return "contradiction"
    if raw.startswith("o"):
        return "overlap"
    if raw.startswith("b"):
        return "bloat"

    raise ValueError(f"Model did not answer with a valid label: {raw}")


def classify_paragraph_pair(p_a: str, p_b: str) -> str:
    """
    Return one of: 'contradiction', 'overlap', 'bloat'.
    """
    prompt = f"""{CLASSIFICATION_INSTRUCTIONS}

Paragraph A:
\"\"\"{p_a}\"\"\"

Paragraph B:
\"\"\"{p_b}\"\"\"
"""
    return call_gemini(prompt)


def summarize_relations(results: list) -> dict:
    """
    Count overlaps, contradictions, and bloat in a list of relation records.

    Metrics 'overlap' and 'contradiction' EXCLUDE bloat from the 'total_metric'
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


def analyse_clustered_file(
    json_path: str,
    output_path: str = "relations_output.json"
) -> list:
    """
    Run analysis on a single clustered file with structure like
    `compliance_risk_clustered.json`.

    Assumptions about format:
    - Top level: a list of subcategories (clusters).
      Each element has:
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

    - We treat each top-level element as a subcategory.
    - Within each subcategory, each entry in "articles" is an article.
    - We compare WHOLE articles (all paragraphs concatenated) ONLY
      between different articles within the same subcategory.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = []

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

    # save all article-level relations
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


if __name__ == "__main__":
    # Example usage with your file:
    results = analyse_clustered_file(
        "clustered_data/market_risk_clustered.json"
    )
    summary = summarize_relations(results)

    print(len(results), "article pairs analysed (including bloat)")
    print("Overlaps (used in metrics):", summary["overlap"])
    print("Contradictions (used in metrics):", summary["contradiction"])
    print("Bloat (ignored in metrics):", summary["bloat"])
    print("Total used for metrics (overlap + contradiction):",
          summary["total_metric"])
    print("Total pairs including bloat:", summary["total_all"])
