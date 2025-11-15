
import json
import itertools
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

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


def call_ollama(prompt: str, model: str = MODEL_NAME, temperature: float = 0.1) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    raw = resp.json()["response"].strip().lower()

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
    return call_ollama(prompt)


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
        print("start new subcat")
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
        "clustered_data/compliance_risk_clustered_old.json"
    )
    summary = summarize_relations(results)

    print(len(results), "article pairs analysed (including bloat)")
    print("Overlaps (used in metrics):", summary["overlap"])
    print("Contradictions (used in metrics):", summary["contradiction"])
    print("Bloat (ignored in metrics):", summary["bloat"])
    print("Total used for metrics (overlap + contradiction):",
          summary["total_metric"])
    print("Total pairs including bloat:", summary["total_all"])


# import json
# import itertools
# import requests

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "gemma3:4b"

# CLASSIFICATION_INSTRUCTIONS = """
# Compare Paragraph A and Paragraph B. Output one label: contradiction, overlap, or bloat.

# contradiction
# Use only if:

# Their obligations/permissions/prohibitions cannot be followed at the same time, and

# One directly negates or forbids what the other requires.
# If simultaneous compliance is possible, it is not a contradiction.

# overlap
# Use only if:

# Both address the same specific regulatory topic/process/requirement, and

# They are not a contradiction.

# bloat
# Use when:

# Any similarity is generic (e.g., “risk”, “capital”, “liquidity”), and

# They do not regulate the same concrete behaviour/process.

# Decision order:

# contradiction

# overlap

# bloat

# Output: one lowercase word, nothing else.
# """


# def call_ollama(prompt: str, model: str = MODEL_NAME, temperature: float = 0.1) -> str:
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "stream": False,
#         "options": {"temperature": temperature},
#     }
#     resp = requests.post(OLLAMA_URL, json=payload)
#     resp.raise_for_status()
#     raw = resp.json()["response"].strip().lower()

#     # Normalise to one of the three labels
#     if "contradiction" in raw:
#         return "contradiction"
#     if "overlap" in raw:
#         return "overlap"
#     if "bloat" in raw:
#         return "bloat"

#     # Fallbacks if it only gives first letters or similar
#     if raw.startswith("c"):
#         return "contradiction"
#     if raw.startswith("o"):
#         return "overlap"
#     if raw.startswith("b"):
#         return "bloat"

#     raise ValueError(f"Model did not answer with a valid label: {raw}")


# def classify_paragraph_pair(p_a: str, p_b: str) -> str:
#     """
#     Return one of: 'contradiction', 'overlap', 'bloat'.
#     """
#     prompt = f"""{CLASSIFICATION_INSTRUCTIONS}

# Paragraph A:
# \"\"\"{p_a}\"\"\"

# Paragraph B:
# \"\"\"{p_b}\"\"\"
# """
#     return call_ollama(prompt)


# def summarize_relations(results: list) -> dict:
#     """
#     Count overlaps, contradictions, and bloat in a list of relation records.

#     Metrics 'overlap' and 'contradiction' EXCLUDE bloat from the 'total_metric'
#     so you can evaluate only on substantive pairs.
#     """
#     counts = {"overlap": 0, "contradiction": 0, "bloat": 0}
#     for r in results:
#         rel = r.get("relation")
#         if rel in counts:
#             counts[rel] += 1

#     # Total number of pairs used for core metrics (exclude bloat)
#     counts["total_metric"] = counts["overlap"] + counts["contradiction"]
#     # Total pairs including bloat (for sanity check)
#     counts["total_all"] = counts["total_metric"] + counts["bloat"]
#     return counts


# def analyse_clustered_file(
#     json_path: str,
#     output_path: str = "relations_output.json"
# ) -> list:
#     """
#     Run analysis on a single clustered file with structure like
#     `compliance_risk_clustered.json`.

#     Assumptions about format:
#     - Top level: a list of subcategories (clusters).
#       Each element has:
#         {
#           "reference_label": "...",
#           "articles": [
#              {
#                "article": {
#                   "article id": "...",
#                   "article paragraphs": [ "...", ... ]
#                },
#                "reference_article": true/false
#              },
#              ...
#           ]
#         }

#     - We treat each top-level element as a subcategory.
#     - Within each subcategory, each entry in "articles" is a section.
#     - We compare paragraphs ONLY between different sections within the same subcategory.
#     """
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     all_results = []

#     for subcat in data:
#         subcat_label = subcat.get("reference_label", "UNKNOWN_SUBCATEGORY")
#         sections = subcat.get("articles", [])

#         # All pairs of DIFFERENT sections: i != j
#         for i, j in itertools.combinations(range(len(sections)), 2):
#             sec_a = sections[i].get("article", {})
#             sec_b = sections[j].get("article", {})

#             sec_a_id = sec_a.get("article id", f"section_{i}")
#             sec_b_id = sec_b.get("article id", f"section_{j}")

#             paras_a = sec_a.get("article paragraphs", []) or []
#             paras_b = sec_b.get("article paragraphs", []) or []

#             # All paragraph pairs between these two sections
#             for idx_a, p_a in enumerate(paras_a):
#                 for idx_b, p_b in enumerate(paras_b):
#                     relation = classify_paragraph_pair(p_a, p_b)
#                     all_results.append(
#                         {
#                             "subcategory": subcat_label,
#                             "section_a_id": sec_a_id,
#                             "section_b_id": sec_b_id,
#                             "paragraph_a_index": idx_a,
#                             "paragraph_b_index": idx_b,
#                             "paragraph_a_text": p_a,
#                             "paragraph_b_text": p_b,
#                             "relation": relation,  # 'overlap', 'contradiction', or 'bloat'
#                         }
#                     )

#     # save all pairwise relations
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(all_results, f, indent=2, ensure_ascii=False)

#     return all_results


# if __name__ == "__main__":
#     # Example usage with your file:
#     results = analyse_clustered_file(
#         "clustered_data/operational_risk_clustered.json"
#     )
#     summary = summarize_relations(results)

#     print(len(results), "paragraph pairs analysed (including bloat)")
#     print("Overlaps (used in metrics):", summary["overlap"])
#     print("Contradictions (used in metrics):", summary["contradiction"])
#     print("Bloat (ignored in metrics):", summary["bloat"])
#     print("Total used for metrics (overlap + contradiction):",
#           summary["total_metric"])
#     print("Total pairs including bloat:", summary["total_all"])


# import json
# import itertools
# import requests

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "gemma3:4b"

# CONTRADICTION_INSTRUCTIONS = """
# You are analysing two regulatory paragraphs.

# A contradiction between legal paragraphs occurs ONLY when:
# - the obligations, permissions, or prohibitions they impose **cannot be complied with at the same time**, AND
# - one requirement **directly conflicts with or negates** the other.

# Important clarifications:
# - Different wording, phrasing, or level of detail does **not** imply contradiction.
# - Two paragraphs that can both be followed at the same time are **not** a contradiction.
# - One paragraph being broader, narrower, or more general than another is **not** a contradiction unless they directly conflict.
# - If there is any way to comply with both paragraphs simultaneously, the answer must be **no**.

# Question: Do Paragraph A and Paragraph B contradict each other according to this definition?

# Answer STRICTLY with "yes" or "no" in lowercase.

# Do NOT output anything else before or after the word.
# """


# def call_ollama(prompt: str, model: str = MODEL_NAME, temperature: float = 0.1) -> str:
#     """
#     Call Ollama with the given prompt and return a sanitized 'yes' or 'no'.
#     """
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "stream": False,
#         "options": {
#             "temperature": temperature
#         }
#     }
#     resp = requests.post(OLLAMA_URL, json=payload)
#     resp.raise_for_status()
#     data = resp.json()
#     raw = data["response"].strip().lower()

#     # Sanitize to 'yes' / 'no'
#     if "yes" in raw:
#         return "yes"
#     if "no" in raw:
#         return "no"

#     # Fallback for slightly off outputs
#     if raw.startswith("y"):
#         return "yes"
#     if raw.startswith("n"):
#         return "no"

#     raise ValueError(f"Model did not answer with yes/no: {raw}")


# def classify_paragraph_pair(p_a: str, p_b: str) -> str:
#     """
#     Return one of: "overlap" or "contradiction".

#     Logic:
#     - If model says 'yes' to contradiction → 'contradiction'
#     - If model says 'no' → 'overlap'
#     """
#     prompt = f"""{CONTRADICTION_INSTRUCTIONS}

# Paragraph A:
# \"\"\"{p_a}\"\"\"

# Paragraph B:
# \"\"\"{p_b}\"\"\"
# """
#     answer = call_ollama(prompt)

#     if answer == "yes":
#         return "contradiction"
#     else:
#         return "overlap"


# def analyse_subcategory_pairs(subcat_name: str, paragraphs: list) -> list:
#     """
#     paragraphs: list of {"id": ..., "text": ...}
#     returns: list of dicts:
#       {
#         "subcategory": ...,
#         "id_a": ...,
#         "id_b": ...,
#         "relation": "overlap" | "contradiction"
#       }
#     """
#     results = []

#     # all unordered pairs: combinations of 2
#     for p_a, p_b in itertools.combinations(paragraphs, 2):
#         relation = classify_paragraph_pair(p_a["text"], p_b["text"])
#         results.append({
#             "subcategory": subcat_name,
#             "id_a": p_a["id"],
#             "id_b": p_b["id"],
#             "relation": relation
#         })
#     return results


# def analyse_all_subcategories(json_path: str, output_path: str = "relations_output.json"):
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     all_results = []

#     for subcat_name, paragraphs in data.items():
#         subcat_results = analyse_subcategory_pairs(subcat_name, paragraphs)
#         all_results.extend(subcat_results)

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(all_results, f, indent=2, ensure_ascii=False)

#     return all_results


# def summarize_relations(results: list) -> dict:
#     """
#     Count how many overlaps and contradictions exist in the results.

#     results: list of dicts with key "relation" == "overlap" or "contradiction"

#     Returns:
#         {
#             "overlap": <int>,
#             "contradiction": <int>,
#             "total": <int>
#         }
#     """
#     counts = {"overlap": 0, "contradiction": 0}
#     for r in results:
#         rel = r.get("relation")
#         if rel in counts:
#             counts[rel] += 1

#     counts["total"] = counts["overlap"] + counts["contradiction"]
#     return counts


# if __name__ == "__main__":
#     results = analyse_all_subcategories("aku_dev.json")
#     summary = summarize_relations(results)
#     print("Overlaps:", summary["overlap"])
#     print("Contradictions:", summary["contradiction"])
#     print("Total pairs:", summary["total"])
