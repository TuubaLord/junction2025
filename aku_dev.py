import json
import itertools
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

CONTRADICTION_INSTRUCTIONS = """
You are analysing two regulatory paragraphs.

A contradiction between legal paragraphs occurs ONLY when:
- the obligations, permissions, or prohibitions they impose **cannot be complied with at the same time**, AND
- one requirement **directly conflicts with or negates** the other.

Important clarifications:
- Different wording, phrasing, or level of detail does **not** imply contradiction.
- Two paragraphs that can both be followed at the same time are **not** a contradiction.
- One paragraph being broader, narrower, or more general than another is **not** a contradiction unless they directly conflict.
- If there is any way to comply with both paragraphs simultaneously, the answer must be **no**.

Question: Do Paragraph A and Paragraph B contradict each other according to this definition?

Answer STRICTLY with "yes" or "no" in lowercase.

Do NOT output anything else before or after the word.
"""


def call_ollama(prompt: str, model: str = MODEL_NAME, temperature: float = 0.1) -> str:
    """
    Call Ollama with the given prompt and return a sanitized 'yes' or 'no'.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    raw = data["response"].strip().lower()

    # Sanitize to 'yes' / 'no'
    if "yes" in raw:
        return "yes"
    if "no" in raw:
        return "no"

    # Fallback for slightly off outputs
    if raw.startswith("y"):
        return "yes"
    if raw.startswith("n"):
        return "no"

    raise ValueError(f"Model did not answer with yes/no: {raw}")


def classify_paragraph_pair(p_a: str, p_b: str) -> str:
    """
    Return one of: "overlap" or "contradiction".

    Logic:
    - If model says 'yes' to contradiction → 'contradiction'
    - If model says 'no' → 'overlap'
    """
    prompt = f"""{CONTRADICTION_INSTRUCTIONS}

Paragraph A:
\"\"\"{p_a}\"\"\"

Paragraph B:
\"\"\"{p_b}\"\"\"
"""
    answer = call_ollama(prompt)

    if answer == "yes":
        return "contradiction"
    else:
        return "overlap"


def analyse_subcategory_pairs(subcat_name: str, paragraphs: list) -> list:
    """
    paragraphs: list of {"id": ..., "text": ...}
    returns: list of dicts:
      {
        "subcategory": ...,
        "id_a": ...,
        "id_b": ...,
        "relation": "overlap" | "contradiction"
      }
    """
    results = []

    # all unordered pairs: combinations of 2
    for p_a, p_b in itertools.combinations(paragraphs, 2):
        relation = classify_paragraph_pair(p_a["text"], p_b["text"])
        results.append({
            "subcategory": subcat_name,
            "id_a": p_a["id"],
            "id_b": p_b["id"],
            "relation": relation
        })
    return results


def analyse_all_subcategories(json_path: str, output_path: str = "relations_output.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = []

    for subcat_name, paragraphs in data.items():
        subcat_results = analyse_subcategory_pairs(subcat_name, paragraphs)
        all_results.extend(subcat_results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


def summarize_relations(results: list) -> dict:
    """
    Count how many overlaps and contradictions exist in the results.

    results: list of dicts with key "relation" == "overlap" or "contradiction"

    Returns:
        {
            "overlap": <int>,
            "contradiction": <int>,
            "total": <int>
        }
    """
    counts = {"overlap": 0, "contradiction": 0}
    for r in results:
        rel = r.get("relation")
        if rel in counts:
            counts[rel] += 1

    counts["total"] = counts["overlap"] + counts["contradiction"]
    return counts


if __name__ == "__main__":
    results = analyse_all_subcategories("aku_dev.json")
    summary = summarize_relations(results)
    print("Overlaps:", summary["overlap"])
    print("Contradictions:", summary["contradiction"])
    print("Total pairs:", summary["total"])


# import json
# import itertools
# import requests

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "gemma3:4b"

# CLASSIFICATION_INSTRUCTIONS = """
# You are analysing two regulatory paragraphs.

# Decide the relationship of Paragraph A and Paragraph B, using ONLY these labels:

# - "overlap": They express substantially the same requirement, obligation or restriction, or one is a narrower/clear restatement of the other.
# - "contradiction": They cannot both be true or valid at the same time in the same context (e.g. one permits what the other prohibits, or they impose incompatible requirements).
# - "neither": They address different topics, different conditions, or are compatible but non-overlapping (e.g. complementary requirements).

# Return a single JSON object with this exact structure:
# {
#   "relation": "overlap" | "contradiction" | "neither",
#   "explanation": "one or two sentences explaining your choice"
# }

# Do NOT add any other keys or text outside the JSON.
# """


# def call_ollama(prompt: str, model: str = MODEL_NAME, temperature: float = 0.1) -> str:
#     """
#     Call Ollama with the given prompt and return the model's raw text response.
#     Requires Ollama to be running locally.
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
#     return data["response"]


# def classify_paragraph_pair(p_a: str, p_b: str) -> dict:
#     prompt = f"""{CLASSIFICATION_INSTRUCTIONS}

# Paragraph A:
# \"\"\"{p_a}\"\"\"

# Paragraph B:
# \"\"\"{p_b}\"\"\"
# """
#     raw = call_ollama(prompt)
#     # Try to parse JSON directly; if the model adds junk, strip around braces
#     try:
#         # Fast path: direct JSON
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         # fallback: extract the first {...} block
#         start = raw.find("{")
#         end = raw.rfind("}")
#         if start == -1 or end == -1 or end <= start:
#             raise ValueError(f"Model output is not valid JSON:\n{raw}")
#         cleaned = raw[start:end+1]
#         return json.loads(cleaned)


# def analyse_subcategory_pairs(subcat_name: str, paragraphs: list) -> list:
#     """
#     paragraphs: list of {"id": ..., "text": ...}
#     returns: list of dicts:
#       {
#         "subcategory": ...,
#         "id_a": ...,
#         "id_b": ...,
#         "relation": "overlap" | "contradiction" | "neither",
#         "explanation": ...
#       }
#     """
#     results = []

#     # all unordered pairs: combinations of 2
#     for p_a, p_b in itertools.combinations(paragraphs, 2):
#         res = classify_paragraph_pair(p_a["text"], p_b["text"])
#         results.append({
#             "subcategory": subcat_name,
#             "id_a": p_a["id"],
#             "id_b": p_b["id"],
#             "relation": res.get("relation", "").lower(),
#             "explanation": res.get("explanation", "")
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


# if __name__ == "__main__":

#     results = analyse_all_subcategories("aku_dev.json")
#     print(len(results), "pairs analysed")
