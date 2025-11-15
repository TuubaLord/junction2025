#!/usr/bin/env python3
import json
import re
import requests
import csv
from pathlib import Path

from PyPDF2 import PdfReader  # pip install PyPDF2

# --- CONFIG ---------------------------------------------------------

GUIDELINES_PDF = "junction2025/category_guidelines.pdf"  # <- your PDF with 4-category rules
DOC_JSON = "bofjunction_dataset/gold/output_simple/BRRD/CELEX_32014L0059_EN_TXT.di.json"
OUTPUT_CSV = "paragraph_classification.csv"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
# OLLAMA_MODEL = "deepseek-r1:14b"  # change to your local model name
OLLAMA_MODEL = "gemma3:4b"  # change to your local model name

# Define your 4 categories here
CATEGORIES = [
    "LIQUIDITY_RISK",
    "OPERATIONAL_RISK",
    "MARKET_RISK",
    "CREDIT_RISK",
    "OTHER",  # for paragraphs not related to any of the above
]


# --- HELPERS --------------------------------------------------------


def extract_guidelines_from_pdf(pdf_path: str) -> str:
    """Extract all text from the guidelines PDF."""
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n\n".join(pages_text)

import json
from pathlib import Path

def load_paragraphs_from_json(json_path: str):
    """
    Load a JSON document and return a list of paragraph strings.

    Expected structure (as you described):
        {
          "pages": [
            {
              "paragraphs": [
                ... smallest units to process ...
              ]
            },
            ...
          ]
        }

    Each element in "paragraphs" can be either:
      - a plain string
      - or an object with a "text" field (and maybe other metadata)
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    paragraphs = []

    if isinstance(data, dict) and "pages" in data:
        for page in data["pages"][1:]: # XXX
            # Be tolerant: skip pages that don't have paragraphs
            para_list = page.get("paragraphs", [])
            for para in para_list:
                # Case 1: paragraph is already a string
                if isinstance(para, str):
                    text = para.strip()
                    if text:
                        paragraphs.append(text)

                # Case 2: paragraph is an object with a "text" field
                elif isinstance(para, dict):
                    # Try 'text' first, then fall back to other common keys
                    text = (
                        para.get("text")
                        or para.get("content")
                        or para.get("para")
                    )
                    if isinstance(text, str):
                        text = text.strip()
                        if text:
                            paragraphs.append(text)

                # If there's some other weird structure, ignore it silently
    else:
        # Fallback: if the format is totally different, just dump and don't crash
        print("Warning: JSON structure not as expected (no 'pages' key).")
        text = json.dumps(data, ensure_ascii=False)
        text = text.strip()
        if text:
            paragraphs.append(text)

    return paragraphs

# def classify_paragraph_with_ollama(paragraph: str, guidelines: str = "") -> str:
#     """
#     Classify one paragraph into exactly one of:
#     CREDIT_RISK, LIQUIDITY_RISK, MARKET_RISK, OPERATIONAL_RISK, OTHER

#     Uses a DeepSeek-optimized system prompt and Ollama's /api/generate endpoint.
#     """

#     system_prompt = f"""classify this in to 1 of these categories. output nothing else
#      CATEGORIES = [ "LIQUIDITY_RISK", "OPERATIONAL_RISK", "MARKET_RISK", "CREDIT_RISK", "OTHER"] """
# #     system_prompt = f"""
# # You are a strict text classifier for paragraphs from EU Directive and Regulation documents.

# # You must answer using exactly one category from the following list: [{", ".join(CATEGORIES)}]

# # Categorization rules:
# # - I do not care about paragraphs that are just section headers, preamble/recitals, or other non-requirements. They are all OTHER.
# # - I only care about requirement paragraphs (also known as obligation paragraphs). Classify these into one of the given categories.

# # Formatting rules (follow strictly):
# # - Output only the label.
# # - No explanations.
# # - No punctuation.
# # - No extra words.
# # - If unsure, output OTHER.
# #     """

# #     system_prompt = f"""
# # You are a strict text classifier for paragraphs from EU Directive and Regulation documents.

# # You have ONE task:
# # Given a single paragraph, classify it into exactly ONE of the following categories:

# # 1. CREDIT_RISK  
# # 2. LIQUIDITY_RISK  
# # 3. MARKET_RISK  
# # 4. OPERATIONAL_RISK  
# # 5. OTHER

# # -------------------------
# # CATEGORY DEFINITIONS
# # -------------------------

# # CREDIT_RISK:
# # - About a borrower failing to meet obligations.
# # - Capital adequacy, default risk, counterparty risk, exposures, NPLs, loss absorbency.
# # - Anything related to loans, creditworthiness, capital buffers, resolution tools for losses.

# # LIQUIDITY_RISK:
# # - About inability to meet short-term obligations.
# # - Liquidity shortages, emergency liquidity assistance, funding issues, deposit withdrawals.
# # - Anything related to cash-flow stress or inability to liquidate assets quickly.

# # MARKET_RISK:
# # - About losses from market movements.
# # - Interest rate risk, FX risk, equity/commodity price changes, valuation impacts.
# # - Any references to volatility or price-driven losses.

# # OPERATIONAL_RISK:
# # - About failures of processes, governance, systems, human error, legal disputes.
# # - Includes resolution procedure failures, institutional process issues, insolvency frameworks, administrative shortcomings.

# # OTHER:
# # - If the paragraph is not clearly about any RISK TYPE above.
# # - If the paragraph describes legal frameworks, harmonisation, procedures, mandates, powers, governance without referring to a specific risk mechanism.
# # - If you are NOT 100% certain.

# # -------------------------
# # INSTRUCTIONS
# # -------------------------

# # - Return EXACTLY ONE label from the list.
# # - Use ALL-CAPS, no punctuation, no explanation.  
# # - If the text is mostly about legal structures, harmonisation, procedural rules, EU governance, or resolution frameworks → choose OTHER.
# # - If multiple risks could apply, choose the BEST SINGLE one.
# # - Never output sentences.
# # - Never create new categories.

# # -------------------------
# # EXAMPLES
# # -------------------------

# # Example A (paragraph about insolvency frameworks, legal harmonisation):
# # "Member States apply different insolvency procedures and corporate frameworks... general corporate insolvency procedures may not be appropriate for institutions..."

# # Correct classification: OTHER

# # Example B (paragraph about insufficient liquidity):
# # "During periods of stress institutions may be unable to meet their payment obligations..."

# # Correct classification: LIQUIDITY_RISK

# # Example C (paragraph about credit exposure and counterparty default):
# # "Institutions must evaluate the creditworthiness of their counterparties..."

# # Correct classification: CREDIT_RISK

# # Example D (paragraph about volatility in interest rates and currency prices):
# # "Rapid interest rate movements may cause valuation losses..."

# # Correct classification: MARKET_RISK

# # Example E (paragraph about process failures or administrative shortcomings):
# # "Inadequate internal controls and reporting failures increased losses..."

# # Correct classification: OPERATIONAL_RISK

# # -------------------------
# # OUTPUT FORMAT (MANDATORY)
# # -------------------------

# # Return ONLY one of:
# # CREDIT_RISK
# # LIQUIDITY_RISK
# # MARKET_RISK
# # OPERATIONAL_RISK
# # OTHER
# #     """.strip()



#     user_prompt = f"""
# Now classify:
# Input: "{paragraph}"
# Output:
# """
# # user_prompt = f'Paragraph:\n"""{paragraph}"""\n\nWhat is the single best category?'

#     full_prompt = system_prompt + "\n\n" + user_prompt

#     url = f"{OLLAMA_BASE_URL}/api/generate"
#     payload = {
#         "model": OLLAMA_MODEL,   # e.g. "deepseek-1:4b" in your Ollama setup
#         "prompt": full_prompt,
#         "stream": False,
#     }

#     resp = requests.post(url, json=payload)
#     resp.raise_for_status()
#     data = resp.json()

#     raw = data.get("response", "") or ""
#     content = raw.strip()

#     # --- Post-process to enforce a single clean label -----------------
#     normalized = content.upper().strip()

#     # 1) Exact match
#     for cat in CATEGORIES:
#         if normalized == cat:
#             return cat

#     # 2) First token (e.g. "CREDIT_RISK because ...")
#     first_token = re.split(r"\s+|\n+", normalized)[0]
#     for cat in CATEGORIES:
#         if first_token == cat:
#             return cat

#     # 3) Category mentioned anywhere
#     for cat in CATEGORIES:
#         if re.search(rf"\b{re.escape(cat)}\b", normalized):
#             return cat

#     # 4) If it really went off the rails, treat as OTHER
#     return "OTHER"
def classify_paragraph_with_ollama(paragraph: str, guidelines: str = "") -> str:
    """
    Classify paragraph into REQUIREMENT or NON_REQUIREMENT.
    Optimized for small 4B models (Gemma).
    """

    system_prompt = """
Classify the paragraph as one label:

REQUIREMENT = contains an obligation (e.g., "shall", "must", "should ensure", "are required to").
NON_REQUIREMENT = background, explanation, recitals, context, headings, or anything not imposing an action.

Return ONLY one label: REQUIREMENT or NON_REQUIREMENT.
    """.strip()

    user_prompt = f'Paragraph:\n"""{paragraph}"""\nLabel?'

    full_prompt = system_prompt + "\n\n" + user_prompt

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
    }

    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()

    raw = data.get("response", "") or ""
    text = raw.strip().upper()

    # Clean output
    if text.startswith("REQUIREMENT"):
        return "REQUIREMENT"
    if text.startswith("NON_REQUIREMENT"):
        return "NON_REQUIREMENT"

    # Fallback: small models guess "requirement" too often → be conservative
    return "NON_REQUIREMENT"
# --- MAIN SCRIPT ----------------------------------------------------


def main():
    print("Extracting guidelines from PDF...")
    guidelines_text = extract_guidelines_from_pdf(GUIDELINES_PDF)

    print("Loading paragraphs from JSON...")
    paragraphs = load_paragraphs_from_json(DOC_JSON)
    print(f"Found {len(paragraphs)} paragraphs.")

    rows = []
    for idx, para in enumerate(paragraphs, start=1):
        print(f"Classifying paragraph {idx}/{len(paragraphs)}...")
        category = classify_paragraph_with_ollama(para, guidelines_text)
        print(idx, category, para)
        rows.append({
            "paragraph_index": idx,
            "category": category,
            "paragraph": para,
        })

    print(f"Saving results to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["paragraph_index", "category", "paragraph"])
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    main()
