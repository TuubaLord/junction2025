import json
import subprocess
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------
# 1. Query Ollama locally (model: gemma3:4b)
# ---------------------------------------------------------

def query_llm(prompt: str) -> str:
    """
    Calls Ollama locally with model gemma3:4b.
    Returns the raw model output as a string.
    """
    process = subprocess.Popen(
        ["ollama", "run", "gemma3:4b"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = process.communicate(prompt)
    if err:
        print("LLM error:", err)
    return out.strip()


# ---------------------------------------------------------
# 2. Map LLM output to one of the 5 categories
# ---------------------------------------------------------

VALID_CATEGORIES = {
    "CREDIT_RISK",
    "LIQUIDITY_RISK",
    "MARKET_RISK",
    "OPERATIONAL_RISK",
    "COMPLIANCE_RISK",
}


def classify_article_category(article_paragraphs: List[str]) -> Optional[str]:
    """
    Ask the LLM which of the 5 risk categories this *article* belongs to.
    Uses all paragraphs joined as the article text.
    Returns the category name (e.g. 'CREDIT_RISK') or None if it can't be parsed.
    """
    article_text = "\n".join(article_paragraphs)

    prompt = (
        "Which of the following categories is this legistlation related to. "
        "answer only the name of the category\n\n"
        "1. CREDIT_RISK — Risk of financial loss arising when borrowers or "
        "counterparties fail to meet their contractual obligations.\n\n"
        "2. LIQUIDITY_RISK — Risk that the institution cannot meet cash or "
        "collateral demands without incurring unacceptable costs or losses.\n\n"
        "3. MARKET_RISK — Risk of loss from adverse movements in market "
        "variables such as interest rates, currencies, or credit spreads.\n\n"
        "4. OPERATIONAL_RISK — Risk of loss resulting from failures in internal "
        "processes, people, systems, or from external disruptions or cyber events.\n\n"
        "5. COMPLIANCE_RISK — Risk of legal, regulatory, or conduct breaches "
        "leading to penalties, restrictions, or reputational harm.\n\n"
        f"{article_text}"
    )

    raw = query_llm(prompt)
    ans = raw.strip().upper()

    # Take the first token and strip punctuation
    first_token = ans.split()[0].strip(" .,:;")

    if first_token in VALID_CATEGORIES:
        return first_token

    # Sometimes the model might answer like "1. CREDIT_RISK"
    # so we also look for any valid category substring
    for cat in VALID_CATEGORIES:
        if cat in ans:
            return cat

    # If we really can't map it, return None so we can count it
    return None


# ---------------------------------------------------------
# 3. Main splitting logic (article-level classification)
# ---------------------------------------------------------

def split_credit_related_by_risk(
    input_json: str = "credit_related_eba.json",
    out_credit: str = "credit_risk_eba.json",
    out_liquidity: str = "liquidity_risk_eba.json",
    out_market: str = "market_risk_eba.json",
    out_operational: str = "operational_risk_eba.json",
    out_compliance: str = "compliance_risk_eba.json",
) -> None:
    # Load the credit-related articles (from previous step)
    with open(input_json, "r", encoding="utf-8") as f:
        articles: List[Dict[str, Any]] = json.load(f)

    credit_risk: List[Dict[str, Any]] = []
    liquidity_risk: List[Dict[str, Any]] = []
    market_risk: List[Dict[str, Any]] = []
    operational_risk: List[Dict[str, Any]] = []
    compliance_risk: List[Dict[str, Any]] = []

    unclassified_count = 0
    total_articles = len(articles)

    print(f"Loaded {total_articles} credit-related articles\n")

    for art_index, art in enumerate(articles, start=1):
        doc_title = art.get("document title", "")
        doc_name = art.get("document name", "")
        article_id = art.get("article id")
        article_name = art.get("article name")
        paragraphs = art.get("article paragraphs", [])

        print(f"Processing article {art_index}/{total_articles} "
              f"(id={article_id}, {len(paragraphs)} paragraphs)")

        if not paragraphs:
            print(f"  [WARN] Article {article_id} has no paragraphs, skipping")
            unclassified_count += 1
            continue

        category = classify_article_category(paragraphs)

        # Always print classification result
        if category is not None:
            print(
                f"  Article {article_id} classified as {category}"
            )
        else:
            print(
                f"  Article {article_id} classified as UNCLASSIFIED"
            )

        # The record we store is still per-article, with all paragraphs included
        record = {
            "document title": doc_title,
            "document name": doc_name,
            "article id": article_id,
            "article name": article_name,
            "article paragraphs": paragraphs,
        }

        if category == "CREDIT_RISK":
            credit_risk.append(record)
        elif category == "LIQUIDITY_RISK":
            liquidity_risk.append(record)
        elif category == "MARKET_RISK":
            market_risk.append(record)
        elif category == "OPERATIONAL_RISK":
            operational_risk.append(record)
        elif category == "COMPLIANCE_RISK":
            compliance_risk.append(record)
        else:
            unclassified_count += 1

    # Save outputs
    with open(out_credit, "w", encoding="utf-8") as f:
        json.dump(credit_risk, f, ensure_ascii=False, indent=4)

    with open(out_liquidity, "w", encoding="utf-8") as f:
        json.dump(liquidity_risk, f, ensure_ascii=False, indent=4)

    with open(out_market, "w", encoding="utf-8") as f:
        json.dump(market_risk, f, ensure_ascii=False, indent=4)

    with open(out_operational, "w", encoding="utf-8") as f:
        json.dump(operational_risk, f, ensure_ascii=False, indent=4)

    with open(out_compliance, "w", encoding="utf-8") as f:
        json.dump(compliance_risk, f, ensure_ascii=False, indent=4)

    print("\nDone.")
    print(f"Total articles processed: {total_articles}")
    print(f"CREDIT_RISK articles: {len(credit_risk)}")
    print(f"LIQUIDITY_RISK articles: {len(liquidity_risk)}")
    print(f"MARKET_RISK articles: {len(market_risk)}")
    print(f"OPERATIONAL_RISK articles: {len(operational_risk)}")
    print(f"COMPLIANCE_RISK articles: {len(compliance_risk)}")
    print(f"Unclassified articles: {unclassified_count}")


if __name__ == "__main__":
    split_credit_related_by_risk()
