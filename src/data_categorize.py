import json
from pathlib import Path

from helpers import query_llm

# --------------------------------------------------------------------
# Constants and configurations for different sources
# --------------------------------------------------------------------

VALID_CATEGORIES = {
    "CREDIT_RISK",
    "LIQUIDITY_RISK",
    "MARKET_RISK",
    "OPERATIONAL_RISK",
    "COMPLIANCE_RISK",
}

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------


def classify_article_category(article_paragraphs):
    """
    Classify article paragraphs into one of five risk categories using LLM.

    Uses an LLM to analyze legislation text and determine which of the five
    banking risk categories (CREDIT_RISK, LIQUIDITY_RISK, MARKET_RISK,
    OPERATIONAL_RISK, COMPLIANCE_RISK) the content relates to most closely.

    Args:
        article_paragraphs: List of strings containing the article text paragraphs

    Returns:
        str or None: The risk category name (e.g. 'CREDIT_RISK') if classification
                    is successful, None if the response cannot be parsed or LLM fails
    """

    article_text = "\n".join(article_paragraphs)

    prompt = (
        "Which of the following categories is this legislation related to. "
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

    try:
        raw = query_llm(prompt)
        if not raw:
            # Handle empty response
            return None

        ans = raw.strip().upper()

        # Take the first token and strip punctuation
        tokens = ans.split()
        if not tokens:
            # Handle empty response after processing
            return None
    except Exception as e:
        # Handle LLM errors - return None to count as unclassified
        print(f"Warning: LLM query failed for article classification: {e}")
        return None

    first_token = tokens[0].strip(" .,:;")

    if first_token in VALID_CATEGORIES:
        return first_token

    # Sometimes the model might answer like "1. CREDIT_RISK"
    # so we also look for any valid category substring
    for cat in VALID_CATEGORIES:
        if cat in ans:
            return cat

    # If we really can't map it, return None so we can count it
    return None


# --------------------------------------------------------------------
# Core function
# --------------------------------------------------------------------


def categorize_articles(source, verbose=False):
    """
    Categorize credit-related articles into banking risk categories using LLM classification.

    Loads pre-filtered credit-related articles and uses an LLM to classify each article
    into one of five banking risk categories: CREDIT_RISK, LIQUIDITY_RISK, MARKET_RISK,
    OPERATIONAL_RISK, or COMPLIANCE_RISK. Articles that cannot be classified are counted
    as unclassified.

    Args:
        source (str): Source type ("EBA", "fiva_mok", etc.) - used to locate input file
        verbose (bool): Whether to print detailed progress information during processing

    Returns:
        Tuple[Dict[str, List], int]: A tuple containing:
            - categorized_articles_dict: Dictionary mapping category names to lists of article records
            - unclassified_count: Number of articles that could not be classified

    Note:
        Expects input file at: data/intermediate/filtered/credit_related_{source}.json
        Articles without paragraphs are automatically marked as unclassified.

    Raises:
        FileNotFoundError: If the input filtered file doesn't exist
        ValueError: If the input JSON file is malformed
    """

    # Load the credit-related articles (from previous step)
    input_file = (
        Path(__file__).parent.parent.resolve()
        / "data"
        / "intermediate"
        / "filtered"
        / f"credit_related_{source}.json"
    )

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            articles = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Input file not found: {input_file}. Run data_filter.py first."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in input file {input_file}: {e}")

    credit_risk = []
    liquidity_risk = []
    market_risk = []
    operational_risk = []
    compliance_risk = []

    unclassified_count = 0
    total_articles = len(articles)

    print(f"Loaded {total_articles} credit-related articles from {source}\n")

    for art_index, art in enumerate(articles, start=1):
        doc_title = art.get("document title", "")
        doc_name = art.get("document name", "")
        article_id = art.get("article id")
        article_name = art.get("article name")
        paragraphs = art.get("article paragraphs", [])

        if verbose:
            print(
                f"Processing article {art_index}/{total_articles} "
                f"(id={article_id}, {len(paragraphs)} paragraphs)"
            )

        if not paragraphs:
            if verbose:
                print(f"  [WARN] Article {article_id} has no paragraphs, skipping")
            unclassified_count += 1
            continue

        category = classify_article_category(paragraphs)

        # Print classification result if verbose
        if verbose:
            if category is not None:
                print(f"  Article {article_id} classified as {category}")
            else:
                print(f"  Article {article_id} classified as UNCLASSIFIED")

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

    # Create results dictionary
    categorized_articles = {
        "CREDIT_RISK": credit_risk,
        "LIQUIDITY_RISK": liquidity_risk,
        "MARKET_RISK": market_risk,
        "OPERATIONAL_RISK": operational_risk,
        "COMPLIANCE_RISK": compliance_risk,
    }

    print("\nProcessing completed!")
    print(f"Total articles processed: {total_articles}")
    print(f"CREDIT_RISK articles: {len(credit_risk)}")
    print(f"LIQUIDITY_RISK articles: {len(liquidity_risk)}")
    print(f"MARKET_RISK articles: {len(market_risk)}")
    print(f"OPERATIONAL_RISK articles: {len(operational_risk)}")
    print(f"COMPLIANCE_RISK articles: {len(compliance_risk)}")
    print(f"Unclassified articles: {unclassified_count}")

    return categorized_articles, unclassified_count


# --------------------------------------------------------------------
# Example usage and main execution
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage: Categorize credit-related articles from EBA documents
    # source = "EBA"
    source = "FIVA_MOK"

    # Process
    categorized_articles, unclassified_count = categorize_articles(source, verbose=True)

    # Save to files
    print("Saving results")
    save_dir = (
        Path(__file__).parent.parent.resolve() / "data" / "intermediate" / "categorized"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save each category to separate files
    for category, articles in categorized_articles.items():
        category_filename = f"{category.lower()}_{source}.json"
        with open(save_dir / category_filename, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(articles)} {category} articles to {category_filename}")
