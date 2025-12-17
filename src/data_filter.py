import json
from pathlib import Path

from helpers import query_llm

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------


def is_credit_relevant(article_paragraphs):
    """
    Determine if article paragraphs are relevant for credit-giving organizations.

    Uses an LLM to classify legislation text as relevant or not relevant for
    organizations that provide credit services.

    Args:
        article_paragraphs: List of strings containing the article text paragraphs

    Returns:
        bool: True if relevant for credit organizations, False otherwise.
              Defaults to False if LLM fails or provides unclear response.
    """

    article_text = "\n".join(article_paragraphs)

    prompt = (
        "answer only YES or NO. "
        "Is this legislation relevant for organizations giving credit: "
        f"{article_text}"
    )

    raw = query_llm(prompt)
    try:
        if not raw:
            # Handle empty response - default to False (not relevant)
            return False

        answer = raw.strip().upper()

        if answer.startswith("YES"):
            return True
        if answer.startswith("NO"):
            return False

        # Handle unexpected response - default to False
        return False
    except Exception as e:
        # Handle LLM errors - default to False and optionally log
        print(f"Warning: LLM query failed: {e}")
        return False


# --------------------------------------------------------------------
# Core function
# --------------------------------------------------------------------


def classify_articles(
    source,
    verbose=False,
):
    """
    Classify parsed articles into credit-relevant and unrelated categories.

    Args:
        source: Source type ("EBA", "fiva_mok", etc.)
        verbose: Whether to print detailed progress information

    Returns:
        Tuple of (relevant_articles, unrelated_articles) where each is a list of article records

    Raises:
        FileNotFoundError: If the input parsed file doesn't exist
        ValueError: If the input JSON file is malformed
    """
    # Load parsed documents
    input_file = (
        Path(__file__).parent.parent.resolve()
        / "data"
        / "intermediate"
        / "parsed"
        / f"all_{source}_parsed.json"
    )

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            documents = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Input file not found: {input_file}. Run data_parse.py first."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in input file {input_file}: {e}")

    total_docs = len(documents)
    relevant_articles = []
    unrelated_articles = []

    print(f"Total documents to process: {total_docs}\n")

    for doc_index, doc in enumerate(documents, start=1):
        doc_title = doc.get("document title", "")
        doc_name = doc.get("document name", "")
        articles = doc.get("articles", [])
        total_articles = len(articles)

        if verbose:
            print(
                f"Processing document {doc_index}/{total_docs}: {doc_name} "
                f"({total_articles} articles)"
            )
        for art_index, article in enumerate(articles, start=1):
            paragraphs = article.get("article paragraphs", [])
            if not paragraphs:
                continue

            # classify with LLM
            is_rel = is_credit_relevant(paragraphs)
            record = {
                "document title": doc_title,
                "document name": doc_name,
                "article id": article.get("article id", ""),
                "article name": article.get("article name", ""),
                "article paragraphs": paragraphs,
            }

            if is_rel:
                relevant_articles.append(record)
            else:
                unrelated_articles.append(record)

            if verbose:
                print(
                    f"  Article {art_index}/{total_articles} classified as {'RELEVANT' if is_rel else 'UNRELATED'}"
                )

        if verbose:
            print("")

    print("Processing completed!")
    print(f"Total relevant: {len(relevant_articles)}")
    print(f"Total unrelated: {len(unrelated_articles)}")

    return relevant_articles, unrelated_articles


# --------------------------------------------------------------------
# Example usage and main execution
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage: Filter out non-credit related articles from specified source
    source = "EBA"
    # source = "fiva_mok"

    # Process
    relevant_articles, unrelated_articles = classify_articles(source, verbose=True)

    # Save to file
    print("Saving results")
    save_dir = (
        Path(__file__).parent.parent.resolve() / "data" / "intermediate" / "filtered"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    relevant_filename = f"credit_related_{source}.json"
    unrelated_filename = f"unrelated_{source}.json"

    with open(save_dir / relevant_filename, "w", encoding="utf-8") as f:
        json.dump(relevant_articles, f, ensure_ascii=False, indent=4)

    with open(save_dir / unrelated_filename, "w", encoding="utf-8") as f:
        json.dump(unrelated_articles, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(relevant_articles)} relevant articles to {relevant_filename}")
    print(f"Saved {len(unrelated_articles)} unrelated articles to {unrelated_filename}")
