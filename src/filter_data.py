import json
import subprocess
from pathlib import Path


intermediate_data_dir = Path(__file__).parent.parent.resolve() / "data" / "intermediate"

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------


def query_llm(prompt: str) -> str:
    """Calls Ollama locally with model gemma3:4b."""
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


def is_credit_relevant(article_paragraphs):
    """Builds prompt and interprets YES/NO model output."""

    article_text = "\n".join(article_paragraphs)

    prompt = (
        "answer only YES or NO. "
        "Is this legislation relevant for organizations giving credit: "
        f"{article_text}"
    )

    raw = query_llm(prompt)
    answer = raw.strip().upper()

    if answer.startswith("YES"):
        return True
    if answer.startswith("NO"):
        return False

    return False


# --------------------------------------------------------------------
# Core function
# --------------------------------------------------------------------


def classify_articles(
    source,
    verbose=False,
):
    # Load parsed documents
    with open(
        Path(__file__).parent.parent.resolve()
        / "data"
        / "intermediate"
        / "parsed"
        / f"all_{source}_parsed.json",
        "r",
        encoding="utf-8",
    ) as f:
        documents = json.load(f)

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
                "article id": article["article id"],
                "article name": article["article name"],
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
    # Example usage: Filter out non-credit related articles from EBA documents
    source = "EBA"
    # source = "fiva_mok"

    # Process
    relevant_articles, unrelated_articles = classify_articles(source)

    # Save to file
    print("Saving results")
    save_dir = (
        Path(__file__).parent.parent.resolve() / "data" / "intermediate" / "filtered"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(
        save_dir / f"credit_related_{source}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            relevant_articles,
            f,
            ensure_ascii=False,
            indent=4,
        )

    with open(
        save_dir / f"unrelated_{source}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            unrelated_articles,
            f,
            ensure_ascii=False,
            indent=4,
        )
