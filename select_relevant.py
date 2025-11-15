import json
import subprocess
from typing import List, Dict, Any


# ---------------------------------------------------------
# 1. Query Ollama locally (model: qwen3:4b)
# ---------------------------------------------------------

def query_llm(prompt: str) -> str:
    """Calls Ollama locally with model gemma3:4b."""
    process = subprocess.Popen(
        ["ollama", "run", "gemma3:4b"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = process.communicate(prompt)
    if err:
        print("LLM error:", err)
    return out.strip()


# ---------------------------------------------------------
# 2. Classification helper
# ---------------------------------------------------------

def is_credit_relevant(article_paragraphs: List[str]) -> bool:
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


# ---------------------------------------------------------
# 3. Main classification function with PROGRESS PRINTING
# ---------------------------------------------------------

def classify_articles(
    input_json: str = "eba_parsed.json",
    output_relevant: str = "credit_related_eba.json",
    output_unrelated: str = "unrelated_eba.json",
) -> None:

    # Load parsed documents
    with open(input_json, "r", encoding="utf-8") as f:
        documents: List[Dict[str, Any]] = json.load(f)

    total_docs = len(documents)
    relevant_articles = []
    unrelated_articles = []

    print(f"Total documents to process: {total_docs}\n")

    for doc_index, doc in enumerate(documents, start=1):
        doc_title = doc.get("document title", "")
        doc_name = doc.get("document name", "")
        articles = doc.get("articles", [])
        total_articles = len(articles)

        print(f"Processing document {doc_index}/{total_docs}: {doc_name} "
              f"({total_articles} articles)")

        for art_index, article in enumerate(articles, start=1):
            paragraphs = article.get("article paragraphs", [])
            if not paragraphs:
                continue

            # classify with LLM
            is_rel = is_credit_relevant(paragraphs)
            print(is_rel)
            record = {
                "document title": doc_title,
                "document name": doc_name,
                "article id": article.get("article id"),
                "article name": article.get("article name"),
                "article paragraphs": paragraphs,
            }

            if is_rel:
                relevant_articles.append(record)
            else:
                unrelated_articles.append(record)

            print(f"  Article {art_index}/{total_articles} classified")

        print("")  # spacing between documents

    # Save output files
    with open(output_relevant, "w", encoding="utf-8") as f:
        json.dump(relevant_articles, f, ensure_ascii=False, indent=4)

    with open(output_unrelated, "w", encoding="utf-8") as f:
        json.dump(unrelated_articles, f, ensure_ascii=False, indent=4)

    print("Processing completed!")
    print(f"Relevant articles saved to: {output_relevant}")
    print(f"Unrelated articles saved to: {output_unrelated}")
    print(f"Total relevant: {len(relevant_articles)}")
    print(f"Total unrelated: {len(unrelated_articles)}")


if __name__ == "__main__":
    classify_articles()
