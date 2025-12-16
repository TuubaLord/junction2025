import json
import re
from pathlib import Path

# --------------------------------------------------------------------
# Constants and configurations for different sources
# --------------------------------------------------------------------

# Common patterns
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

# Source-specific configurations
SOURCE_CONFIGS = {
    "common": {
        "data_folder": Path(__file__).parent.parent.resolve() / "data" / "gold" / "EBA"
    },
    "EBA": {
        "file_pattern": "*.di.json",
        "heading_pattern": re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$"),
        # "(1) text" OR "1. text"
        "paragraph_pattern": re.compile(r"^(?:\((\d+)\)|(\d+)\.)\s*(.*)"),
        "title_keywords": ["Guidelines", "GUIDELINES"],
    },
    "fiva_mok": {
        "file_pattern": "*.di.json",
        "heading_pattern": re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$"),
        # Only "(1) text"
        "paragraph_pattern": re.compile(r"^\((\d+)\)\s*(.*)"),
        "title_keywords": ["Regulations and guidelines"],
    },
}


# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------


def guess_document_title(paragraphs, config):
    """
    Try to find the main title based on source-specific keywords.
    Fallback: first non-empty paragraph.
    """

    for p in paragraphs[:50]:
        if any(keyword in p for keyword in config["title_keywords"]):
            return p.strip()

    # Fallback to first non-empty paragraph
    for p in paragraphs:
        if p.strip():
            return p.strip()

    return ""


def is_probable_article_heading(article_id, article_name):
    """
    Filter out things that look like metadata instead of real section headings.
    This heuristic works for both EBA and FIVA_MOK documents.
    """

    # Page footers / pure numbers: no letters in title
    if not re.search(r"[A-Za-z]", article_name):
        return False

    # Date-like IDs: e.g. "29.6.2014"
    parts = article_id.split(".")
    if len(parts) == 3 and len(parts[-1]) == 4:
        try:
            year = int(parts[-1])
            if 1900 <= year <= 2100:
                return False
        except ValueError:
            pass

    lower_name = article_name.lower()

    # Metadata-style titles (Issued, Valid from, etc.)
    first_word = article_name.split()[0] if article_name.split() else ""
    if first_word.lower() == "issued" or lower_name.startswith("valid from"):
        return False

    # 'until further notice' lines, often combined with dates
    if "until further notice" in lower_name:
        return False

    # Month names in the title usually mean it's part of a date line
    if any(m in article_name for m in MONTHS):
        return False

    # Common metadata prefixes
    meta_starts = (
        "Journal Number",
        "J. No",
        "J.No",
        "FIN-FSA",
        "FS ",
    )
    if article_name.startswith(meta_starts):
        return False

    return True


# --------------------------------------------------------------------
# Core parsing functions
# --------------------------------------------------------------------


def parse_articles(paragraphs, config):
    """
    Parse numbered article headings and their numbered paragraphs.
    Uses source-specific configuration for paragraph patterns.
    """

    para_pattern = config["paragraph_pattern"]

    articles = []
    current_article = None
    current_para_text = None

    def flush_current_paragraph():
        nonlocal current_para_text
        if current_article is not None and current_para_text:
            text = current_para_text.strip()
            if text:
                current_article["article paragraphs"].append(text)
        current_para_text = None

    def flush_current_article():
        nonlocal current_article
        if current_article is not None:
            flush_current_paragraph()
            articles.append(current_article)
        current_article = None

    for line in paragraphs:
        if not line:
            continue

        # Possible heading?
        m_head = config["heading_pattern"].match(line)
        if m_head:
            article_id = m_head.group(1).strip()
            article_name = m_head.group(2).strip()

            if is_probable_article_heading(article_id, article_name):
                flush_current_article()
                current_article = {
                    "article id": article_id,
                    "article name": article_name,
                    "article paragraphs": [],
                }
                current_para_text = None
                continue

        # If we are inside an article, collect its paragraphs
        if current_article is not None:
            m_para = para_pattern.match(line)

            if m_para:
                # New numbered paragraph
                flush_current_paragraph()
                current_para_text = line
            else:
                # Continuation line / unnumbered text
                if current_para_text:
                    current_para_text += " " + line.strip()
                else:
                    # Only start unnumbered text if the article already
                    # has at least one numbered paragraph
                    if any(
                        para_pattern.match(p)
                        for p in current_article["article paragraphs"]
                    ):
                        current_para_text = line

    # Flush last article
    flush_current_article()

    # Final filter: keep only articles that contain at least one numbered paragraph
    filtered_articles = []
    for a in articles:
        if any(para_pattern.match(p) for p in a["article paragraphs"]):
            filtered_articles.append(a)

    return filtered_articles


def parse_document(file_path, config):
    """Parse a single document into the target structure."""

    with open(file_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    # Document name from filename
    document_name = file_path.stem

    # Get paragraphs
    paragraphs = []
    for page in doc.get("pages", []):
        for p in page.get("paragraphs", []):
            if isinstance(p, str):
                paragraphs.append(p.strip())

    document_title = guess_document_title(paragraphs, config)
    articles = parse_articles(paragraphs, config)

    return {
        "document title": document_title,
        "document name": document_name,
        "articles": articles,
    }


def parse_all_documents(source):
    """
    Parse all documents matching the glob pattern for the specified source.
    If pattern is None, uses the default pattern for the source.
    """

    # Check that source exists
    if source not in SOURCE_CONFIGS:
        raise ValueError(
            f"Unknown source: {source}. Supported sources: {list(SOURCE_CONFIGS.keys())}"
        )

    config = SOURCE_CONFIGS[source]

    # Loop through all documents matching the pattern
    results = []
    for path in sorted(
        SOURCE_CONFIGS["common"]["data_folder"].glob(config["file_pattern"])
    ):
        print(f"Parsing {path}...")
        results.append(parse_document(path, config))

    return results


# --------------------------------------------------------------------
# Example usage and main execution
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage: Parse EBA documents
    source = "EBA"
    # source = "fiva_mok"

    results = parse_all_documents(source)
    print(f"Successfully parsed {len(results)} EBA documents")

    # Save to file
    print("Saving results")
    save_dir = (
        Path(__file__).parent.parent.resolve()
        / "data"
        / "intermediate"
        / "parsed_documents"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / f"all_{source}_parsed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
