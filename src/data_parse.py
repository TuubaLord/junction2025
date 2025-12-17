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
    "EBA": {
        "data_folder": Path(__file__).parent.parent.resolve() / "data" / "gold" / "EBA",
        "file_pattern": "*.di.json",
        "heading_pattern": re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$"),
        # "(1) text" OR "1. text"
        "paragraph_pattern": re.compile(r"^(?:\((\d+)\)|(\d+)\.)\s*(.*)"),
        "title_keywords": ["Guidelines", "GUIDELINES"],
    },
    "fiva_mok": {
        "data_folder": Path(__file__).parent.parent.resolve()
        / "data"
        / "gold"
        / "FIVA_MOK",
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
    Extract the main document title from paragraphs using source-specific keywords.

    Searches through the first 50 paragraphs for lines containing title keywords
    specific to the document source (e.g., "Guidelines" for EBA, "Regulations and guidelines"
    for FIVA_MOK). Falls back to the first non-empty paragraph if no keywords are found.

    Args:
        paragraphs (List[str]): List of paragraph strings from the document
        config (Dict): Source configuration containing "title_keywords" list

    Returns:
        str: The identified document title, or empty string if no content found
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
    Determine if a heading represents a genuine article section vs. metadata.

    Uses heuristics to filter out document metadata like dates, page numbers,
    publication info, and other non-content headings that appear in both EBA
    and FIVA_MOK documents.

    Args:
        article_id (str): The numeric ID part of the heading (e.g., "4.1")
        article_name (str): The text part of the heading (e.g., "General provisions")

    Returns:
        bool: True if this appears to be a legitimate article heading, False if likely metadata
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
# Core functions
# --------------------------------------------------------------------


def parse_articles(paragraphs, config):
    """
    Extract structured articles from document paragraphs using source-specific patterns.

    Processes a flat list of paragraphs to identify article headings (e.g., "4.1 General provisions")
    and groups subsequent numbered paragraphs under each heading. Uses source-specific regex
    patterns to handle different paragraph numbering formats (EBA supports both "(1)" and "1.",
    while FIVA_MOK only supports "(1)").

    Args:
        paragraphs (List[str]): Flat list of all paragraph strings from the document
        config (Dict): Source configuration containing regex patterns for headings and paragraphs

    Returns:
        List[Dict]: List of article dictionaries, each containing:
            - "article id": The article number (e.g., "4.1")
            - "article name": The article title (e.g., "General provisions")
            - "article paragraphs": List of paragraph strings belonging to this article

    Note:
        Only returns articles that contain at least one numbered paragraph to filter out
        table-of-contents entries and other non-content sections.
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
    """
    Parse a single JSON document file into structured article data.

    Loads a .di.json file, extracts paragraphs from all pages, identifies the document
    title, and parses articles using source-specific configuration patterns.

    Args:
        file_path (Path): Path to the .di.json document file to parse
        config (Dict): Source configuration containing parsing patterns and keywords

    Returns:
        Dict: Document dictionary containing:
            - "document title": Main title extracted from document content
            - "document name": Filename without extension
            - "articles": List of parsed article dictionaries

    Raises:
        FileNotFoundError: If the document file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        PermissionError: If file access is denied
    """

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
    Parse all documents for a specified source type using configured patterns and locations.

    Loads the source configuration, finds all matching .di.json files in the configured
    data folder, and parses each document into structured article data.

    Args:
        source (str): Source type identifier ("EBA", "fiva_mok", etc.) that must exist
                     in SOURCE_CONFIGS

    Returns:
        List[Dict]: List of parsed document dictionaries, each containing document title,
                   document name, and structured articles

    Raises:
        ValueError: If the source is not found in SOURCE_CONFIGS

    Note:
        Prints progress information during parsing and a summary upon completion.
    """

    # Check that source exists
    if source not in SOURCE_CONFIGS:
        raise ValueError(
            f"Unknown source: {source}. Supported sources: {list(SOURCE_CONFIGS.keys())}"
        )

    config = SOURCE_CONFIGS[source]

    # Loop through all documents matching the pattern
    results = []
    for path in sorted(config["data_folder"].glob(config["file_pattern"])):
        print(f"Parsing {path}...")
        results.append(parse_document(path, config))

    print(f"Successfully parsed {len(results)} {source} documents")

    return results


# --------------------------------------------------------------------
# Example usage and main execution
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage: Parse EBA documents
    source = "EBA"
    # source = "fiva_mok"

    # Process
    results = parse_all_documents(source)

    # Save to file
    print("Saving results")
    save_dir = (
        Path(__file__).parent.parent.resolve() / "data" / "intermediate" / "parsed"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"all_{source}_parsed.json"
    with open(save_dir / filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(results)} documents to {filename}")
