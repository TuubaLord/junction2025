import json
import os
import glob
import re
from typing import List, Dict, Any

# --------------------------------------------------------------------
# Patterns for headings and numbered paragraphs
# --------------------------------------------------------------------

# e.g. "4.1 Reportable transactions and content of the report"
HEADING_RE = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')

# e.g. "(1) leipä", "(2) textiä"
PARA_START_RE = re.compile(r'^\((\d+)\)\s*(.*)')

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
)

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def load_paragraphs(doc: Dict[str, Any]) -> List[str]:
    """Flatten all string paragraphs from all pages into a single list."""
    paragraphs: List[str] = []
    for page in doc.get("pages", []):
        for p in page.get("paragraphs", []):
            if isinstance(p, str):
                paragraphs.append(p.strip())
    return paragraphs


def guess_document_title(paragraphs: List[str]) -> str:
    """
    Try to find the main title, typically a line like
    'Regulations and guidelines 4/2018 ...'.
    Fallback: first non-empty paragraph.
    """
    for p in paragraphs[:40]:  # look near the beginning
        if "Regulations and guidelines" in p:
            return p.strip()
    for p in paragraphs:
        if p.strip():
            return p.strip()
    return ""


def guess_document_name(file_path: str) -> str:
    """Use the filename (without extension) as a simple document name."""
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    return name


# --------------------------------------------------------------------
# Heading heuristics
# --------------------------------------------------------------------

def is_probable_article_heading(article_id: str, article_name: str) -> bool:
    """
    Filter out things that look like metadata instead of real section headings:
    - page footers like '2 (14)'
    - date lines like '29.6.2014 until further notice'
    - 'Valid from ...', 'Issued ...', etc.
    - lines with month names / 'until further notice'
    """
    # Page footers / pure numbers: no letters in title
    if not re.search(r'[A-Za-z]', article_name):
        return False

    # Date-like IDs: e.g. "29.6.2014"
    parts = article_id.split('.')
    if len(parts) == 3 and len(parts[-1]) == 4:
        try:
            year = int(parts[-1])
            if 1900 <= year <= 2100:
                return False
        except ValueError:
            pass

    lower_name = article_name.lower()

    # Metadata-style titles (Issued, Valid from, etc.)
    first_word = article_name.split()[0]
    if first_word.lower() == "issued" or lower_name.startswith("valid from"):
        return False

    # 'until further notice' lines, often combined with dates
    if "until further notice" in lower_name:
        return False

    # Month names in the title usually mean it's part of 'Valid from 31 December 2012 ...'
    if any(m in article_name for m in MONTHS):
        return False

    # Other obvious metadata starts
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
# Core article parsing
# --------------------------------------------------------------------

def parse_articles(paragraphs: List[str]) -> List[Dict[str, Any]]:
    """
    Parse numbered article headings and their numbered paragraphs.

    Strategy:
    - Detect headings like '4.1 Title...' or '1 Scope of application...' with HEADING_RE.
    - Use heuristics to ignore metadata / dates / ToC.
    - Inside an article, we only *keep* paragraphs if the article has at least one
      numbered paragraph '(n) ...'. This filters out table-of-contents entries.
    - In the final result, we only return articles that contain at least one
      numbered paragraph '(n)...'.
    """
    articles: List[Dict[str, Any]] = []
    current_article: Dict[str, Any] | None = None
    current_para_text: str | None = None

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
        m_head = HEADING_RE.match(line)
        if m_head:
            article_id = m_head.group(1).strip()
            article_name = m_head.group(2).strip()
            if is_probable_article_heading(article_id, article_name):
                # Start a new article
                flush_current_article()
                current_article = {
                    "article id": article_id,
                    "article name": article_name,
                    "article paragraphs": []
                }
                current_para_text = None
                continue  # go to next line

        # If we are inside an article, collect its paragraphs
        if current_article is not None:
            m_para = PARA_START_RE.match(line)
            if m_para:
                # New numbered paragraph "(n) ..."
                flush_current_paragraph()
                current_para_text = line
            else:
                # Continuation / unnumbered line
                if current_para_text:
                    # append continuation to existing paragraph
                    current_para_text += " " + line.strip()
                else:
                    # Only start an unnumbered paragraph if this article
                    # already has at least one numbered paragraph; this avoids
                    # turning pure ToC sections into "articles".
                    if any(PARA_START_RE.match(p)
                           for p in current_article["article paragraphs"]):
                        current_para_text = line

    # Flush last open article
    flush_current_article()

    # Final filter: keep only articles that contain at least one "(n) ..." paragraph
    filtered_articles: List[Dict[str, Any]] = []
    for a in articles:
        if any(PARA_START_RE.match(p) for p in a["article paragraphs"]):
            filtered_articles.append(a)

    return filtered_articles


# --------------------------------------------------------------------
# Top-level document API
# --------------------------------------------------------------------

def parse_document(file_path: str) -> Dict[str, Any]:
    """Parse a single *.di.json document into the desired output structure."""
    with open(file_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    paragraphs = load_paragraphs(doc)
    document_title = guess_document_title(paragraphs)
    document_name = guess_document_name(file_path)
    articles = parse_articles(paragraphs)

    return {
        "document title": document_title,
        "document name": document_name,
        "articles": articles,
    }


def parse_all_documents(pattern: str = "*.di.json") -> List[Dict[str, Any]]:
    """
    Parse all documents matching the glob pattern (default: all *.di.json
    in the current directory) and return them as a list.
    """
    result: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        print(f"Parsing {path}...")
        result.append(parse_document(path))
    return result


if __name__ == "__main__":
    # Adjust pattern if your files are elsewhere, e.g. "data/*.di.json"
    documents = parse_all_documents("*.di.json")

    with open("all_documents_parsed.json", "w", encoding="utf-8") as out_f:
        json.dump(documents, out_f, ensure_ascii=False, indent=4)
