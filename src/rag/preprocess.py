"""
preprocess.py — Clean and chunk consumer complaint narratives for the RAG pipeline.

Steps
-----
1. Drop rows without a narrative.
2. Apply text normalisation (strip boilerplate, lowercase optional tokens, etc.).
3. Optionally chunk long narratives into overlapping segments.
4. Return a DataFrame ready for embedding.
"""
from __future__ import annotations

import re
import textwrap
import pandas as pd
from pathlib import Path

from src import config

# ── Constants ────────────────────────────────────────────────────────────────

# CFPB redactions appear as "XXXX" placeholders — we remove them gracefully
_REDACTION_PATTERN = re.compile(r"\bXX+\b", re.IGNORECASE)

# Boilerplate phrases that add no semantic value
_BOILERPLATE = [
    r"i am writing to complain about",
    r"i would like to file a complaint",
    r"to whom it may concern",
    r"dear (sir|madam|cfpb)",
    r"thank you for your (help|assistance|time)",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE), re.IGNORECASE)

# Maximum characters per chunk (≈ 400 tokens at ~4 chars/token)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


# ── Text normalisation ────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise a single complaint narrative string.

    1. Replace CFPB redactions (XX+) with '[REDACTED]'.
    2. Strip common boilerplate openers.
    3. Collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = _REDACTION_PATTERN.sub("[REDACTED]", text)
    text = _BOILERPLATE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping character-level chunks.

    Returns a list with at least one element (even if text is shorter than
    chunk_size, the original text is returned as-is).
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a sentence boundary
        boundary = text.rfind(". ", start, end)
        if boundary > start:
            end = boundary + 1  # include the period
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start >= len(text):
            break
    return [c for c in chunks if c]


# ── DataFrame-level preprocessing ────────────────────────────────────────────

REQUIRED_COLUMNS = ["consumer_complaint_narrative"]


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that have no narrative and ensure required columns exist."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    return df.dropna(subset=REQUIRED_COLUMNS)


def build_context_string(row: pd.Series) -> str:
    """
    Build an enriched context string per complaint that includes product/issue
    metadata alongside the cleaned narrative. Richer context improves retrieval.
    """
    parts = []
    if pd.notna(row.get("product")):
        parts.append(f"Product: {row['product']}")
    if pd.notna(row.get("issue")):
        parts.append(f"Issue: {row['issue']}")
    if pd.notna(row.get("sub_issue")):
        parts.append(f"Sub-issue: {row['sub_issue']}")
    if pd.notna(row.get("company")):
        parts.append(f"Company: {row['company']}")
    parts.append(f"Narrative: {row['cleaned_narrative']}")
    return "\n".join(parts)


def preprocess_complaints(
    df: pd.DataFrame,
    chunk: bool = False,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a complaints DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw complaints DataFrame (must have 'consumer_complaint_narrative').
    chunk : bool
        If True, long narratives are chunked; each chunk becomes a separate row.
    chunk_size, overlap : int
        Chunking parameters (characters).

    Returns
    -------
    pd.DataFrame with additional columns:
      - cleaned_narrative : normalised text
      - context           : enriched string for embedding
      - chunk_index       : 0-based chunk number (0 if not chunked)
    """
    df = validate_schema(df).copy()

    # Clean narrative
    df["cleaned_narrative"] = df["consumer_complaint_narrative"].apply(clean_text)

    if chunk:
        # Explode into chunks
        rows = []
        for _, row in df.iterrows():
            chunks = chunk_text(row["cleaned_narrative"], chunk_size, overlap)
            for i, ch in enumerate(chunks):
                r = row.copy()
                r["cleaned_narrative"] = ch
                r["chunk_index"] = i
                rows.append(r)
        df = pd.DataFrame(rows).reset_index(drop=True)
    else:
        df["chunk_index"] = 0

    # Build enriched context column
    df["context"] = df.apply(build_context_string, axis=1)

    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    use_sample = "--sample" in sys.argv
    src_path = (
        config.SAMPLE_DATA_DIR / "cfpb_complaints_sample.csv"
        if use_sample
        else config.RAW_DATA_DIR / "cfpb_complaints.csv"
    )

    if not src_path.exists():
        print(f"Source file {src_path} does not exist. Run download.py first.")
        sys.exit(1)

    raw_df = pd.read_csv(src_path)
    print(f"Loaded {len(raw_df)} raw records from {src_path}")

    processed_df = preprocess_complaints(raw_df)
    out_path = config.PROCESSED_DATA_DIR / "cfpb_complaints_processed.csv"
    processed_df.to_csv(out_path, index=False)
    print(f"Saved {len(processed_df)} preprocessed records to {out_path}")
