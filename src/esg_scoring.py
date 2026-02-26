"""
ESG scoring helper functions.

Provides utilities for:
- Loading the ESG dictionary (Baier, Berninger & Kiesel 2020)
- Loading manager speech segments from processed transcript files
- Computing TF-IDF-weighted ESG topic scores per document

@author: Bastian Koch
"""

import csv
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_esg_dict(path: str | Path) -> dict[str, str]:
    """
    Load the ESG wordlist CSV and return a mapping of {word: topic}.

    The CSV is expected to have at least two columns: 'Word' and 'Topic'.
    Words are lowercased to match TfidfVectorizer's default behaviour.

    Parameters
    ----------
    path : str or Path
        Path to the ESG wordlist CSV file.

    Returns
    -------
    dict
        {lowercase_word: topic} e.g. {'climate': 'Environmental', ...}
    """
    esg_dict = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["Word"].strip().lower()
            topic = row["Topic"].strip()
            if word:
                esg_dict[word] = topic
    return esg_dict


def load_segment_texts(
    segments_dir: str | Path,
    filenames: list[str],
    suffix: str,
) -> tuple[list[str], list[str]]:
    """
    Load text from processed transcript segment files.

    For each entry in `filenames` (e.g. '1665733' or
    '2021-Mar-04-BKTI.A-...-transcript'), attempts to open
    ``segments_dir/{filename}_{suffix}.txt``.  Missing files produce an
    empty string so the document index stays aligned with `filenames`.

    Parameters
    ----------
    segments_dir : str or Path
        Directory containing the segment .txt files.
    filenames : list of str
        Transcript filename stems (without .txt extension).
    suffix : str
        One of 'presentation', 'answers', 'questions', 'deleted_text'.

    Returns
    -------
    texts : list of str
        Raw text for each filename (empty string if file not found).
    found_flags : list of bool
        True if the file existed, False otherwise.
    """
    segments_dir = Path(segments_dir)
    texts = []
    found_flags = []

    for fname in filenames:
        filepath = segments_dir / f"{fname}_{suffix}.txt"
        if filepath.exists():
            texts.append(filepath.read_text(encoding="utf-8"))
            found_flags.append(True)
        else:
            texts.append("")
            found_flags.append(False)

    return texts, found_flags


def compute_tfidf_esg_scores(
    texts: list[str],
    filenames: list[str],
    esg_dict: dict[str, str],
) -> pd.DataFrame:
    """
    Compute TF-IDF-weighted ESG topic scores for a list of documents.

    Fits a TfidfVectorizer restricted to the ESG vocabulary on the full
    corpus, then sums the TF-IDF weights per document per topic
    (Environmental, Social, Governance) and as an aggregate (esg_total).

    Scores are normalised by the number of unique ESG words in each topic
    so that documents with fewer matching words are not artificially penalised.

    Parameters
    ----------
    texts : list of str
        Document texts (one per transcript segment).
    filenames : list of str
        Transcript filename stems — used as the index of the output DataFrame.
    esg_dict : dict
        {lowercase_word: topic} as returned by load_esg_dict().

    Returns
    -------
    pd.DataFrame
        One row per document with columns:
        ['filename', 'esg_total', 'esg_environmental', 'esg_social', 'esg_governance']
        All scores are floats (0.0 for empty documents or no matches).
    """
    esg_words = list(esg_dict.keys())
    topics = sorted(set(esg_dict.values()))

    # Build vocabulary: only ESG words (IDF is still computed across full corpus)
    vectorizer = TfidfVectorizer(vocabulary=esg_words, lowercase=True)

    # Fit & transform — results in a (n_docs × n_esg_words) sparse matrix
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Map each ESG word to its column index in the matrix
    feature_names = vectorizer.get_feature_names_out()
    word_to_col = {word: idx for idx, word in enumerate(feature_names)}

    # Build topic → column indices lookup
    topic_cols = {}
    for word, topic in esg_dict.items():
        if word in word_to_col:
            topic_cols.setdefault(topic, []).append(word_to_col[word])

    # Sum TF-IDF weights per document per topic
    records = []
    for i, fname in enumerate(filenames):
        row_vec = tfidf_matrix[i]
        scores = {"filename": fname}
        total = 0.0
        for topic in topics:
            cols = topic_cols.get(topic, [])
            score = float(row_vec[:, cols].sum()) if cols else 0.0
            scores[f"esg_{topic.lower()}"] = score
            total += score
        scores["esg_total"] = total
        records.append(scores)

    return pd.DataFrame(records)
