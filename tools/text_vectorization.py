"""Train-only TF-IDF preprocessing for profiled free-text columns."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf_vectorizers(
    train_df: pd.DataFrame,
    text_columns: list[str],
    max_features: int = 100,
) -> tuple[dict[str, TfidfVectorizer], dict]:
    vectorizers: dict[str, TfidfVectorizer] = {}
    log: dict = {"columns": {}, "feature_tokens": {}}
    for column in text_columns:
        if column not in train_df.columns:
            continue
        values = train_df[column].fillna("").astype(str)
        vectorizer = TfidfVectorizer(max_features=max_features)
        try:
            vectorizer.fit(values)
        except ValueError:
            vectorizer = TfidfVectorizer(max_features=max_features, min_df=1)
            try:
                vectorizer.fit(values)
            except ValueError as exc:
                log["columns"][column] = {"vocab_size": 0, "top_tokens": [], "skipped_reason": str(exc)}
                continue
        vectorizers[column] = vectorizer
        names = vectorizer.get_feature_names_out().tolist()
        log["columns"][column] = {
            "vocab_size": len(names),
            "top_tokens": names[:10],
        }
        log["feature_tokens"][column] = {
            f"{column}_tfidf_{index}": token for index, token in enumerate(names)
        }
    return vectorizers, log


def apply_tfidf_vectorizers(
    df: pd.DataFrame,
    vectorizers: dict[str, TfidfVectorizer],
    text_columns: list[str],
) -> pd.DataFrame:
    out = df.copy()
    for column in text_columns:
        if column not in out.columns:
            continue
        vectorizer = vectorizers.get(column)
        if vectorizer is not None:
            matrix = vectorizer.transform(out[column].fillna("").astype(str))
            names = [f"{column}_tfidf_{index}" for index in range(matrix.shape[1])]
            vectors = pd.DataFrame(matrix.toarray(), index=out.index, columns=names)
            out = pd.concat([out.drop(columns=[column]), vectors], axis=1)
        else:
            out = out.drop(columns=[column])
    return out