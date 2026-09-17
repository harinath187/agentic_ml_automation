import pandas as pd

from tools.text_vectorization import apply_tfidf_vectorizers, fit_tfidf_vectorizers


def test_tfidf_fits_on_train_only_and_drops_raw_column():
    train = pd.DataFrame({"notes": ["free winner offer", "regular account update"]})
    test = pd.DataFrame({"notes": ["unseen token free"]})

    vectorizers, log = fit_tfidf_vectorizers(train, ["notes"], max_features=2)
    transformed = apply_tfidf_vectorizers(test, vectorizers, ["notes"])

    assert "notes" not in transformed.columns
    assert all(column.startswith("notes_tfidf_") for column in transformed.columns)
    assert len(vectorizers["notes"].vocabulary_) <= 2
    assert "unseen" not in vectorizers["notes"].vocabulary_
    assert "notes" in log["feature_tokens"]


def test_tfidf_empty_vocabulary_is_skipped_with_a_safe_log():
    train = pd.DataFrame({"notes": ["", None]})

    vectorizers, log = fit_tfidf_vectorizers(train, ["notes"])

    assert "notes" not in vectorizers
    assert log["columns"]["notes"]["vocab_size"] == 0
    assert "skipped_reason" in log["columns"]["notes"]