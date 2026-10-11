import pytest
import pandas as pd

from src.privacy import append_csv_record, pseudonymiser_usager_id


def test_pseudonymiser_usager_id_is_stable_and_does_not_reveal_id(monkeypatch):
    monkeypatch.setenv("USAGER_ID_HMAC_KEY", "test-secret-key")

    pseudonym = pseudonymiser_usager_id(" ID_0001 ")

    assert pseudonym == pseudonymiser_usager_id("ID_0001")
    assert pseudonym != "ID_0001"
    monkeypatch.setenv("USAGER_ID_HMAC_KEY", "another-test-secret")
    assert pseudonym != pseudonymiser_usager_id("ID_0001")


def test_pseudonymiser_usager_id_requires_a_key(monkeypatch):
    monkeypatch.delenv("USAGER_ID_HMAC_KEY", raising=False)

    with pytest.raises(RuntimeError, match="USAGER_ID_HMAC_KEY"):
        pseudonymiser_usager_id("ID_0001")


def test_append_csv_record_initializes_an_empty_file(tmp_path):
    path = tmp_path / "predictions.csv"
    path.touch()

    append_csv_record(
        path,
        {"session_id": "session-1", "usager_id": "ID_0001", "probability": 0.75},
    )

    assert pd.read_csv(path).to_dict("records") == [
        {"session_id": "session-1", "usager_id": "ID_0001", "probability": 0.75}
    ]


def test_append_csv_record_removes_legacy_raw_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("USAGER_ID_HMAC_KEY", "test-secret-key")
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "session_id": "old-session",
                "usager_id": "ID_0001",
                "usager_id_pseudonymise": "existing-pseudonym",
                "probability": 0.75,
            }
        ]
    ).to_csv(path, index=False)

    append_csv_record(
        path,
        {
            "session_id": "new-session",
            "usager_id_pseudonymise": "new-pseudonym",
            "probability": 0.8,
        },
        pseudonymiser_ancien_id=True,
    )

    historique = pd.read_csv(path)
    assert "usager_id" not in historique.columns
    assert historique["usager_id_pseudonymise"].tolist() == [
        "existing-pseudonym",
        "new-pseudonym",
    ]