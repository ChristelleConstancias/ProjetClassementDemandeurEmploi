import hashlib
import hmac
import os
from pathlib import Path
from typing import Any

import pandas as pd

# Fonctions de confidentialité pour le traitement des identifiants des usagers
def pseudonymiser_usager_id(usager_id: str) -> str:
    key = os.getenv("USAGER_ID_HMAC_KEY")
    if not key or not key.strip():
        raise RuntimeError("USAGER_ID_HMAC_KEY doit être configurée")

    return hmac.new(
        key.encode("utf-8"),
        usager_id.strip().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

# Fonction pour ajouter un enregistrement à un fichier CSV, 
# avec option de pseudonymisation de l'ancien identifiant usager
def append_csv_record(
    path: Path,
    record: dict[str, Any],
    *,
    pseudonymiser_ancien_id: bool = False,
) -> None:
    row = pd.DataFrame([record])
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        row.to_csv(path, index=False)
        return

    existing = pd.read_csv(path, dtype={"usager_id": "string"})
    migrated_raw_ids = pseudonymiser_ancien_id and "usager_id" in existing.columns
    if migrated_raw_ids:
        anciens_ids = existing.pop("usager_id")
        if "usager_id_pseudonymise" not in existing.columns:
            existing["usager_id_pseudonymise"] = anciens_ids.map(
                lambda value: (
                    pseudonymiser_usager_id(value) if pd.notna(value) else pd.NA
                )
            )

    if not migrated_raw_ids and set(existing.columns) == set(row.columns):
        row.reindex(columns=existing.columns).to_csv(
            path,
            mode="a",
            header=False,
            index=False,
        )
        return

    columns = list(dict.fromkeys([*existing.columns, *row.columns]))
    combined = pd.concat(
        [existing.reindex(columns=columns), row.reindex(columns=columns)],
        ignore_index=True,
    )
    temporary_path = path.with_name(f".{path.name}.tmp")
    combined.to_csv(temporary_path, index=False)
    temporary_path.replace(path)