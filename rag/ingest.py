"""Ingest policy library into the vector store."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.store import PolicyVectorStore


def ingest_policy_library(
    store: PolicyVectorStore,
    library_path: Optional[str] = None,
) -> int:
    """Ingest policies from JSON library into the vector store.
    
    Returns:
        Number of chunks ingested.
    """
    if library_path is None:
        library_path = str(Path(__file__).parent / "policy_library.json")

    with open(library_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    policies = data.get("policies", [])
    count = 0

    for policy in policies:
        c_type = policy["clause_type"]
        notes = policy.get("notes", "")
        full_dict = {
            "preferred": policy.get("preferred", ""),
            "acceptable": policy.get("acceptable", ""),
            "unacceptable": policy.get("unacceptable", ""),
            "notes": notes,
        }

        # Add preferred variant
        if "preferred" in policy:
            store.add_policy_chunk(
                chunk_id=f"{c_type}_preferred",
                document=f"{c_type} preferred position: {policy['preferred']}",
                clause_type=c_type,
                variant="preferred",
                notes=notes,
                full_policy=full_dict,
            )
            count += 1

        # Add acceptable variant
        if "acceptable" in policy:
            store.add_policy_chunk(
                chunk_id=f"{c_type}_acceptable",
                document=f"{c_type} acceptable position: {policy['acceptable']}",
                clause_type=c_type,
                variant="acceptable",
                notes=notes,
                full_policy=full_dict,
            )
            count += 1

        # Add unacceptable variant
        if "unacceptable" in policy:
            store.add_policy_chunk(
                chunk_id=f"{c_type}_unacceptable",
                document=f"{c_type} unacceptable position: {policy['unacceptable']}",
                clause_type=c_type,
                variant="unacceptable",
                notes=notes,
                full_policy=full_dict,
            )
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest standard policy library into vector store.")
    parser.add_argument("--mock", action="store_true", help="Run with mock embedding function")
    parser.add_argument("--persist-dir", type=str, default=None, help="Directory to persist vector store")
    parser.add_argument("--library", type=str, default=None, help="Path to policy_library.json")
    args = parser.parse_args()

    store = PolicyVectorStore(persist_dir=args.persist_dir, mock_mode=args.mock)
    num_ingested = ingest_policy_library(store, args.library)
    print(f"Successfully ingested {num_ingested} policy variants into vector store.")


if __name__ == "__main__":
    main()
