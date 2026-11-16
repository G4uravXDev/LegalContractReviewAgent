"""RAG Vector Store implementation for Policy Library retrieval.

Supports Chroma vector store with metadata filtering on clause_type.
Includes deterministic fallback embedding for mock/offline execution.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings


class DeterministicMockEmbeddingFunction(EmbeddingFunction):
    """Deterministic hash/n-gram based embedding function for offline/mock execution.
    
    Produces 384-dimensional normalized dense vectors matching standard sentence-transformer dimensions.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def name(self) -> str:
        return "deterministic_mock"

    def get_config(self) -> Dict[str, Any]:
        return {"dim": self.dim}

    def _embed_text(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self.dim
        if not tokens:
            return vec
        for token in tokens:
            h = hash(token) % self.dim
            vec[h] += 1.0
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_text(doc) for doc in input]


class PolicyVectorStore:
    """Vector store for standard policy clauses."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        mock_mode: bool = False,
        embedding_function: Optional[EmbeddingFunction] = None,
    ):
        self.mock_mode = mock_mode
        self.persist_dir = persist_dir

        if embedding_function is not None:
            self.ef = embedding_function
        elif mock_mode:
            self.ef = DeterministicMockEmbeddingFunction(dim=384)
        else:
            # When not in mock mode and no custom ef provided, let Chroma use its ONNX embedding
            self.ef = None

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.EphemeralClient()

        # Differentiate collection names between mock and live modes to prevent dimension collisions
        self.col_name = "standard_policy_clauses_mock" if self.mock_mode else "standard_policy_clauses_live"
        self.collection = self._init_collection()

    def _init_collection(self):
        """Initialize collection safely, recreating if a dimension mismatch or schema conflict exists."""
        try:
            if self.ef is not None:
                return self.client.get_or_create_collection(
                    name=self.col_name,
                    embedding_function=self.ef,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                return self.client.get_or_create_collection(
                    name=self.col_name,
                    metadata={"hnsw:space": "cosine"},
                )
        except Exception:
            try:
                self.client.delete_collection(name=self.col_name)
            except Exception:
                pass
            if self.ef is not None:
                return self.client.create_collection(
                    name=self.col_name,
                    embedding_function=self.ef,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                return self.client.create_collection(
                    name=self.col_name,
                    metadata={"hnsw:space": "cosine"},
                )

    def add_policy_chunk(
        self,
        chunk_id: str,
        document: str,
        clause_type: str,
        variant: str,
        notes: str = "",
        full_policy: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a policy chunk to the vector store with metadata."""
        metadata = {
            "clause_type": clause_type.lower(),
            "variant": variant.lower(),
            "notes": notes,
        }
        if full_policy:
            metadata["preferred"] = full_policy.get("preferred", "")
            metadata["acceptable"] = full_policy.get("acceptable", "")
            metadata["unacceptable"] = full_policy.get("unacceptable", "")

        try:
            self.collection.upsert(
                ids=[chunk_id],
                documents=[document],
                metadatas=[metadata],
            )
        except Exception as e:
            if "dimension" in str(e).lower() or "expecting embedding" in str(e).lower():
                try:
                    self.client.delete_collection(name=self.col_name)
                except Exception:
                    pass
                self.collection = self._init_collection()
                self.collection.upsert(
                    ids=[chunk_id],
                    documents=[document],
                    metadatas=[metadata],
                )
            else:
                raise

    def query(
        self,
        clause_type: str,
        query_text: str,
        top_k: int = 3,
        min_similarity: float = 0.35,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Query policy library filtered by clause_type.
        
        Returns:
            (formatted_standard_position, raw_results)
            If nothing found above threshold, returns ("no policy found", [])
        """
        clause_type_norm = clause_type.lower().strip()
        where_filter = {"clause_type": clause_type_norm}

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return "no policy found", []

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        documents = results.get("documents", [[]])[0] if results.get("documents") else []

        if not ids:
            return "no policy found", []

        valid_matches = []
        for i, doc_id in enumerate(ids):
            # In Chroma with cosine distance: similarity = 1.0 - distance
            dist = distances[i] if i < len(distances) else 1.0
            similarity = max(0.0, 1.0 - dist)
            if similarity >= min_similarity or self.mock_mode:
                meta = metadatas[i] if i < len(metadatas) else {}
                doc = documents[i] if i < len(documents) else ""
                valid_matches.append({
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                    "similarity": similarity,
                })

        if not valid_matches:
            return "no policy found", []

        # Construct comprehensive standard position from metadata / matched policy
        first_meta = valid_matches[0]["metadata"]
        preferred = first_meta.get("preferred")
        acceptable = first_meta.get("acceptable")
        unacceptable = first_meta.get("unacceptable")
        notes = first_meta.get("notes", "")

        if preferred and acceptable and unacceptable:
            formatted = (
                f"Standard Position for {clause_type.upper()}:\n"
                f"- PREFERRED: {preferred}\n"
                f"- ACCEPTABLE: {acceptable}\n"
                f"- UNACCEPTABLE: {unacceptable}\n"
                f"- NOTES: {notes}"
            )
        else:
            parts = [f"Standard Position for {clause_type.upper()}:"]
            for m in valid_matches:
                variant = m["metadata"].get("variant", "variant").upper()
                parts.append(f"- {variant}: {m['document']}")
            if notes:
                parts.append(f"- NOTES: {notes}")
            formatted = "\n".join(parts)

        return formatted, valid_matches
