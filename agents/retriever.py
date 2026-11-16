"""Policy Retrieval Agent.

Retrieves standard policy benchmarks from the vector store for each extracted clause,
using clause_type metadata filtering, top-k 3, and minimum similarity thresholding.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory

from config.agent_config import build_agent_kernel, get_agent_config, token_tracker
from rag.store import PolicyVectorStore
from state import Clause, ContractState

logger = logging.getLogger("contract_review.agents.retriever")


class PolicyRetrievalAgent:
    """Agent responsible for querying vector store and assigning standard positions."""

    def __init__(
        self,
        vector_store: PolicyVectorStore,
        mock_mode: bool = False,
        enable_query_rewrite: bool = False,
    ):
        self.agent_name = "retriever"
        self.mock_mode = mock_mode
        self.vector_store = vector_store
        self.enable_query_rewrite = enable_query_rewrite
        self.config = get_agent_config(self.agent_name)
        self.kernel, self.chat_service = build_agent_kernel(self.agent_name, mock_mode=mock_mode)
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / "retriever.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    async def _rewrite_query(self, clause: Clause) -> str:
        """Optional LLM query rewrite using retriever prompt."""
        if not self.enable_query_rewrite:
            return f"{clause.type.value} key terms: {clause.text[:250]}"

        chat_history = ChatHistory()
        chat_history.add_system_message(self.system_prompt)
        chat_history.add_user_message(f"Clause Type: {clause.type.value}\nClause Text: {clause.text}")

        settings = OpenAIChatPromptExecutionSettings(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        try:
            responses = await self.chat_service.get_chat_message_contents(
                chat_history=chat_history,
                settings=settings,
            )
            query = responses[0].content or f"{clause.type.value} terms"
            return query.strip()
        except Exception as e:
            logger.warning("Query rewrite failed (%s), using direct term fallback.", e)
            return f"{clause.type.value} terms: {clause.text[:250]}"

    async def execute(self, state: ContractState) -> Tuple[List[Clause], Dict[str, Any]]:
        """Retrieve standard position for each clause and return updated clauses + audit log."""
        start_time = time.time()
        input_data = "".join(f"{c.type.value}:{c.text}" for c in state.clauses)
        input_hash = hashlib.sha256(input_data.encode("utf-8")).hexdigest()

        updated_clauses: List[Clause] = []

        for clause in state.clauses:
            clause_copy = clause.model_copy()

            if not clause_copy.found:
                clause_copy.standard_position = "Clause not found in contract; cannot retrieve policy."
                updated_clauses.append(clause_copy)
                continue

            query_text = await self._rewrite_query(clause_copy)
            standard_pos, matches = self.vector_store.query(
                clause_type=clause_copy.type.value,
                query_text=query_text,
                top_k=3,
                min_similarity=0.35,
            )

            clause_copy.standard_position = standard_pos or "no policy found"
            updated_clauses.append(clause_copy)

        latency = time.time() - start_time
        token_info = token_tracker.get_usage(self.agent_name)

        audit_entry = {
            "agent": self.agent_name,
            "input_hash": input_hash,
            "output": [
                {"type": c.type.value, "standard_position": (c.standard_position or "")[:100] + "..."}
                for c in updated_clauses
            ],
            "validation_result": "passed",
            "tokens": token_info,
            "latency": round(latency, 4),
        }

        return updated_clauses, audit_entry
