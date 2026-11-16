"""Sequential Pipeline coordinating Extractor -> Retriever -> Flagger -> Summarizer.

Enforces:
- Single shared ContractState passed through the pipeline.
- Input guardrails and prompt injection scanning.
- Scope boundary enforcement in code via ScopeGuard.
- Complete audit logging per agent.
- Standard report generation (Markdown and JSON).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agents.extractor import ClauseExtractorAgent
from agents.flagger import RiskFlaggerAgent
from agents.retriever import PolicyRetrievalAgent
from agents.summarizer import ContractSummarizerAgent
from config.agent_config import validate_all_keys
from guardrails.input_guard import unwrap_contract, validate_and_sanitize_input
from guardrails.scope_guard import ScopeGuard
from rag.ingest import ingest_policy_library
from rag.store import PolicyVectorStore
from state import ClauseRisk, ContractState

logger = logging.getLogger("contract_review.pipeline")

FOOTER_DISCLAIMER: str = "Decision support only, not legal advice."


class ContractReviewPipeline:
    """Enterprise multi-agent pipeline for automated contract review and risk flagging."""

    def __init__(
        self,
        mock_mode: bool = False,
        enable_query_rewrite: bool = False,
        vector_store: Optional[PolicyVectorStore] = None,
        persist_dir: Optional[str] = None,
    ):
        self.mock_mode = mock_mode

        # Fail fast if required keys are missing (when not in mock mode)
        validate_all_keys(mock_mode=mock_mode)

        # Initialize vector store and ingest policy library
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = PolicyVectorStore(persist_dir=persist_dir, mock_mode=mock_mode)
            ingest_policy_library(self.vector_store)

        # Initialize the four sequential agents
        self.extractor = ClauseExtractorAgent(mock_mode=mock_mode)
        self.retriever = PolicyRetrievalAgent(
            vector_store=self.vector_store,
            mock_mode=mock_mode,
            enable_query_rewrite=enable_query_rewrite,
        )
        self.flagger = RiskFlaggerAgent(mock_mode=mock_mode)
        self.summarizer = ContractSummarizerAgent(mock_mode=mock_mode)

    async def run(self, raw_contract_text: str) -> ContractState:
        """Run the sequential review pipeline on contract text.
        
        Sequential execution: Extractor -> Retriever -> Flagger -> Summarizer
        """
        logger.info("Starting contract review pipeline (mock_mode=%s)...", self.mock_mode)

        # 1. Input Guardrails: Sanitize, wrap, scan for injection
        wrapped_text, injection_warnings = validate_and_sanitize_input(raw_contract_text)
        clean_raw_text = unwrap_contract(wrapped_text)

        # Create single shared state
        state = ContractState(raw_text=clean_raw_text)
        if injection_warnings:
            state.audit_log.append({
                "agent": "input_guard",
                "warnings": [f"Potential prompt injection pattern detected: '{w}'" for w in injection_warnings],
                "action": "Delimited in <contract> tags; treated strictly as untrusted data.",
            })

        # ------------------------------------------------------------------
        # Agent 1: Clause Extractor
        # ------------------------------------------------------------------
        logger.info("Executing Agent 1: Clause Extractor...")
        before_state = state.model_copy(deep=True)
        extracted_clauses, extractor_audit = await self.extractor.execute(state)

        # Candidate after state for scope validation
        candidate_state = state.model_copy(deep=True)
        candidate_state.clauses = extracted_clauses
        ScopeGuard.validate_state_transition(before_state, candidate_state, "extractor")

        # Apply mutation and log
        state.clauses = extracted_clauses
        state.audit_log.append(extractor_audit)

        # ------------------------------------------------------------------
        # Agent 2: Policy Retrieval
        # ------------------------------------------------------------------
        logger.info("Executing Agent 2: Policy Retriever...")
        before_state = state.model_copy(deep=True)
        retrieved_clauses, retriever_audit = await self.retriever.execute(state)

        candidate_state = state.model_copy(deep=True)
        candidate_state.clauses = retrieved_clauses
        ScopeGuard.validate_state_transition(before_state, candidate_state, "retriever")

        state.clauses = retrieved_clauses
        state.audit_log.append(retriever_audit)

        # ------------------------------------------------------------------
        # Agent 3: Risk Flagger
        # ------------------------------------------------------------------
        logger.info("Executing Agent 3: Risk Flagger...")
        before_state = state.model_copy(deep=True)
        flagged_clauses, flagger_audit = await self.flagger.execute(state)

        candidate_state = state.model_copy(deep=True)
        candidate_state.clauses = flagged_clauses
        ScopeGuard.validate_state_transition(before_state, candidate_state, "flagger")

        state.clauses = flagged_clauses
        state.audit_log.append(flagger_audit)

        # ------------------------------------------------------------------
        # Agent 4: Summarizer
        # ------------------------------------------------------------------
        logger.info("Executing Agent 4: Summarizer...")
        before_state = state.model_copy(deep=True)
        summary, overall_risk, recommendation, summarizer_audit = await self.summarizer.execute(state)

        candidate_state = state.model_copy(deep=True)
        candidate_state.summary = summary
        candidate_state.overall_risk = overall_risk
        candidate_state.recommendation = recommendation
        ScopeGuard.validate_state_transition(before_state, candidate_state, "summarizer")

        state.summary = summary
        state.overall_risk = overall_risk
        state.recommendation = recommendation
        state.audit_log.append(summarizer_audit)

        logger.info(
            "Pipeline completed successfully. Overall Risk: %s, Recommendation: %s",
            state.overall_risk.value if state.overall_risk else "NONE",
            state.recommendation.value if state.recommendation else "NONE",
        )

        return state

    def format_markdown_report(self, state: ContractState, contract_name: str = "Contract") -> str:
        """Generate human-readable Markdown report (§11)."""
        lines = [
            f"# Contract Review and Risk Assessment Report: {contract_name}",
            "",
            "## 1. Executive Summary",
            f"**Overall Recommendation:** `{state.recommendation.value if state.recommendation else 'N/A'}`  ",
            f"**Overall Risk Level:** `{state.overall_risk.value if state.overall_risk else 'N/A'}`  ",
            "",
            state.summary or "No summary generated.",
            "",
            "## 2. Per-Clause Risk Assessment",
            "| Clause | Risk | Error / Clause Status |",
            "| :--- | :--- | :--- |",
        ]

        for c in state.clauses:
            c_name = c.type.value.capitalize()
            c_risk = f"`{c.risk.value}`" if c.risk else "`UNKNOWN`"
            if not c.found:
                status_text = "Clause not found in contract; cannot retrieve policy."
            elif c.risk and c.risk in [ClauseRisk.REJECT, ClauseRisk.REVIEW]:
                status_text = (c.reason or "Deviation detected.").replace("|", "\\|").replace("\n", " ")
            else:
                status_text = "Clause is present in contract."
            lines.append(f"| {c_name} | {c_risk} | {status_text} |")

        lines.extend([
            "",
            "## 3. Extracted Clause Text",
            "",
        ])
        for c in state.clauses:
            status_text = "Found" if c.found else "NOT FOUND"
            lines.append(f"### {c.type.value.capitalize()} ({status_text})")
            if c.found and c.text:
                lines.append(f"```text\n{c.text}\n```")
            else:
                lines.append("*Clause omitted or not found in contract text.*")
            lines.append("")

        lines.extend([
            "---",
            f"*{FOOTER_DISCLAIMER}*",
        ])
        return "\n".join(lines)

    def format_json_report(self, state: ContractState, contract_name: str = "Contract") -> Dict[str, Any]:
        """Generate machine-readable JSON report (§11)."""
        return {
            "contract_name": contract_name,
            "overall_recommendation": state.recommendation.value if state.recommendation else None,
            "overall_risk": state.overall_risk.value if state.overall_risk else None,
            "summary": state.summary,
            "clauses": [c.model_dump() for c in state.clauses],
            "audit_log": state.audit_log,
            "disclaimer": FOOTER_DISCLAIMER,
        }

    def save_reports(
        self,
        state: ContractState,
        output_dir: str = "outputs",
        contract_name: str = "contract",
    ) -> Tuple[str, str]:
        """Save report to Markdown and JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(contract_name).stem

        md_path = os.path.join(output_dir, f"{base_name}_report.md")
        json_path = os.path.join(output_dir, f"{base_name}_report.json")

        md_content = self.format_markdown_report(state, base_name)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        json_content = self.format_json_report(state, base_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, indent=2)

        return md_path, json_path
