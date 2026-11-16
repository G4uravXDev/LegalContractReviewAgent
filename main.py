"""Command-line interface for the Legal Contract Review and Risk Flagging System.

Usage:
    python main.py --contract data/contracts/mixed.txt [--mock] [--out outputs/]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import ContractReviewPipeline
from utils.document_parser import parse_document_async, UnsupportedFileTypeError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("contract_review.main")


async def run_cli(contract_path: str, mock_mode: bool, output_dir: str, enable_query_rewrite: bool) -> None:
    path_obj = Path(contract_path)
    if not path_obj.exists():
        logger.error("Contract file not found: %s", contract_path)
        sys.exit(1)

    try:
        with open(path_obj, "rb") as f:
            file_bytes = f.read()
        contract_text = await parse_document_async(file_bytes, path_obj.name, mock_mode=mock_mode)
    except Exception as e:
        logger.error("Failed to read or parse contract '%s': %s", contract_path, e)
        sys.exit(1)

    pipeline = ContractReviewPipeline(
        mock_mode=mock_mode,
        enable_query_rewrite=enable_query_rewrite,
    )

    state = await pipeline.run(contract_text)

    # Print Report to standard output
    report_md = pipeline.format_markdown_report(state, contract_name=path_obj.name)
    print("\n" + "=" * 80)
    print(report_md)
    print("=" * 80 + "\n")

    # Save reports to outputs/
    md_path, json_path = pipeline.save_reports(state, output_dir=output_dir, contract_name=path_obj.name)
    print(f"Saved Markdown report to: {md_path}")
    print(f"Saved JSON report to:     {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legal Contract Review and Risk Flagging Multi-Agent System"
    )
    parser.add_argument(
        "--contract",
        type=str,
        required=True,
        help="Path to contract text file to evaluate (e.g. data/contracts/clean.txt)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Run in deterministic mock mode without needing API keys",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs",
        help="Directory to save generated reports (default: outputs)",
    )
    parser.add_argument(
        "--query-rewrite",
        action="store_true",
        default=False,
        help="Enable optional LLM query rewriting in Policy Retrieval agent",
    )

    args = parser.parse_args()
    asyncio.run(run_cli(args.contract, args.mock, args.out, args.query_rewrite))


if __name__ == "__main__":
    main()
