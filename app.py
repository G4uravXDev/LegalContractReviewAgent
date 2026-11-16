"""FastAPI Web UI for the Legal Contract Review and Risk Flagging System.

Features:
- Landing Page at / with Hero, Feature Highlights, How It Works, Agent Directory, and Security Guardrails.
- Dedicated Contract Analyzer Workspace at /analyze-contracts with 3-stage UI:
  1. Upload screen with drag-and-drop (.pdf, .docx, .txt, .png, .jpg, .webp) and one-click sample contracts.
  2. Paced progress screen with bold percentage and dynamic agent indicator.
  3. Executive report screen with 3-column Per-Clause Risk Assessment, verbatim text, and MD/JSON downloads.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import ContractReviewPipeline
from utils.document_parser import (
    ALLOWED_EXTENSIONS,
    IMAGE_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    UnsupportedFileTypeError,
    parse_document_async,
)

app = FastAPI(
    title="Legal Contract Review & Risk Flagging System",
    description="Multi-agent legal contract analysis powered by Semantic Kernel",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DATA_CONTRACTS_DIR = Path(__file__).parent / "data" / "contracts"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the landing page UI."""
    return LANDING_PAGE_HTML


@app.get("/analyze-contracts", response_class=HTMLResponse)
async def serve_analyzer():
    """Serve the dedicated contract review and risk analysis workspace."""
    return ANALYZER_PAGE_HTML


@app.post("/api/review")
async def review_contract_upload(
    file: UploadFile = File(...),
    mock_mode: bool = Form(default=False),
):
    """Upload and analyze a contract file (.pdf, .docx, .txt, or contract images .png, .jpg, .webp)."""
    filename = file.filename or "contract.txt"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{ext}'. "
                "Only documents (.pdf, .docx, .txt) and images (.png, .jpg, .jpeg, .webp) are accepted."
            ),
        )

    try:
        file_bytes = await file.read()
        extracted_text = await parse_document_async(file_bytes, filename, mock_mode=mock_mode)
    except (UnsupportedFileTypeError, ValueError) as err:
        raise HTTPException(status_code=400, detail=str(err))

    try:
        pipeline = ContractReviewPipeline(mock_mode=mock_mode)
        state = await pipeline.run(extracted_text)

        md_report = pipeline.format_markdown_report(state, contract_name=filename)
        json_report = pipeline.format_json_report(state, contract_name=filename)

        # Save to outputs directory
        pipeline.save_reports(state, output_dir="outputs", contract_name=filename)

        return JSONResponse(content={
            "success": True,
            "filename": filename,
            "overall_recommendation": state.recommendation.value if state.recommendation else None,
            "overall_risk": state.overall_risk.value if state.overall_risk else None,
            "summary": state.summary,
            "clauses": [c.model_dump() for c in state.clauses],
            "audit_log": state.audit_log,
            "markdown_report": md_report,
            "json_report": json_report,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline review failed: {e}")


@app.post("/api/review/sample")
async def review_sample_contract(
    sample_name: str = Form(...),
    mock_mode: bool = Form(default=True),
):
    """Run review on a built-in sample contract."""
    sample_path = DATA_CONTRACTS_DIR / sample_name
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample contract '{sample_name}' not found.")

    with open(sample_path, "r", encoding="utf-8") as f:
        contract_text = f.read()

    try:
        pipeline = ContractReviewPipeline(mock_mode=mock_mode)
        state = await pipeline.run(contract_text)

        md_report = pipeline.format_markdown_report(state, contract_name=sample_name)
        json_report = pipeline.format_json_report(state, contract_name=sample_name)

        pipeline.save_reports(state, output_dir="outputs", contract_name=sample_name)

        return JSONResponse(content={
            "success": True,
            "filename": sample_name,
            "overall_recommendation": state.recommendation.value if state.recommendation else None,
            "overall_risk": state.overall_risk.value if state.overall_risk else None,
            "summary": state.summary,
            "clauses": [c.model_dump() for c in state.clauses],
            "audit_log": state.audit_log,
            "markdown_report": md_report,
            "json_report": json_report,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample review failed: {e}")


# ==============================================================================
# 1. LANDING PAGE HTML
# ==============================================================================
LANDING_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Legal Contract Review &bull; AI Contract Risk Intelligence</title>
  <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Inter', sans-serif;
      background: radial-gradient(circle at 50% -10%, rgba(99, 102, 241, 0.08) 0%, rgba(248, 250, 252, 0) 65%), #f8fafc;
      opacity: 0;
      transform: translateY(8px);
      transition: opacity 0.3s cubic-bezier(0.16, 1, 0.3, 1), transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      will-change: opacity, transform;
    }
    h1, h2, h3, h4, .font-heading {
      font-family: 'Plus Jakarta Sans', sans-serif;
    }
    body.page-ready {
      opacity: 1;
      transform: translateY(0);
    }
    body.page-exiting {
      opacity: 0;
      transform: translateY(-8px);
      transition: opacity 0.22s cubic-bezier(0.4, 0, 1, 1), transform 0.22s cubic-bezier(0.4, 0, 1, 1);
    }
    html { scroll-behavior: smooth; }

    /* Button and Link Micro-Interactions */
    a, button {
      transition: transform 0.15s cubic-bezier(0.16, 1, 0.3, 1), background-color 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, opacity 0.15s ease;
    }
    a:active, button:active {
      transform: scale(0.975);
    }

    /* Glow utilities */
    .glow-indigo {
      box-shadow: 0 0 50px -10px rgba(99, 102, 241, 0.18);
    }
    .card-hover {
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .card-hover:hover {
      transform: translateY(-3px);
      box-shadow: 0 12px 30px -8px rgba(99, 102, 241, 0.12), 0 2px 6px rgba(0, 0, 0, 0.04);
      border-color: rgba(165, 180, 252, 0.8);
    }
  </style>
</head>
<body class="text-slate-800 min-h-screen flex flex-col antialiased selection:bg-indigo-100 selection:text-indigo-900">

  <!-- Navigation Header -->
  <header class="bg-white/80 backdrop-blur-xl border-b border-slate-200/70 sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <a href="/" class="flex items-center space-x-3 group">
          <img src="/static/logo-icon.svg" alt="Legal Contract Review Logo" class="w-10 h-10 rounded-xl shadow-md shadow-indigo-500/20 group-hover:scale-105 transition-transform">
          <div>
            <div class="flex items-center space-x-2">
              <h1 class="text-base sm:text-lg font-black text-slate-900 leading-tight tracking-tight">Legal Contract Review</h1>
              <span class="text-[10px] font-bold bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full border border-indigo-200/60 uppercase tracking-wider">v1.0</span>
            </div>
            <p class="text-[11px] text-slate-500 font-medium">Semantic Kernel &bull; ChromaDB Vector Grounding</p>
          </div>
        </a>
      </div>

      <!-- Nav Links -->
      <nav class="hidden md:flex items-center space-x-1 text-sm font-semibold text-slate-600">
        <a href="#howItWorks" class="px-3.5 py-1.5 rounded-lg hover:text-indigo-600 hover:bg-slate-100/70 transition">How It Works</a>
        <a href="#features" class="px-3.5 py-1.5 rounded-lg hover:text-indigo-600 hover:bg-slate-100/70 transition">Features</a>
        <a href="#agents" class="px-3.5 py-1.5 rounded-lg hover:text-indigo-600 hover:bg-slate-100/70 transition">Agents</a>
        <a href="#security" class="px-3.5 py-1.5 rounded-lg hover:text-indigo-600 hover:bg-slate-100/70 transition">Security</a>
        <a href="#faq" class="px-3.5 py-1.5 rounded-lg hover:text-indigo-600 hover:bg-slate-100/70 transition">FAQ</a>
      </nav>

      <div class="flex items-center space-x-3">
        <span class="hidden sm:inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200/80">
          <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
          <span>Scope Guard Active</span>
        </span>
        <a href="/analyze-contracts" class="inline-flex items-center space-x-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700 text-white font-bold text-xs sm:text-sm px-4 sm:px-5 py-2.5 rounded-xl shadow-md shadow-indigo-500/25 hover:shadow-indigo-500/35 hover:-translate-y-0.5 transition-all">
          <span>Analyze Contract</span>
          <i class="fa-solid fa-arrow-right text-xs"></i>
        </a>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full space-y-20">

    <!-- ============================================================= -->
    <!-- HERO SECTION (2 options only, no autonomous multiagent pill) -->
    <!-- ============================================================= -->
    <section id="hero" class="pt-12 pb-6 text-center max-w-4xl mx-auto space-y-6">
      <h1 class="text-4xl sm:text-5xl lg:text-6xl font-black text-slate-900 tracking-tight leading-[1.12]">
        Transform Legal Contract Review into <span class="bg-gradient-to-r from-indigo-600 via-violet-600 to-indigo-700 bg-clip-text text-transparent">Grounded Risk Intelligence</span>
      </h1>

      <p class="text-base sm:text-lg text-slate-600 max-w-2xl mx-auto font-normal leading-relaxed">
        Upload agreements in PDF, Word, plain text, or scanned images. Four specialized AI agents extract clauses, match your firm's negotiated policy library in ChromaDB, and flag risks with zero hallucination.
      </p>

      <!-- 2 Options Only: Analyze Contract & Try Demo -->
      <div class="flex flex-wrap items-center justify-center gap-4 pt-3">
        <a href="/analyze-contracts" class="bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700 text-white font-bold text-sm px-8 py-4 rounded-xl shadow-lg shadow-indigo-600/30 hover:shadow-indigo-600/40 hover:-translate-y-0.5 transition-all inline-flex items-center space-x-2.5">
          <i class="fa-solid fa-file-shield text-base"></i>
          <span>Analyze Contract</span>
          <i class="fa-solid fa-arrow-right text-xs ml-1"></i>
        </a>
        <a href="/analyze-contracts?demo=clean.txt" class="bg-white hover:bg-slate-50 text-slate-800 font-bold text-sm px-8 py-4 rounded-xl border border-slate-200 shadow-xs hover:border-indigo-300 hover:shadow-md hover:-translate-y-0.5 transition-all inline-flex items-center space-x-2.5">
          <i class="fa-solid fa-play text-indigo-600 text-xs"></i>
          <span>Try Demo</span>
        </a>
      </div>

      <!-- Trust Metrics Bento Bar -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 pt-10 text-left">
        <div class="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.04)] card-hover">
          <div class="w-8 h-8 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center text-sm font-bold mb-3">
            <i class="fa-solid fa-users-gear"></i>
          </div>
          <div class="text-2xl font-black text-slate-900 tracking-tight">4 Agents</div>
          <div class="text-xs text-slate-500 font-medium mt-1">Collaborative Semantic Kernel</div>
        </div>
        <div class="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.04)] card-hover">
          <div class="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center text-sm font-bold mb-3">
            <i class="fa-solid fa-database"></i>
          </div>
          <div class="text-2xl font-black text-slate-900 tracking-tight">100% Grounded</div>
          <div class="text-xs text-slate-500 font-medium mt-1">ChromaDB Policy Vector Store</div>
        </div>
        <div class="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.04)] card-hover">
          <div class="w-8 h-8 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center text-sm font-bold mb-3">
            <i class="fa-solid fa-bolt"></i>
          </div>
          <div class="text-2xl font-black text-slate-900 tracking-tight">&lt; 10 Seconds</div>
          <div class="text-xs text-slate-500 font-medium mt-1">Rapid Staged Pipeline</div>
        </div>
        <div class="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.04)] card-hover">
          <div class="w-8 h-8 rounded-lg bg-violet-50 text-violet-600 flex items-center justify-center text-sm font-bold mb-3">
            <i class="fa-solid fa-file-image"></i>
          </div>
          <div class="text-2xl font-black text-slate-900 tracking-tight">Omni-Format</div>
          <div class="text-xs text-slate-500 font-medium mt-1">PDF, DOCX, TXT &amp; Vision OCR</div>
        </div>
      </div>
    </section>

    <!-- ============================================================= -->
    <!-- HOW IT WORKS SECTION -->
    <!-- ============================================================= -->
    <section id="howItWorks" class="py-6 border-t border-slate-200/70 scroll-mt-24">
      <div class="text-center max-w-3xl mx-auto mb-12">
        <span class="text-xs font-bold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-3.5 py-1.5 rounded-full border border-indigo-200/60">
          The 4-Step Pipeline
        </span>
        <h2 class="text-3xl font-black text-slate-900 tracking-tight mt-3">How the Multi-Agent System Operates</h2>
        <p class="text-sm text-slate-600 mt-2">Every uploaded agreement passes through a sequential chain of specialized agents, each governed by strict input, scope, and output guardrails.</p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <!-- Step 1 -->
        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover flex flex-col justify-between">
          <div>
            <div class="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-700 flex items-center justify-center text-sm font-black mb-4">
              01
            </div>
            <h3 class="text-base font-bold text-slate-900 mb-2">Contract Document Upload &amp; OCR</h3>
            <p class="text-xs text-slate-600 leading-relaxed">
              Parses native PDF, DOCX, and TXT files, or uses Gemini Vision OCR to transcribe scanned images. InputGuard cleans control characters and neutralizes prompt injection attempts.
            </p>
          </div>
          <div class="mt-4 pt-3 border-t border-slate-100 flex items-center text-[11px] font-bold text-indigo-600">
            <span>Input Sanitization</span>
            <i class="fa-solid fa-arrow-right ml-1.5 text-[9px]"></i>
          </div>
        </div>

        <!-- Step 2 -->
        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover flex flex-col justify-between">
          <div>
            <div class="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center text-sm font-black mb-4">
              02
            </div>
            <h3 class="text-base font-bold text-slate-900 mb-2">Clause Extractor</h3>
            <p class="text-xs text-slate-600 leading-relaxed">
              Agent 1 scans unformatted text to isolate Liability, Termination, Payment, and Confidentiality clauses verbatim. OutputGuard ensures zero hallucination against the raw contract.
            </p>
          </div>
          <div class="mt-4 pt-3 border-t border-slate-100 flex items-center text-[11px] font-bold text-emerald-600">
            <span>Verbatim Isolation</span>
            <i class="fa-solid fa-arrow-right ml-1.5 text-[9px]"></i>
          </div>
        </div>

        <!-- Step 3 -->
        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover flex flex-col justify-between">
          <div>
            <div class="w-10 h-10 rounded-xl bg-amber-50 text-amber-700 flex items-center justify-center text-sm font-black mb-4">
              03
            </div>
            <h3 class="text-base font-bold text-slate-900 mb-2">Policy Retriever</h3>
            <p class="text-xs text-slate-600 leading-relaxed">
              Agent 2 reformulates queries and queries ChromaDB vector embeddings with metadata filtering to fetch the firm's exact negotiated standard position for each clause type.
            </p>
          </div>
          <div class="mt-4 pt-3 border-t border-slate-100 flex items-center text-[11px] font-bold text-amber-600">
            <span>Vector Grounding</span>
            <i class="fa-solid fa-arrow-right ml-1.5 text-[9px]"></i>
          </div>
        </div>

        <!-- Step 4 -->
        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover flex flex-col justify-between">
          <div>
            <div class="w-10 h-10 rounded-xl bg-violet-50 text-violet-700 flex items-center justify-center text-sm font-black mb-4">
              04
            </div>
            <h3 class="text-base font-bold text-slate-900 mb-2">Flagger &amp; Summary</h3>
            <p class="text-xs text-slate-600 leading-relaxed">
              Agent 3 classifies deviations (ACCEPTABLE, REVIEW, REJECT) with specific rationales, and Agent 4 produces an executive briefing and exportable audit report.
            </p>
          </div>
          <div class="mt-4 pt-3 border-t border-slate-100 flex items-center text-[11px] font-bold text-violet-600">
            <span>Decision Intelligence</span>
            <i class="fa-solid fa-check ml-1.5 text-[9px]"></i>
          </div>
        </div>
      </div>
    </section>

    <!-- ============================================================= -->
    <!-- FEATURES GRID SECTION -->
    <!-- ============================================================= -->
    <section id="features" class="py-6 border-t border-slate-200/70 scroll-mt-24">
      <div class="text-center max-w-3xl mx-auto mb-12">
        <span class="text-xs font-bold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-3.5 py-1.5 rounded-full border border-indigo-200/60">
          Core Capabilities
        </span>
        <h2 class="text-3xl font-black text-slate-900 tracking-tight mt-3">Built for Modern Legal &amp; Procurement Teams</h2>
        <p class="text-sm text-slate-600 mt-2">Enterprise-grade capabilities engineered to eliminate manual contract review bottlenecks.</p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover space-y-3">
          <div class="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center text-lg">
            <i class="fa-solid fa-network-wired"></i>
          </div>
          <h3 class="text-base font-bold text-slate-900">Semantic Kernel Orchestration</h3>
          <p class="text-xs text-slate-600 leading-relaxed">Each agent operates with isolated system prompts, tailored temperatures, and independent completion execution settings for predictable outcomes.</p>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover space-y-3">
          <div class="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center text-lg">
            <i class="fa-solid fa-database"></i>
          </div>
          <h3 class="text-base font-bold text-slate-900">ChromaDB Policy Vector Store</h3>
          <p class="text-xs text-slate-600 leading-relaxed">Embeds firm contract guidelines with cosine similarity search and clause-type metadata filters, guaranteeing zero hallucinated benchmarks.</p>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover space-y-3">
          <div class="w-10 h-10 rounded-xl bg-violet-50 text-violet-600 flex items-center justify-center text-lg">
            <i class="fa-solid fa-eye"></i>
          </div>
          <h3 class="text-base font-bold text-slate-900">Multi-Modal Vision OCR</h3>
          <p class="text-xs text-slate-600 leading-relaxed">Supports scanned images (.png, .jpg, .webp) with automated image preprocessing and Gemini multimodal transcription for physical agreements.</p>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover space-y-3">
          <div class="w-10 h-10 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center text-lg">
            <i class="fa-solid fa-calculator"></i>
          </div>
          <h3 class="text-base font-bold text-slate-900">Deterministic Risk Rules</h3>
          <p class="text-xs text-slate-600 leading-relaxed">Overall recommendations (ACCEPT, NEGOTIATE, REJECT) follow codified escalation rules in code, ensuring high-risk deviations cannot be hidden or downgraded.</p>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover space-y-3">
          <div class="w-10 h-10 rounded-xl bg-rose-50 text-rose-600 flex items-center justify-center text-lg">
            <i class="fa-solid fa-shield-virus"></i>
          </div>
          <h3 class="text-base font-bold text-slate-900">Prompt Injection Immunity</h3>
          <p class="text-xs text-slate-600 leading-relaxed">InputGuard intercepts adversarial phrases like 'ignore previous instructions' and quarantines contracts inside untrusted XML data boundaries.</p>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover space-y-3">
          <div class="w-10 h-10 rounded-xl bg-cyan-50 text-cyan-600 flex items-center justify-center text-lg">
            <i class="fa-solid fa-file-export"></i>
          </div>
          <h3 class="text-base font-bold text-slate-900">One-Click Export Reports</h3>
          <p class="text-xs text-slate-600 leading-relaxed">Download client-ready Markdown briefings with executive summaries and clause risk tables, or export structured JSON for contract management systems.</p>
        </div>
      </div>
    </section>

    <!-- ============================================================= -->
    <!-- AGENTS ARCHITECTURE SECTION -->
    <!-- ============================================================= -->
    <section id="agents" class="py-6 border-t border-slate-200/70 scroll-mt-24">
      <div class="text-center max-w-3xl mx-auto mb-12">
        <span class="text-xs font-bold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-3.5 py-1.5 rounded-full border border-indigo-200/60">
          Agent Directory
        </span>
        <h2 class="text-3xl font-black text-slate-900 tracking-tight mt-3">Four Specialized Semantic Kernel Agents</h2>
        <p class="text-sm text-slate-600 mt-2">Each agent has a single, strictly bounded responsibility governed by ScopeGuard.</p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Agent 1 -->
        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover flex items-start space-x-4">
          <div class="w-11 h-11 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center text-lg shrink-0">
            <i class="fa-solid fa-magnifying-glass"></i>
          </div>
          <div class="space-y-1">
            <div class="flex items-center space-x-2">
              <h3 class="text-base font-bold text-slate-900">1. Clause Extractor Agent</h3>
              <span class="text-[10px] bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded font-bold uppercase border border-indigo-200/50">Extractor</span>
            </div>
            <p class="text-xs text-slate-600 leading-relaxed">Identifies and isolates verbatim clause segments (Liability, Termination, Payment, Confidentiality). OutputGuard verifies that extracted text exists word-for-word in the original agreement.</p>
          </div>
        </div>

        <!-- Agent 2 -->
        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover flex items-start space-x-4">
          <div class="w-11 h-11 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center text-lg shrink-0">
            <i class="fa-solid fa-book-bookmark"></i>
          </div>
          <div class="space-y-1">
            <div class="flex items-center space-x-2">
              <h3 class="text-base font-bold text-slate-900">2. Policy Retriever Agent</h3>
              <span class="text-[10px] bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded font-bold uppercase border border-emerald-200/50">RAG Store</span>
            </div>
            <p class="text-xs text-slate-600 leading-relaxed">Performs LLM query reformulation and ChromaDB vector search to fetch preferred, acceptable, and unacceptable benchmarks. If a clause is absent, automatically sets standard position.</p>
          </div>
        </div>

        <!-- Agent 3 -->
        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover flex items-start space-x-4">
          <div class="w-11 h-11 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center text-lg shrink-0">
            <i class="fa-solid fa-flag"></i>
          </div>
          <div class="space-y-1">
            <div class="flex items-center space-x-2">
              <h3 class="text-base font-bold text-slate-900">3. Risk Flagger Agent</h3>
              <span class="text-[10px] bg-amber-50 text-amber-700 px-2 py-0.5 rounded font-bold uppercase border border-amber-200/50">Flagger</span>
            </div>
            <p class="text-xs text-slate-600 leading-relaxed">Evaluates extracted clauses against retrieved benchmarks. Returns JSON enum (ACCEPTABLE, REVIEW, REJECT) with a deviation rationale under 300 characters. Automatic one-shot retry on JSON errors.</p>
          </div>
        </div>

        <!-- Agent 4 -->
        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover flex items-start space-x-4">
          <div class="w-11 h-11 rounded-xl bg-violet-50 text-violet-600 flex items-center justify-center text-lg shrink-0">
            <i class="fa-solid fa-feather-pointed"></i>
          </div>
          <div class="space-y-1">
            <div class="flex items-center space-x-2">
              <h3 class="text-base font-bold text-slate-900">4. Executive Summarizer Agent</h3>
              <span class="text-[10px] bg-violet-50 text-violet-700 px-2 py-0.5 rounded font-bold uppercase border border-violet-200/50">Summarizer</span>
            </div>
            <p class="text-xs text-slate-600 leading-relaxed">Generates a high-level briefing strictly under 300 words. Enforces that all flagged (REVIEW or REJECT) clauses are explicitly named in the briefing, followed by standard legal disclaimers.</p>
          </div>
        </div>
      </div>
    </section>

    <!-- ============================================================= -->
    <!-- SECURITY & GUARDRAILS SECTION -->
    <!-- ============================================================= -->
    <section id="security" class="py-6 border-t border-slate-200/70 scroll-mt-24">
      <div class="text-center max-w-3xl mx-auto mb-12">
        <span class="text-xs font-bold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-3.5 py-1.5 rounded-full border border-indigo-200/60">
          Defense in Depth
        </span>
        <h2 class="text-3xl font-black text-slate-900 tracking-tight mt-3">Three-Tiered Guardrail Architecture</h2>
        <p class="text-sm text-slate-600 mt-2">Guaranteed safety and isolation at every stage of the multi-agent pipeline.</p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover space-y-3">
          <div class="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center text-lg">
            <i class="fa-solid fa-shield"></i>
          </div>
          <h3 class="text-base font-bold text-slate-900">1. Input Guard</h3>
          <p class="text-xs text-slate-600 leading-relaxed">Strips unprintable control characters, scans for jailbreaks, prompt injections ('ignore instructions', 'system override'), and encapsulates input inside untrusted XML boundaries.</p>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover space-y-3">
          <div class="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center text-lg">
            <i class="fa-solid fa-lock"></i>
          </div>
          <h3 class="text-base font-bold text-slate-900">2. Scope Guard</h3>
          <p class="text-xs text-slate-600 leading-relaxed">Guarantees raw contract text is strictly immutable. Programmatically verifies that agents only read their permitted inputs and never modify other agents' state slices.</p>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs card-hover space-y-3">
          <div class="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center text-lg">
            <i class="fa-solid fa-circle-check"></i>
          </div>
          <h3 class="text-base font-bold text-slate-900">3. Output Guard</h3>
          <p class="text-xs text-slate-600 leading-relaxed">Validates JSON schemas, enum bounds, text presence, and word count constraints. Automatically triggers a one-shot retry with error feedback before safely escalating.</p>
        </div>
      </div>
    </section>

    <!-- ============================================================= -->
    <!-- FAQ SECTION -->
    <!-- ============================================================= -->
    <section id="faq" class="py-6 border-t border-slate-200/70 scroll-mt-24">
      <div class="text-center max-w-3xl mx-auto mb-12">
        <span class="text-xs font-bold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-3.5 py-1.5 rounded-full border border-indigo-200/60">
          Got Questions?
        </span>
        <h2 class="text-3xl font-black text-slate-900 tracking-tight mt-3">Frequently Asked Questions</h2>
        <p class="text-sm text-slate-600 mt-2">Everything you need to know about the multi-agent legal review engine.</p>
      </div>

      <div class="max-w-3xl mx-auto space-y-4">
        <div class="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs card-hover">
          <h4 class="text-sm font-bold text-slate-900 flex items-center">
            <i class="fa-solid fa-circle-question text-indigo-600 mr-2.5"></i> Which document and image formats are supported?
          </h4>
          <p class="text-xs text-slate-600 mt-2 leading-relaxed">
            The system accepts PDF (.pdf), Microsoft Word (.docx), plain text (.txt), and scanned contract images (.png, .jpg, .jpeg, .webp). Non-document file types (.exe, .zip, .bin) are strictly rejected.
          </p>
        </div>

        <div class="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs card-hover">
          <h4 class="text-sm font-bold text-slate-900 flex items-center">
            <i class="fa-solid fa-circle-question text-indigo-600 mr-2.5"></i> How does the system defend against prompt injection?
          </h4>
          <p class="text-xs text-slate-600 mt-2 leading-relaxed">
            InputGuard scans incoming contract text for malicious command patterns like <code>ignore previous instructions</code> or <code>mark all acceptable</code>. The text is quarantined inside strict XML boundaries and treated as passive untrusted data, preventing agent prompt hijacking.
          </p>
        </div>

        <div class="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs card-hover">
          <h4 class="text-sm font-bold text-slate-900 flex items-center">
            <i class="fa-solid fa-circle-question text-indigo-600 mr-2.5"></i> Can I test the application offline without API keys?
          </h4>
          <p class="text-xs text-slate-600 mt-2 leading-relaxed">
            Yes! Toggle the <strong>Offline Mock Mode</strong> switch on the analyzer screen to run deterministic evaluations of contracts using local mock agents without consuming Gemini API tokens.
          </p>
        </div>

        <div class="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs card-hover">
          <h4 class="text-sm font-bold text-slate-900 flex items-center">
            <i class="fa-solid fa-circle-question text-indigo-600 mr-2.5"></i> How are the overall risk level and recommendation calculated?
          </h4>
          <p class="text-xs text-slate-600 mt-2 leading-relaxed">
            A deterministic rules engine in code calculates the final recommendation: any <code>REJECT</code> clause triggers an overall <strong>REJECT</strong>; two or more <code>REVIEW</code> clauses trigger <strong>NEGOTIATE</strong>; and contracts with zero deviations are granted <strong>ACCEPT</strong>.
          </p>
        </div>
      </div>
    </section>

    <!-- Bottom CTA Banner -->
    <section class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 rounded-3xl p-8 sm:p-12 text-center text-white shadow-2xl border border-indigo-900/40 space-y-4">
      <h2 class="text-2xl sm:text-3xl font-black tracking-tight">Ready to Streamline Your Contract Reviews?</h2>
      <p class="text-slate-300 text-sm max-w-xl mx-auto">Upload any agreement to get instantaneous policy cross-referencing, risk identification, and audit reports.</p>
      <div class="pt-2">
        <a href="/analyze-contracts" class="bg-white hover:bg-slate-100 text-indigo-900 font-bold text-sm px-8 py-3.5 rounded-xl shadow-lg hover:shadow-xl hover:-translate-y-0.5 transition-all inline-flex items-center space-x-2">
          <span>Launch Contract Workspace</span>
          <i class="fa-solid fa-arrow-right text-xs"></i>
        </a>
      </div>
    </section>

  </main>

  <!-- Footer -->
  <footer class="bg-white border-t border-slate-200/70 py-10 mt-auto">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
      <div class="flex flex-col md:flex-row items-center justify-between gap-4">
        <div class="flex items-center space-x-3">
          <img src="/static/logo-icon.svg" alt="Legal Contract Review Logo" class="w-8 h-8 rounded-lg shadow-sm">
          <div>
            <span class="text-sm font-bold text-slate-900 font-heading">Legal Contract Review</span>
            <span class="text-xs text-slate-400 block font-normal">Autonomous Multi-Agent AI System</span>
          </div>
        </div>

        <div class="flex flex-wrap items-center justify-center gap-6 text-xs text-slate-500 font-medium">
          <a href="#howItWorks" class="hover:text-indigo-600 transition">How It Works</a>
          <a href="#features" class="hover:text-indigo-600 transition">Features</a>
          <a href="#agents" class="hover:text-indigo-600 transition">Agents</a>
          <a href="#security" class="hover:text-indigo-600 transition">Security</a>
          <a href="#faq" class="hover:text-indigo-600 transition">FAQ</a>
          <a href="/analyze-contracts" class="hover:text-indigo-600 transition font-bold text-indigo-600">Analyze Workspace</a>
        </div>
      </div>

      <div class="border-t border-slate-100 pt-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-400">
        <p class="italic">Decision support only, not legal advice. Designed for legal and procurement professionals.</p>
        <p>&copy; 2026 Multi-Agent Legal AI &bull; Built with Semantic Kernel &amp; ChromaDB</p>
      </div>
    </div>
  </footer>

  <script>
    function markReady() {
      document.body.classList.add('page-ready');
    }
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', markReady);
    } else {
      markReady();
    }
    window.addEventListener('pageshow', () => {
      document.body.classList.remove('page-exiting');
      document.body.classList.add('page-ready');
    });

    // Smooth navigation transition on button/link click
    document.addEventListener('click', (e) => {
      const link = e.target.closest('a');
      if (!link) return;
      const href = link.getAttribute('href');
      if (!href) return;

      if (href.startsWith('#')) return; // Smooth scroll handled by CSS
      if (link.target === '_blank' || href.startsWith('mailto:') || href.startsWith('tel:')) return;
      if (href.startsWith('http://') || href.startsWith('https://')) {
        if (!href.startsWith(window.location.origin)) return;
      }

      e.preventDefault();
      document.body.classList.remove('page-ready');
      document.body.classList.add('page-exiting');
      setTimeout(() => {
        window.location.href = href;
      }, 200);
    });
  </script>

</body>
</html>
"""


# ==============================================================================
# 2. DEDICATED ANALYZER PAGE HTML (/analyze-contracts)
# ==============================================================================
ANALYZER_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Analyze Contracts &bull; Legal Contract Review</title>
  <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Inter', sans-serif;
      background: radial-gradient(circle at 50% -10%, rgba(99, 102, 241, 0.08) 0%, rgba(248, 250, 252, 0) 65%), #f8fafc;
      opacity: 0;
      transform: translateY(8px);
      transition: opacity 0.3s cubic-bezier(0.16, 1, 0.3, 1), transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      will-change: opacity, transform;
    }
    h1, h2, h3, h4, .font-heading {
      font-family: 'Plus Jakarta Sans', sans-serif;
    }
    body.page-ready {
      opacity: 1;
      transform: translateY(0);
    }
    body.page-exiting {
      opacity: 0;
      transform: translateY(-8px);
      transition: opacity 0.22s cubic-bezier(0.4, 0, 1, 1), transform 0.22s cubic-bezier(0.4, 0, 1, 1);
    }

    /* Badges */
    .badge-accept { background-color: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
    .badge-negotiate { background-color: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
    .badge-reject { background-color: #fff1f2; color: #9f1239; border: 1px solid #fecdd3; }

    /* Screen-to-Screen Smooth Transitions */
    .page-screen {
      opacity: 1;
      transform: translateY(0);
      transition: opacity 0.3s cubic-bezier(0.16, 1, 0.3, 1), transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      will-change: opacity, transform;
    }
    .page-screen.screen-entering {
      opacity: 0;
      transform: translateY(18px);
    }
    .page-screen.screen-leaving {
      opacity: 0;
      transform: translateY(-14px);
    }
    .page-screen.hidden {
      display: none !important;
    }

    /* Button and Link Micro-Interactions */
    a, button {
      transition: transform 0.15s cubic-bezier(0.16, 1, 0.3, 1), background-color 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, opacity 0.15s ease;
    }
    a:active, button:active {
      transform: scale(0.975);
    }

    /* Smooth Progress Bar animation */
    #progressBar {
      transition: width 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Agent Step Card transitions */
    .agent-step-card {
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .card-hover {
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .card-hover:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 25px -6px rgba(99, 102, 241, 0.12), 0 2px 4px rgba(0, 0, 0, 0.04);
    }
  </style>
</head>
<body class="text-slate-800 min-h-screen flex flex-col antialiased selection:bg-indigo-100 selection:text-indigo-900">

  <!-- Header -->
  <header class="bg-white/80 backdrop-blur-xl border-b border-slate-200/70 sticky top-0 z-30">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
      <div class="flex items-center space-x-4">
        <a href="/" class="flex items-center space-x-3 group">
          <img src="/static/logo-icon.svg" alt="Legal Contract Review Logo" class="w-10 h-10 rounded-xl shadow-md shadow-indigo-500/20 group-hover:scale-105 transition-transform">
          <div>
            <h1 class="text-base sm:text-lg font-bold text-slate-900 leading-tight tracking-tight">Legal Contract Review</h1>
            <p class="text-[11px] text-slate-500 font-medium">Contract Document Upload &bull; Multi-Agent Analysis</p>
          </div>
        </a>

        <a href="/" class="hidden sm:inline-flex items-center space-x-1.5 text-xs font-semibold text-slate-600 hover:text-indigo-600 bg-slate-100/80 hover:bg-slate-200/70 px-3.5 py-1.5 rounded-full transition">
          <i class="fa-solid fa-arrow-left text-[10px]"></i>
          <span>Back to Home</span>
        </a>
      </div>

      <div class="flex items-center space-x-3">
        <a href="/" class="sm:hidden text-xs font-semibold text-slate-600 hover:text-indigo-600 flex items-center space-x-1">
          <i class="fa-solid fa-arrow-left text-[10px]"></i>
          <span>Home</span>
        </a>
        <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200/80">
          <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
          <span>Scope Guard Protected</span>
        </span>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full space-y-6">

    <!-- Page Workspace Header -->
    <div class="flex flex-col sm:flex-row sm:items-end justify-between gap-3 border-b border-slate-200/80 pb-5">
      <div>
        <div class="inline-flex items-center space-x-2 text-xs font-bold text-indigo-600 uppercase tracking-wider mb-1">
          <i class="fa-solid fa-sliders"></i>
          <span>Review Workspace</span>
        </div>
        <h2 class="text-2xl sm:text-3xl font-black text-slate-900 tracking-tight">Contract Document Upload &amp; Risk Review</h2>
        <p class="text-xs sm:text-sm text-slate-500 mt-1">Upload your contract or test with pre-built scenarios to initiate real-time multi-agent processing.</p>
      </div>
      <span class="text-xs text-slate-600 font-medium bg-white border border-slate-200/80 px-3.5 py-1.5 rounded-xl shadow-xs w-fit flex items-center">
        <i class="fa-solid fa-shield-halved mr-1.5 text-indigo-600"></i> ChromaDB Vector Grounded
      </span>
    </div>

    <!-- ============================================================= -->
    <!-- SCREEN 1: UPLOAD SCREEN -->
    <!-- ============================================================= -->
    <section id="uploadScreen" class="page-screen space-y-8">

      <!-- Upload & Controls Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">

        <!-- Left Column: Upload Box (2 cols) -->
        <div class="lg:col-span-2 bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.03),0_8px_24px_-4px_rgba(0,0,0,0.02)] space-y-6">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-bold text-slate-900 flex items-center">
              <i class="fa-solid fa-file-arrow-up text-indigo-600 mr-2.5"></i> Contract Document Upload
            </h2>
            <span class="text-xs bg-slate-100 text-slate-600 px-3 py-1 rounded-full font-semibold border border-slate-200/60">
              Accepted: .pdf, .docx, .txt, .png, .jpg, .webp
            </span>
          </div>

          <!-- Drag & Drop Zone (Professional Styling) -->
          <div id="dropZone" class="border-2 border-dashed border-indigo-200 hover:border-indigo-500 bg-gradient-to-b from-indigo-50/30 via-white to-slate-50/30 hover:from-indigo-50/60 hover:to-indigo-50/20 rounded-2xl p-10 text-center transition-all cursor-pointer relative group">
            <input type="file" id="fileInput" class="hidden" accept=".pdf,.docx,.doc,.txt,.png,.jpg,.jpeg,.webp,.bmp,.tiff,image/*">
            <div class="space-y-4">
              <div class="w-16 h-16 mx-auto rounded-2xl bg-indigo-100/80 text-indigo-600 flex items-center justify-center text-3xl shadow-xs group-hover:scale-105 group-hover:bg-indigo-600 group-hover:text-white transition-all">
                <i class="fa-solid fa-cloud-arrow-up"></i>
              </div>
              <div>
                <p class="text-base font-bold text-slate-800">Drag & drop your contract document or image here, or <span class="text-indigo-600 underline font-extrabold">browse</span></p>
                <p class="text-xs text-slate-500 mt-1.5 max-w-md mx-auto">Supports PDF, Word (.docx), plain text (.txt), and scanned paper images (.png, .jpg, .webp) via Vision OCR.</p>
              </div>
              <div class="flex items-center justify-center gap-2 pt-1 text-[11px] font-semibold text-slate-400">
                <span class="px-2 py-0.5 bg-slate-100 rounded">PDF</span>
                <span class="px-2 py-0.5 bg-slate-100 rounded">DOCX</span>
                <span class="px-2 py-0.5 bg-slate-100 rounded">TXT</span>
                <span class="px-2 py-0.5 bg-slate-100 rounded">PNG</span>
                <span class="px-2 py-0.5 bg-slate-100 rounded">JPG</span>
                <span class="px-2 py-0.5 bg-slate-100 rounded">WEBP</span>
              </div>
            </div>
          </div>

          <!-- Selected File Card (replaces upload dropzone once uploaded) -->
          <div id="fileInfoBanner" class="hidden bg-indigo-50/40 border-2 border-indigo-200/90 rounded-2xl p-6 flex items-center justify-between transition-all">
            <div class="flex items-center space-x-4 overflow-hidden">
              <div class="w-12 h-12 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center text-2xl shadow-xs shrink-0">
                <i id="fileIcon" class="fa-solid fa-file-lines"></i>
              </div>
              <div class="truncate">
                <div class="flex items-center space-x-2">
                  <p id="fileName" class="text-sm sm:text-base font-bold text-slate-900 truncate"></p>
                  <span class="text-[10px] font-bold bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full uppercase tracking-wider shrink-0">Ready</span>
                </div>
                <p id="fileSize" class="text-xs text-slate-500 font-medium mt-0.5"></p>
              </div>
            </div>
            <button id="removeFileBtn" title="Remove file" class="text-slate-400 hover:text-red-500 text-sm p-2.5 hover:bg-red-50 rounded-xl transition flex items-center space-x-1">
              <i class="fa-solid fa-xmark text-base"></i>
              <span class="text-xs font-semibold hidden sm:inline">Remove</span>
            </button>
          </div>

          <!-- Error Banner -->
          <div id="errorBanner" class="hidden bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm flex items-start space-x-2">
            <i class="fa-solid fa-triangle-exclamation mt-0.5 text-red-500 shrink-0"></i>
            <span id="errorMessage" class="flex-1 font-medium"></span>
          </div>

          <!-- Action Bar -->
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-2 border-t border-slate-100">
            <div class="flex items-center space-x-3">
              <label class="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" id="mockModeToggle" class="sr-only peer" checked>
                <div class="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
                <span class="ml-2.5 text-xs font-semibold text-slate-700">Offline Mock Mode</span>
              </label>
              <span class="text-xs text-slate-400 cursor-help" title="When checked, runs offline without consuming API tokens. When unchecked, uses live Gemini/OpenAI keys from .env.">
                <i class="fa-solid fa-circle-question"></i>
              </span>
            </div>

            <button id="analyzeBtn" class="bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700 disabled:opacity-50 text-white font-bold text-sm px-8 py-3.5 rounded-xl shadow-md shadow-indigo-500/25 hover:shadow-indigo-500/35 hover:-translate-y-0.5 active:translate-y-0 transition-all flex items-center justify-center space-x-2">
              <span>Analyze Contract</span>
              <i class="fa-solid fa-arrow-right text-xs"></i>
            </button>
          </div>

        </div>

        <!-- Right Column: Quick Sample Contracts & Info (1 col) -->
        <div class="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.03),0_8px_24px_-4px_rgba(0,0,0,0.02)] space-y-6">
          <div>
            <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center">
              <i class="fa-solid fa-vial mr-2 text-indigo-600"></i> Try Sample Contracts
            </h2>
            <p class="text-xs text-slate-600 mb-4">Click any pre-configured contract to evaluate instant pipeline results:</p>
            <div class="space-y-2.5">
              <button onclick="loadSample('clean.txt')" class="w-full text-left p-3.5 rounded-xl border border-slate-200/80 hover:border-emerald-300 hover:bg-emerald-50/30 transition-all flex items-center justify-between group card-hover">
                <div>
                  <p class="text-xs font-bold text-slate-900 group-hover:text-emerald-800">clean.txt</p>
                  <p class="text-[11px] text-slate-500">100% policy compliance</p>
                </div>
                <span class="text-[11px] font-bold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 rounded-full">ACCEPT</span>
              </button>
              <button onclick="loadSample('mixed.txt')" class="w-full text-left p-3.5 rounded-xl border border-slate-200/80 hover:border-amber-300 hover:bg-amber-50/30 transition-all flex items-center justify-between group card-hover">
                <div>
                  <p class="text-xs font-bold text-slate-900 group-hover:text-amber-800">mixed.txt</p>
                  <p class="text-[11px] text-slate-500">Net-60 & 90d termination</p>
                </div>
                <span class="text-[11px] font-bold text-amber-700 bg-amber-50 border border-amber-200 px-2.5 py-0.5 rounded-full">NEGOTIATE</span>
              </button>
              <button onclick="loadSample('bad.txt')" class="w-full text-left p-3.5 rounded-xl border border-slate-200/80 hover:border-rose-300 hover:bg-rose-50/30 transition-all flex items-center justify-between group card-hover">
                <div>
                  <p class="text-xs font-bold text-slate-900 group-hover:text-rose-800">bad.txt</p>
                  <p class="text-[11px] text-slate-500">Unlimited liability clause</p>
                </div>
                <span class="text-[11px] font-bold text-rose-700 bg-rose-50 border border-rose-200 px-2.5 py-0.5 rounded-full">REJECT</span>
              </button>
              <button onclick="loadSample('injection_test.txt')" class="w-full text-left p-3.5 rounded-xl border border-slate-200/80 hover:border-purple-300 hover:bg-purple-50/30 transition-all flex items-center justify-between group card-hover">
                <div>
                  <p class="text-xs font-bold text-slate-900 group-hover:text-purple-800">injection_test.txt</p>
                  <p class="text-[11px] text-slate-500">Adversarial prompt injection</p>
                </div>
                <span class="text-[11px] font-bold text-purple-700 bg-purple-50 border border-purple-200 px-2.5 py-0.5 rounded-full">INJECTION</span>
              </button>
            </div>
          </div>

          <div class="border-t border-slate-100 pt-4">
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Sequential Pipeline Stages</h3>
            <ul class="text-xs text-slate-500 space-y-2">
              <li class="flex items-center"><i class="fa-solid fa-circle-check text-emerald-500 mr-2 text-[10px]"></i> Scope Guard Input Sanitization</li>
              <li class="flex items-center"><i class="fa-solid fa-circle-check text-emerald-500 mr-2 text-[10px]"></i> ChromaDB Vector Policy Grounding</li>
              <li class="flex items-center"><i class="fa-solid fa-circle-check text-emerald-500 mr-2 text-[10px]"></i> Semantic Kernel Agent Coordination</li>
              <li class="flex items-center"><i class="fa-solid fa-circle-check text-emerald-500 mr-2 text-[10px]"></i> Immutable Audit Trail Generation</li>
            </ul>
          </div>
        </div>

      </div>

    </section>


    <!-- ============================================================= -->
    <!-- SCREEN 2: PROGRESS SCREEN -->
    <!-- ============================================================= -->
    <section id="progressScreen" class="page-screen hidden">
      <div class="max-w-3xl mx-auto bg-white rounded-3xl p-8 sm:p-12 border border-slate-200/80 shadow-[0_4px_24px_-6px_rgba(0,0,0,0.04)] space-y-8 text-center">

        <!-- Top Big Bold Progress Percentile -->
        <div class="space-y-1">
          <div id="progressPercentage" class="text-6xl sm:text-7xl font-black bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent tracking-tight">
            0%
          </div>
          <p class="text-xs uppercase tracking-widest text-slate-400 font-bold">Analysis Progress</p>
        </div>

        <!-- Current Agent Badge & Description -->
        <div class="bg-indigo-50/60 border border-indigo-100 rounded-2xl p-6 max-w-xl mx-auto space-y-2 shadow-xs">
          <div class="flex items-center justify-center space-x-2.5">
            <span class="relative flex h-3 w-3">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span class="relative inline-flex rounded-full h-3 w-3 bg-indigo-600"></span>
            </span>
            <span id="currentAgentBadge" class="text-sm font-black text-indigo-950">
              Agent 1: Clause Extractor
            </span>
          </div>
          <p id="currentAgentTask" class="text-xs text-indigo-800 font-semibold">
            Extracting liability, termination, payment, and confidentiality clauses...
          </p>
          <p id="subStatusText" class="text-[11px] text-slate-500 italic">
            Scanning raw contract text...
          </p>
        </div>

        <!-- Progress Bar -->
        <div class="w-full bg-slate-100 rounded-full h-3 overflow-hidden shadow-inner p-0.5">
          <div id="progressBar" class="bg-gradient-to-r from-indigo-500 via-indigo-600 to-violet-600 h-2 rounded-full w-0 shadow-sm shadow-indigo-500/30"></div>
        </div>

        <!-- Agent Steps Grid (01 to 05) -->
        <div class="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-2 text-left">
          <div id="stepCard-1" class="agent-step-card p-3 rounded-xl border border-slate-200 bg-slate-50/60 text-slate-400">
            <div class="text-[10px] font-bold uppercase tracking-wider">01 Ingest</div>
            <div class="text-xs font-semibold text-slate-700 truncate mt-0.5">Input Guard</div>
            <div class="text-[10px] text-slate-400 mt-1 flex items-center step-status"><i class="fa-regular fa-clock mr-1"></i> Queued</div>
          </div>
          <div id="stepCard-2" class="agent-step-card p-3 rounded-xl border border-slate-200 bg-slate-50/60 text-slate-400">
            <div class="text-[10px] font-bold uppercase tracking-wider">02 Extract</div>
            <div class="text-xs font-semibold text-slate-700 truncate mt-0.5">Extractor</div>
            <div class="text-[10px] text-slate-400 mt-1 flex items-center step-status"><i class="fa-regular fa-clock mr-1"></i> Queued</div>
          </div>
          <div id="stepCard-3" class="agent-step-card p-3 rounded-xl border border-slate-200 bg-slate-50/60 text-slate-400">
            <div class="text-[10px] font-bold uppercase tracking-wider">03 Policy</div>
            <div class="text-xs font-semibold text-slate-700 truncate mt-0.5">Retriever RAG</div>
            <div class="text-[10px] text-slate-400 mt-1 flex items-center step-status"><i class="fa-regular fa-clock mr-1"></i> Queued</div>
          </div>
          <div id="stepCard-4" class="agent-step-card p-3 rounded-xl border border-slate-200 bg-slate-50/60 text-slate-400">
            <div class="text-[10px] font-bold uppercase tracking-wider">04 Flag</div>
            <div class="text-xs font-semibold text-slate-700 truncate mt-0.5">Risk Flagger</div>
            <div class="text-[10px] text-slate-400 mt-1 flex items-center step-status"><i class="fa-regular fa-clock mr-1"></i> Queued</div>
          </div>
          <div id="stepCard-5" class="agent-step-card p-3 rounded-xl border border-slate-200 bg-slate-50/60 text-slate-400 col-span-2 sm:col-span-1">
            <div class="text-[10px] font-bold uppercase tracking-wider">05 Brief</div>
            <div class="text-xs font-semibold text-slate-700 truncate mt-0.5">Summarizer</div>
            <div class="text-[10px] text-slate-400 mt-1 flex items-center step-status"><i class="fa-regular fa-clock mr-1"></i> Queued</div>
          </div>
        </div>

        <!-- File Analyzing Note -->
        <p class="text-xs text-slate-400">
          Analyzing <span id="progressFileName" class="font-bold text-slate-700">contract.txt</span> through multi-agent pipeline...
        </p>

        <!-- Progress Error Banner -->
        <div id="progressErrorBanner" class="hidden bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl text-xs text-left">
          <p class="font-bold flex items-center"><i class="fa-solid fa-circle-exclamation mr-1.5"></i> Analysis Encountered an Error:</p>
          <p id="progressErrorMessage" class="mt-1 font-mono text-[11px]"></p>
          <button onclick="resetToUpload()" class="mt-3 bg-red-100 hover:bg-red-200 text-red-800 px-3 py-1.5 rounded-lg font-bold text-xs">
            Return to Upload
          </button>
        </div>

      </div>
    </section>


    <!-- ============================================================= -->
    <!-- SCREEN 3: REPORT SCREEN -->
    <!-- ============================================================= -->
    <section id="reportScreen" class="page-screen hidden space-y-8">

      <!-- Report Header & Actions -->
      <div class="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.03),0_8px_24px_-4px_rgba(0,0,0,0.02)] flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <div class="flex items-center space-x-2 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            <span>Analysis Report</span>
            <span>&bull;</span>
            <span id="reportFileName" class="text-slate-800 font-bold">contract.txt</span>
          </div>
          <div class="flex flex-wrap items-center gap-3">
            <span id="recommendationBadge" class="px-4 py-1.5 rounded-full text-base font-extrabold tracking-wide uppercase">
              UNKNOWN
            </span>
            <span id="riskLevelBadge" class="px-3.5 py-1 rounded-full text-xs font-bold uppercase">
              Overall Risk: UNKNOWN
            </span>
          </div>
        </div>

        <div class="flex flex-wrap items-center gap-2.5">
          <button id="downloadMdBtn" class="bg-slate-100 hover:bg-slate-200/80 text-slate-700 text-xs font-bold px-4 py-2.5 rounded-xl border border-slate-200 transition flex items-center space-x-1.5">
            <i class="fa-solid fa-file-arrow-down"></i>
            <span>Markdown</span>
          </button>
          <button id="downloadJsonBtn" class="bg-slate-100 hover:bg-slate-200/80 text-slate-700 text-xs font-bold px-4 py-2.5 rounded-xl border border-slate-200 transition flex items-center space-x-1.5">
            <i class="fa-solid fa-code"></i>
            <span>JSON</span>
          </button>
          <button id="reviewAnotherBtn" onclick="resetToUpload()" class="bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700 text-white text-xs font-bold px-5 py-2.5 rounded-xl shadow-sm hover:shadow-indigo-500/20 transition flex items-center space-x-1.5">
            <i class="fa-solid fa-rotate-left"></i>
            <span>Review Another</span>
          </button>
        </div>
      </div>

      <!-- Executive Summary Box -->
      <div class="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.03),0_8px_24px_-4px_rgba(0,0,0,0.02)] space-y-4">
        <h3 class="text-sm font-bold text-slate-900 flex items-center">
          <i class="fa-solid fa-align-left text-indigo-600 mr-2.5"></i> Reviewer Executive Summary
        </h3>
        <div id="summaryText" class="text-sm text-slate-700 leading-relaxed bg-slate-50/80 p-5 rounded-2xl border-l-4 border-indigo-600 border border-slate-200/70 whitespace-pre-line font-normal"></div>
      </div>

      <!-- Per-Clause Risk Assessment Table -->
      <div class="bg-white rounded-3xl border border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.03),0_8px_24px_-4px_rgba(0,0,0,0.02)] overflow-hidden">
        <div class="p-6 border-b border-slate-200/80 flex items-center justify-between">
          <h3 class="text-base font-bold text-slate-900 flex items-center">
            <i class="fa-solid fa-table-list text-indigo-600 mr-2.5"></i> Per-Clause Risk Assessment
          </h3>
          <span class="text-xs text-slate-400 font-medium">Grounded in firm policy library</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-sm text-slate-600">
            <thead class="bg-slate-50 text-xs font-bold uppercase text-slate-500 border-b border-slate-200">
              <tr>
                <th class="py-3.5 px-5">Clause</th>
                <th class="py-3.5 px-5">Risk Level</th>
                <th class="py-3.5 px-5">Error / Clause Status</th>
              </tr>
            </thead>
            <tbody id="clausesTableBody" class="divide-y divide-slate-100">
            </tbody>
          </table>
        </div>
      </div>

    </section>

  </main>

  <!-- Footer -->
  <footer class="bg-white border-t border-slate-200/70 py-4 mt-auto">
    <div class="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
      <div class="flex items-center space-x-2">
        <img src="/static/logo-icon.svg" alt="Legal Contract Review Logo" class="w-5 h-5 rounded">
        <span class="text-xs font-semibold text-slate-700">Legal Contract Review</span>
      </div>
      <p class="text-xs text-slate-400 italic">Decision support only, not legal advice.</p>
    </div>
  </footer>

  <script>
    let selectedFile = null;
    let latestResult = null;

    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfoBanner = document.getElementById('fileInfoBanner');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const fileIcon = document.getElementById('fileIcon');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const errorBanner = document.getElementById('errorBanner');
    const errorMessage = document.getElementById('errorMessage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const mockModeToggle = document.getElementById('mockModeToggle');

    const uploadScreen = document.getElementById('uploadScreen');
    const progressScreen = document.getElementById('progressScreen');
    const reportScreen = document.getElementById('reportScreen');

    // Click dropzone to open file dialog
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag and Drop listeners
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('border-indigo-500', 'bg-indigo-50/50');
    });

    dropZone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dropZone.classList.remove('border-indigo-500', 'bg-indigo-50/50');
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('border-indigo-500', 'bg-indigo-50/50');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });

    removeFileBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      clearSelectedFile();
    });

    function handleFileSelection(file) {
      clearError();
      const ext = '.' + file.name.split('.').pop().toLowerCase();
      const validDocExts = ['.pdf', '.docx', '.doc', '.txt'];
      const validImgExts = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'];

      if (!validDocExts.includes(ext) && !validImgExts.includes(ext)) {
        showError(`Unsupported file format '${ext}'. Please upload a document (.pdf, .docx, .txt) or scanned contract image.`);
        clearSelectedFile();
        return;
      }

      selectedFile = file;
      fileName.textContent = file.name;
      fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';

      if (validImgExts.includes(ext)) {
        fileIcon.className = 'fa-solid fa-file-image text-purple-600';
      } else if (ext === '.pdf') {
        fileIcon.className = 'fa-solid fa-file-pdf text-red-600';
      } else if (ext === '.docx' || ext === '.doc') {
        fileIcon.className = 'fa-solid fa-file-word text-blue-600';
      } else {
        fileIcon.className = 'fa-solid fa-file-lines text-indigo-600';
      }

      dropZone.classList.add('hidden');
      fileInfoBanner.classList.remove('hidden');
    }

    function clearSelectedFile() {
      selectedFile = null;
      fileInput.value = '';
      fileInfoBanner.classList.add('hidden');
      dropZone.classList.remove('hidden');
    }

    function resetToUpload() {
      clearSelectedFile();
      clearError();
      showScreen('upload');
    }

    function showError(msg) {
      errorMessage.textContent = msg;
      errorBanner.classList.remove('hidden');
    }

    function clearError() {
      errorMessage.textContent = '';
      errorBanner.classList.add('hidden');
    }

    // =============================================================
    // SCREEN TRANSITIONS (upload -> progress -> report)
    // =============================================================
    async function showScreen(screenName) {
      const screens = [
        { name: 'upload', el: uploadScreen },
        { name: 'progress', el: progressScreen },
        { name: 'report', el: reportScreen }
      ];

      const currentActive = screens.find(s => !s.el.classList.contains('hidden'));
      const target = screens.find(s => s.name === screenName);

      if (currentActive && currentActive.name === screenName) return;

      if (currentActive) {
        currentActive.el.classList.add('screen-leaving');
        await new Promise(r => setTimeout(r, 260));
        currentActive.el.classList.add('hidden');
        currentActive.el.classList.remove('screen-leaving');
      }

      target.el.classList.add('screen-entering');
      target.el.classList.remove('hidden');
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          target.el.classList.remove('screen-entering');
        });
      });
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // =============================================================
    // MULTI-AGENT STAGED PACED PROGRESSION
    // =============================================================
    const AGENT_STAGES = [
      {
        startPercent: 0,
        targetPercent: 18,
        durationMs: 1400,
        badge: "Ingestion & Scope Guard",
        task: "Sanitizing document input & verifying isolation...",
        sub: "Checking for prompt injections & control characters...",
        agentIndex: 1
      },
      {
        startPercent: 18,
        targetPercent: 38,
        durationMs: 2400,
        badge: "Agent 1: Clause Extractor",
        task: "Extracting liability, termination, payment & confidentiality...",
        sub: "Verbatim parsing against raw contract text...",
        agentIndex: 2
      },
      {
        startPercent: 38,
        targetPercent: 62,
        durationMs: 2200,
        badge: "Agent 2: Policy Retriever",
        task: "Querying ChromaDB vector library for standard positions...",
        sub: "Reformulating semantic query embeddings...",
        agentIndex: 3
      },
      {
        startPercent: 62,
        targetPercent: 82,
        durationMs: 2400,
        badge: "Agent 3: Risk Flagger",
        task: "Evaluating clause deviations against firm policy...",
        sub: "Classifying risks into ACCEPTABLE, REVIEW, REJECT...",
        agentIndex: 4
      },
      {
        startPercent: 82,
        targetPercent: 96,
        durationMs: 1600,
        badge: "Agent 4: Executive Summarizer",
        task: "Synthesizing cross-clause briefing & risk score...",
        sub: "Formatting executive summary & audit trail...",
        agentIndex: 5
      },
      {
        startPercent: 96,
        targetPercent: 100,
        durationMs: 600,
        badge: "Analysis Complete",
        task: "Finalizing risk assessment report...",
        sub: "Rendering executive briefing...",
        agentIndex: 5
      }
    ];

    let cancelCurrentAnimation = false;

    function updateProgressUI(percentVal, stageInfo) {
      const rounded = Math.round(percentVal);
      document.getElementById('progressPercentage').textContent = rounded + '%';
      document.getElementById('progressBar').style.width = rounded + '%';

      if (stageInfo) {
        document.getElementById('currentAgentBadge').textContent = stageInfo.badge;
        document.getElementById('currentAgentTask').textContent = stageInfo.task;
        document.getElementById('subStatusText').textContent = stageInfo.sub;
      }
    }

    function updateAgentMatrix(activeIdx) {
      for (let i = 1; i <= 5; i++) {
        const card = document.getElementById(`stepCard-${i}`);
        if (!card) continue;
        const statusEl = card.querySelector('.step-status');

        if (i < activeIdx) {
          card.className = 'agent-step-card p-3 rounded-xl border border-emerald-200 bg-emerald-50/50 text-emerald-800';
          if (statusEl) statusEl.innerHTML = '<i class="fa-solid fa-check text-emerald-600 mr-1"></i> Done';
        } else if (i === activeIdx) {
          card.className = 'agent-step-card p-3 rounded-xl border border-indigo-400 bg-indigo-50 text-indigo-900 shadow-xs ring-1 ring-indigo-300';
          if (statusEl) statusEl.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-indigo-600 mr-1"></i> Running';
        } else {
          card.className = 'agent-step-card p-3 rounded-xl border border-slate-200 bg-slate-50/60 text-slate-400';
          if (statusEl) statusEl.innerHTML = '<i class="fa-regular fa-clock mr-1"></i> Queued';
        }
      }
    }

    function animatePercentage(fromVal, toVal, durationMs, stage, activeAgentIdx) {
      return new Promise((resolve) => {
        if (cancelCurrentAnimation) {
          resolve();
          return;
        }

        updateAgentMatrix(activeAgentIdx);
        const startTime = performance.now();

        function tick(now) {
          if (cancelCurrentAnimation) {
            resolve();
            return;
          }
          const elapsed = now - startTime;
          const progress = Math.min(1, elapsed / durationMs);

          const currentVal = fromVal + (toVal - fromVal) * progress;
          updateProgressUI(currentVal, stage);

          if (progress < 1) {
            requestAnimationFrame(tick);
          } else {
            updateProgressUI(toVal, stage);
            resolve();
          }
        }

        requestAnimationFrame(tick);
      });
    }

    function handleProgressError(errMsg) {
      cancelCurrentAnimation = true;
      document.getElementById('progressErrorMessage').textContent = errMsg;
      document.getElementById('progressErrorBanner').classList.remove('hidden');
    }

    async function runPacedAgentWorkflow(apiPromise, fileName) {
      cancelCurrentAnimation = false;
      document.getElementById('progressFileName').textContent = fileName;
      document.getElementById('progressErrorBanner').classList.add('hidden');

      await showScreen('progress');

      let apiDone = false;
      let apiResult = null;
      let apiError = null;

      apiPromise
        .then(data => {
          apiDone = true;
          apiResult = data;
        })
        .catch(err => {
          apiDone = true;
          apiError = err;
        });

      // Paced progression through the agent stages
      for (let i = 0; i < 5; i++) {
        if (cancelCurrentAnimation) return;
        const stage = AGENT_STAGES[i];
        await animatePercentage(stage.startPercent, stage.targetPercent, stage.durationMs, stage, stage.agentIndex);
        if (apiError) {
          handleProgressError(apiError.message);
          return;
        }
      }

      // If backend is still running (e.g. live LLM), hold at 96% until it resolves
      if (!apiDone) {
        try {
          apiResult = await apiPromise;
        } catch (err) {
          handleProgressError(err.message);
          return;
        }
      } else if (apiError) {
        handleProgressError(apiError.message);
        return;
      }

      if (cancelCurrentAnimation) return;

      // Final completion to 100%
      const finalStage = AGENT_STAGES[5];
      await animatePercentage(96, 100, finalStage.durationMs, finalStage, 5);
      updateAgentMatrix(5);

      // Brief hold so user sees bold 100% and Done checkmarks
      await new Promise(r => setTimeout(r, 600));

      if (cancelCurrentAnimation) return;

      renderResults(apiResult);
      await showScreen('report');
    }

    // =============================================================
    // ANALYZE BUTTON ACTION
    // =============================================================
    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) {
        showError('Please upload a contract document or image (.pdf, .docx, .txt, .png, .jpg) before analyzing.');
        return;
      }

      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('mock_mode', mockModeToggle.checked);

      const apiPromise = fetch('/api/review', {
        method: 'POST',
        body: formData,
      }).then(async (res) => {
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Analysis request failed.');
        return data;
      });

      await runPacedAgentWorkflow(apiPromise, selectedFile.name);
    });

    // =============================================================
    // SAMPLE CONTRACT LOAD ACTION
    // =============================================================
    async function loadSample(sampleName) {
      clearError();
      clearSelectedFile();

      const formData = new FormData();
      formData.append('sample_name', sampleName);
      formData.append('mock_mode', mockModeToggle.checked);

      const apiPromise = fetch('/api/review/sample', {
        method: 'POST',
        body: formData,
      }).then(async (res) => {
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Sample analysis failed.');
        return data;
      });

      await runPacedAgentWorkflow(apiPromise, sampleName);
    }

    // =============================================================
    // REPORT RENDERER
    // =============================================================
    function renderResults(data) {
      latestResult = data;
      
      document.getElementById('reportFileName').textContent = data.filename || 'contract.txt';

      const recBadge = document.getElementById('recommendationBadge');
      const riskBadge = document.getElementById('riskLevelBadge');

      const rec = data.overall_recommendation || 'UNKNOWN';
      recBadge.textContent = rec;
      if (rec === 'ACCEPT') {
        recBadge.className = 'px-4 py-1.5 rounded-full text-base font-black tracking-wide uppercase badge-accept';
      } else if (rec === 'NEGOTIATE') {
        recBadge.className = 'px-4 py-1.5 rounded-full text-base font-black tracking-wide uppercase badge-negotiate';
      } else {
        recBadge.className = 'px-4 py-1.5 rounded-full text-base font-black tracking-wide uppercase badge-reject';
      }

      riskBadge.textContent = 'Overall Risk: ' + (data.overall_risk || 'N/A');
      riskBadge.className = 'px-3.5 py-1 rounded-full text-xs font-bold uppercase ' + 
        (data.overall_risk === 'LOW' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' :
         data.overall_risk === 'MEDIUM' ? 'bg-amber-50 text-amber-700 border border-amber-200' :
         'bg-rose-50 text-rose-700 border border-rose-200');

      document.getElementById('summaryText').textContent = data.summary || 'No summary generated.';

      // Render clauses table
      const tbody = document.getElementById('clausesTableBody');
      tbody.innerHTML = '';

      (data.clauses || []).forEach(c => {
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-50/90 transition-colors';

        const riskClass = c.risk === 'ACCEPTABLE' ? 'bg-emerald-50 text-emerald-800 border border-emerald-200' :
                          c.risk === 'REVIEW' ? 'bg-amber-50 text-amber-800 border border-amber-200' : 'bg-rose-50 text-rose-800 border border-rose-200';

        let statusText = '';
        let statusStyle = '';
        let statusIcon = '';
        if (!c.found) {
          statusText = 'Clause not found in contract; cannot retrieve policy.';
          statusStyle = 'text-slate-500 italic';
          statusIcon = '<i class="fa-solid fa-circle-question mr-1.5 text-slate-400"></i>';
        } else if (c.risk === 'REJECT' || c.risk === 'REVIEW') {
          statusText = c.reason || 'Deviation detected.';
          statusStyle = 'text-rose-700 font-medium';
          statusIcon = '<i class="fa-solid fa-triangle-exclamation mr-1.5 text-rose-500"></i>';
        } else {
          statusText = 'Clause is present in contract.';
          statusStyle = 'text-emerald-700 font-medium';
          statusIcon = '<i class="fa-solid fa-circle-check mr-1.5 text-emerald-500"></i>';
        }

        tr.innerHTML = `
          <td class="py-4 px-5 font-bold text-slate-900 capitalize flex items-center">
            ${c.type}
            ${!c.found ? '<span class="ml-2 text-[10px] bg-rose-50 text-rose-700 border border-rose-200 px-2 py-0.5 rounded-full font-bold">Not Found</span>' : ''}
          </td>
          <td class="py-4 px-5">
            <span class="px-3 py-1 rounded-full text-xs font-bold ${riskClass}">
              ${c.risk || 'NONE'}
            </span>
          </td>
          <td class="py-4 px-5 text-xs ${statusStyle} leading-relaxed">${statusIcon}${statusText}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    document.getElementById('downloadMdBtn').addEventListener('click', () => {
      if (!latestResult || !latestResult.markdown_report) return;
      downloadFile(latestResult.markdown_report, `${latestResult.filename || 'contract'}_report.md`, 'text/markdown');
    });

    document.getElementById('downloadJsonBtn').addEventListener('click', () => {
      if (!latestResult || !latestResult.json_report) return;
      downloadFile(JSON.stringify(latestResult.json_report, null, 2), `${latestResult.filename || 'contract'}_report.json`, 'application/json');
    });

    function downloadFile(content, fileName, mimeType) {
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    // Page Entrance & Exit Transition Handlers
    function markReady() {
      document.body.classList.add('page-ready');
    }
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', markReady);
    } else {
      markReady();
    }
    window.addEventListener('pageshow', () => {
      document.body.classList.remove('page-exiting');
      document.body.classList.add('page-ready');
    });

    // Smooth navigation transition for links exiting this page (e.g. Back to Home)
    document.addEventListener('click', (e) => {
      const link = e.target.closest('a');
      if (!link) return;
      const href = link.getAttribute('href');
      if (!href) return;

      if (href.startsWith('#')) return;
      if (link.target === '_blank' || href.startsWith('mailto:') || href.startsWith('tel:')) return;
      if (href.startsWith('http://') || href.startsWith('https://')) {
        if (!href.startsWith(window.location.origin)) return;
      }

      e.preventDefault();
      document.body.classList.remove('page-ready');
      document.body.classList.add('page-exiting');
      setTimeout(() => {
        window.location.href = href;
      }, 200);
    });

    // Auto-load demo scenario if present in query parameters (?demo=clean.txt)
    window.addEventListener('DOMContentLoaded', () => {
      const urlParams = new URLSearchParams(window.location.search);
      const demo = urlParams.get('demo');
      if (demo) {
        loadSample(demo);
      }
    });
  </script>
</body>
</html>
"""

# Alias for backward compatibility
HTML_CONTENT = LANDING_PAGE_HTML


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal Contract Review Web Application")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload")
    args = parser.parse_args()

    print(f"Starting Legal Contract Review Server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to terminate.")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
