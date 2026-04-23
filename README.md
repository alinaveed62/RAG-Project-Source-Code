# RAG Chatbot for the KCL Informatics Student Handbook

A Retrieval Augmented Generation (RAG) chatbot that answers student
questions based on the Department of Informatics Student Handbook at
King's College London. The system scrapes the handbook from KEATS
(Moodle), processes and chunks the content, indexes it with FAISS and
BM25, and generates grounded answers using open-source LLMs served
locally by Ollama, fronted by a Flask web interface.

**Student:** Muhammad Ali Naveed
**Supervisor:** Dr Jeroen Keppens
**Module:** 6CCS3PRJ Final Year Individual Project
**Variant:** 2, Informatics Student Handbook

This repository contains the full source code, tests, and a small
pre-scraped data bundle (handbook chunks, FAISS index, and the 50-item
evaluation test set) so the chatbot can be run end-to-end without
requiring a KCL login to KEATS.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Running the chatbot](#3-running-the-chatbot)
4. [Re-scraping KEATS (optional)](#4-re-scraping-keats-optional)
5. [Rebuilding the FAISS index](#5-rebuilding-the-faiss-index)
6. [Project layout](#6-project-layout)
7. [Running the tests](#7-running-the-tests)
8. [Static analysis](#8-static-analysis)
9. [Running the evaluation suite](#9-running-the-evaluation-suite)
10. [Environment variables](#10-environment-variables)
11. [Configuration reference](#11-configuration-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [Tech stack](#13-tech-stack)
14. [Academic metadata and licence](#14-academic-metadata-and-licence)

---

## 1. Prerequisites

* Python 3.12 or newer. Tested on CPython 3.12 on macOS arm64 and Linux
  x86_64.
* Ollama 0.4.0 or newer, running locally. On macOS, install with
  `brew install ollama`. On Linux, use the installer script at
  `https://ollama.com/install.sh`.
* Roughly 6 GB of free disk space for the four Ollama models used in the
  evaluation (`gemma2:2b`, `mistral`, `llama3.2`, `phi3:mini`), plus
  about 500 MB for the sentence-transformers and cross-encoder caches
  downloaded on first use.
* 8 GB of RAM is comfortable for `gemma2:2b`; 16 GB or more is
  recommended if you want to swap in Mistral or Llama 3.2.
* A POSIX shell (bash or zsh) and `git`.

Re-scraping the handbook (Section 4) additionally requires a KCL account
with access to the configured KEATS course, and a Chromium install that
Selenium plus `webdriver-manager` can drive. Re-scraping is **not**
required for running the bundled chatbot.

## 2. Installation

```
git clone <repository-url>
cd RAG-Project-Source-Code

python3.12 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.lock
```

`requirements.lock` is a `pip-compile` output that pins every direct and
transitive dependency (including `mypy`, `ruff`, and `pip-tools`
itself), so the installed environment matches the one used for the
reported evaluation. To regenerate the lock file after editing
`requirements.in`:

```
pip install pip-tools
pip-compile --output-file=requirements.lock requirements.in
```

## 3. Running the chatbot

This repository ships the pre-scraped chunks
(`keats_scraper/data/chunks/chunks_for_embedding.jsonl`, 379 chunks)
and a pre-built FAISS index (`rag_pipeline/data/index/index.faiss` plus
`metadata.json`), so the chatbot launches without any KCL login.

```
# Terminal 1
ollama serve

# Terminal 2
ollama pull gemma2:2b                   # roughly 1.6 GB, one-off download
source venv/bin/activate                # if not already active
python run_local.py                     # starts on 127.0.0.1:5000
```

Open `http://localhost:5000` in a browser. The landing page has a
question box; answers render with inline `[Source: c##]` citations and
an expandable sources panel showing the retrieved chunks and the
handbook section each came from.

Useful `run_local.py` flags:

```
python run_local.py --build-index                      # force a rebuild of the FAISS index
python run_local.py --model mistral                    # swap the Ollama model (must be pulled)
python run_local.py --port 8000                        # bind to a different port
python run_local.py --host 0.0.0.0                     # expose on the LAN (no auth layer)
python run_local.py --ollama-url http://host:11434     # point at a remote Ollama
```

`run_local.py` checks that Ollama is responding at its base URL,
verifies that the chunks file is present, builds or loads the FAISS
index, connects the pipeline to Ollama, and starts Flask on
`127.0.0.1:5000`. Flask binds to loopback by default; the chatbot has
no authentication layer and no request rate limiting of its own, so
exposing it on `0.0.0.0` should only be done on trusted networks.

## 4. Re-scraping KEATS (optional)

This section is only needed to regenerate
`chunks_for_embedding.jsonl` from a fresh KEATS scrape. The bundled
chunks are sufficient for running the chatbot and for every test in
the suite.

```
cp keats_scraper/.env.example keats_scraper/.env
# Edit keats_scraper/.env and at minimum set COOKIE_ENCRYPTION_KEY.

cd keats_scraper
python main.py login                    # opens Chrome for KCL SSO and 2FA
python main.py all                      # scrape, then chunk; resumes from checkpoints
```

The full `keats_scraper` CLI:

| Command | Purpose |
|---|---|
| `python main.py login` | Opens Chrome, waits for the user to complete SSO and 2FA, then saves encrypted cookies to `.cookies`. |
| `python main.py logout` | Deletes the saved session cookies. |
| `python main.py scrape` | Discovers resources, fetches HTML and PDFs, cleans them, and writes `data/processed/documents.jsonl`. The `--resume` flag continues a previous run from its checkpoint. |
| `python main.py process` | Reads the processed documents and emits `data/chunks/chunks_for_embedding.jsonl` and `data/chunks/handbook_chunks.jsonl`. |
| `python main.py all` | Runs `scrape` followed by `process`. |
| `python main.py status` | Prints a progress summary (discovered, processed, failed). |
| `python main.py validate` | Runs `ContentValidator` over the processed documents and prints quality warnings. |
| `python main.py clear` | Deletes every file under `keats_scraper/data/` and the checkpoint store. |

Typical timings on a reliable network: the `login` command takes well
under a minute, and `all` completes in 10 to 30 minutes at the 20
requests-per-minute rate limit, producing roughly 185 documents and
379 chunks.

## 5. Rebuilding the FAISS index

Rebuild the index whenever:

1. You have re-scraped KEATS and have a fresh
   `chunks_for_embedding.jsonl`.
2. You change `RAGConfig.embedding_model` (for example, swapping
   `multi-qa-MiniLM-L6-cos-v1` for `all-mpnet-base-v2`). The index's
   vector dimension must match the encoder.
3. You change `CHUNK_SIZE` or `CHUNK_OVERLAP` in the scraper.

The simplest path is:

```
python run_local.py --build-index
```

This uses the default `RAGConfig` (`multi-qa-MiniLM-L6-cos-v1`, 384
dimensions). Index construction takes about 30 seconds on an M2 Pro
for the 379-chunk corpus. The outputs land at
`rag_pipeline/data/index/index.faiss` (roughly 580 KB) and
`rag_pipeline/data/index/metadata.json` (roughly 700 KB).

Programmatic equivalent:

```python
from rag_pipeline.config import RAGConfig
from rag_pipeline.pipeline import RAGPipeline

pipeline = RAGPipeline(RAGConfig())
pipeline.build_index()
```

## 6. Project layout

```
RAG-Project-Source-Code/
    keats_scraper/       KEATS scraper: SSO auth, scraping, cleaning, chunking, CLI
        main.py          Click CLI entry point
        auth/            SessionManager and SSOHandler (Selenium, 2FA)
        scraper/         PageScraper, PDFHandler, CourseNavigator, RateLimiter
        processors/      Chunker, SemanticChunker, HTMLCleaner, TextNormalizer, ContentValidator
        storage/         CheckpointManager, JSONLExporter
        analyses/        coverage_report.py (mypy --strict)
        models/          Document, Chunk, and related dataclasses
        utils/           exceptions, logging_config
        tests/           scraper test suite
        data/chunks/     chunks_for_embedding.jsonl (bundled)
        data/processed/  documents.jsonl (bundled)

    rag_pipeline/        RAG pipeline: embeddings, retrieval, generation
        pipeline.py      RAGPipeline.build_index(), setup(), answer()
        config.py        RAGConfig Pydantic model (mypy --strict)
        models.py        RetrievalResult, RAGResponse, Citation
        embeddings/      ChunkEncoder, FAISSIndexBuilder
        retrieval/       FAISSRetriever, BM25Retriever, HybridRetriever, Reranker, QueryProcessor
        generation/      OllamaGenerator, PromptTemplates, CitationParser
        tests/           pipeline test suite
        data/index/      index.faiss + metadata.json (bundled)

    flask_app/           Web interface
        app.py           Flask application factory create_app(pipeline)
        routes.py        GET|POST / for question-answering, POST /feedback for ratings
        config.py        Flask configuration
        feedback/        logger.py: appends to feedback.jsonl
        templates/       base.html, index.html, answer.html
        static/          css/style.css, js/main.js
        tests/           Flask route and feedback tests

    evaluation/          Evaluation suite
        config.py        Report-directory paths (honours EVAL_REPORT_DIR)
        generate_static_figures.py   Distribution charts from qa_pairs.json
        metrics/         retrieval_metrics, answer_metrics, sgf, bootstrap, effect_size, evaluator
        experiments/     Eight ablation drivers plus run_all.py and post-hoc analyses
        test_set/        qa_pairs.json (50 hand-curated QA pairs)
        tests/           Evaluation test suite

    tests/               Top-level integration test for run_local.py
    scripts/check.sh     ruff plus mypy gate
    run_local.py         One-command startup: Ollama check, index build or load, Flask
    pyproject.toml       mypy per-module strict overrides and ruff rule selection
    requirements.in      Top-level dependency manifest
    requirements.lock    pip-compile output with every transitive pin
    README.md            This file
```

## 7. Running the tests

Each package has its own pytest suite. Run them from the repository
root with the virtualenv activated:

```
pytest keats_scraper/tests -v
pytest rag_pipeline/tests  -v
pytest flask_app/tests     -v
pytest evaluation/tests    -v
pytest tests/              -v

pytest                                # run everything at once
```

Coverage settings live in `pyproject.toml`. To reproduce the
per-package coverage numbers:

```
pytest rag_pipeline/tests --cov=rag_pipeline --cov-report=term-missing
```

The evaluation suite's tests exercise `sentence-transformers` and
`bert-score`, which pull model weights on first use. The first run of
`pytest evaluation/tests` can take a few minutes while those caches
warm up; subsequent runs are fast.

## 8. Static analysis

```
./scripts/check.sh
```

This runs:

1. `ruff check rag_pipeline keats_scraper evaluation flask_app` with
   the rule selection in `pyproject.toml`
   (`E, W, F, I, B, UP, N, S, BLE, RET, SIM, RUF`).
2. `mypy rag_pipeline keats_scraper evaluation flask_app` with the
   per-module overrides in `pyproject.toml`. The newer modules
   (`rag_pipeline.retrieval.reranker`,
   `rag_pipeline.retrieval.hybrid_retriever`,
   `rag_pipeline.generation.citation_parser`,
   `rag_pipeline.config`, `keats_scraper.analyses.coverage_report`,
   `evaluation.metrics.sgf`, and `evaluation.metrics.bootstrap`) are
   type-checked under `mypy --strict`; the legacy modules are checked
   at a permissive baseline.

## 9. Running the evaluation suite

The evaluation runs eight ablations against 50 hand-curated QA pairs
and reports each metric with a 95 percent bootstrap confidence
interval (n = 1000, BCa).

```
python -m evaluation.experiments.run_all
python -m evaluation.experiments.run_all --force                                  # rerun even when CSVs already exist
python -m evaluation.experiments.run_all --only llm_comparison topk_comparison
python -m evaluation.experiments.run_all --skip embedding_comparison
```

Approximate wall-clock time on an M2 Pro with `gemma2:2b` already
loaded in Ollama:

| Experiment | Time |
|---|---|
| Baseline (50 queries, rerank and SGF on) | about 20 minutes |
| `embedding_comparison` | about 1 minute |
| `chunk_size_comparison` | about 3 minutes |
| `topk_comparison` (4 LLM runs) | about 50 minutes |
| `retrieval_comparison` | about 1 minute |
| `llm_comparison` (4 models) | about 35 minutes |
| `reranking_comparison` | about 5 minutes |
| `per_category_breakdown` | about 10 minutes |
| Total | roughly 130 to 150 minutes |

Outputs land under `evaluation/results/`:

* A CSV per experiment (`embedding_comparison.csv`,
  `retrieval_comparison.csv`, and so on).
* `baseline_results.json`.
* `per_query/*.json` for deeper analysis.
* `figures/*.png` when the figure post-hoc scripts are invoked.

Each experiment also has a standalone driver (for example
`python -m evaluation.experiments.embedding_comparison`) if you want
to run a single ablation without the orchestrator. Four post-hoc
analyses are available once `run_all` has completed:

```
python -m evaluation.experiments.failure_modes
python -m evaluation.experiments.latency_pareto
python -m evaluation.experiments.significance_tests
python -m evaluation.experiments.emit_per_query
```

## 10. Environment variables

All environment variables are optional; the defaults below are what
the code ships with, and the chatbot runs on them out of the box.
When re-scraping, copy `keats_scraper/.env.example` to
`keats_scraper/.env` and override the values you need.

| Variable | Default | Component | Purpose |
|---|---|---|---|
| `KEATS_COURSE_URL` | `https://keats.kcl.ac.uk/course/view.php?id=130212` | scraper | Which KEATS course to scrape. |
| `COOKIE_ENCRYPTION_KEY` | empty | scraper | Fernet key used to encrypt the saved session cookies. Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. If empty, cookies are stored in plaintext inside the repo (not recommended). |
| `LOG_LEVEL` | `INFO` | scraper | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `REQUESTS_PER_MINUTE` | `20` | scraper | Rate limit applied to KEATS requests. |
| `MIN_DELAY_SECONDS` | `2` | scraper | Minimum jittered delay between requests. |
| `MAX_DELAY_SECONDS` | `5` | scraper | Maximum jittered delay between requests. |
| `CHUNK_SIZE` | `512` | scraper | Target chunk size in tokens (tiktoken `cl100k_base`). |
| `CHUNK_OVERLAP` | `50` | scraper | Overlap between adjacent chunks in tokens. |
| `CHUNK_STRATEGY` | `recursive` | scraper | Either `recursive` (token-based) or `semantic`. |
| `SEMANTIC_PERCENTILE_THRESHOLD` | `80` | scraper | Boundary percentile for the semantic chunker. |
| `SEMANTIC_MIN_TOKENS` | `100` | scraper | Minimum chunk size for the semantic chunker. |
| `FLASK_SECRET_KEY` | `dev-key-change-in-production` | flask | Session signing key. Override in any real deployment. |
| `FEEDBACK_DIR` | `flask_app/data` | flask | Where `feedback.jsonl` is appended. Created on first submission. |
| `INDEX_DIR` | `rag_pipeline/data/index` | flask | Where Flask expects the FAISS index to live (ignored when `run_local.py` builds an absolute path). |
| `EVAL_REPORT_DIR` | unset | evaluation | Optional override for where evaluation CSVs and figures are written. |

The pipeline runs entirely on the local machine via Ollama, so no
external API keys (OpenAI, HuggingFace Inference Endpoints, and so
on) are required.

## 11. Configuration reference

### `RAGConfig` (`rag_pipeline/config.py`)

`RAGConfig` is a strict Pydantic v2 model; every field has a
`Field(...)` validator. The defaults reflect the empirical winners of
the ablation experiments:

* `embedding_model = "multi-qa-MiniLM-L6-cos-v1"`, `embedding_dim = 384`.
* `top_k = 3`, `similarity_threshold = 0.3` (the per-result filter).
* `retrieval_mode = "dense"`. Also accepts `"sparse"` (BM25) and
  `"hybrid"` (Reciprocal Rank Fusion with `rrf_k = 60`).
* `enable_reranking = False`, `rerank_fetch_k = 20`. The reranker
  lazy-loads `cross-encoder/ms-marco-MiniLM-L-6-v2` only when enabled.
* `enable_query_expansion = True`. Expands KCL abbreviations (EC,
  PGT, SSP, and the like) before retrieval.
* `enable_citation_injection = True`. Appends citation instructions
  and chunk-id brackets to the prompt.
* `low_confidence_threshold = 0.4`. Aggregate mean-score gate used
  for the FR9 refusal behaviour.
* `inference_backend: Literal["ollama"] = "ollama"`,
  `ollama_model = "gemma2:2b"`,
  `ollama_base_url = "http://localhost:11434"`,
  `temperature = 0.3`, `top_p = 0.9`, `max_new_tokens = 512`.

The defaults for `chunks_path` and `index_dir` are **relative** paths,
so they resolve against the current working directory. Run Python
from the repository root, or pass absolute paths explicitly (which is
what `run_local.py` does at `run_local.py:32-34`).

### Scraper configuration (`keats_scraper/config.py`)

Loads environment variables via `python-dotenv` and exposes four
dataclasses: `KEATSConfig` (URLs), `AuthConfig` (cookie path and
encryption key), `ScraperConfig` (rate limits), and `ChunkConfig`
(size, overlap, strategy). Every field has a default that matches
`keats_scraper/.env.example`.

### Flask configuration (`flask_app/config.py`)

`FLASK_SECRET_KEY`, `FEEDBACK_DIR`, `INDEX_DIR`.
`create_app(pipeline)` in `flask_app/app.py` accepts an
already-instantiated `RAGPipeline`, which is how `run_local.py` wires
the two components together.

### Evaluation configuration (`evaluation/config.py`)

Defines `REPORT_DIR`, `REPORT_FIGURES_DIR`, `REPORT_TABLES_DIR`.
Honours `EVAL_REPORT_DIR` when set; otherwise writes under
`evaluation/results/`.

## 12. Troubleshooting

**`Ollama is not running at http://localhost:11434`.** Start the
daemon in a separate terminal with `ollama serve`, then confirm with
`curl http://localhost:11434/api/tags`. A successful response returns
a JSON list of locally pulled models.

**`Chunks file not found at .../chunks_for_embedding.jsonl`.** The
bundled file lives at
`keats_scraper/data/chunks/chunks_for_embedding.jsonl`. If it is
missing, the scraper was run with `clear` or you are running
`run_local.py` from a directory other than the repository root.
`run_local.py` pins paths to its own directory, so invoke it as
`python run_local.py` from the project root.

**`model "gemma2:2b" not found`.** Run `ollama pull gemma2:2b`. You
also need `mistral`, `llama3.2`, and `phi3:mini` pulled in advance if
you want to run the `llm_comparison` ablation.

**FAISS dimension mismatch (an `expected_dim` error).** You changed
`RAGConfig.embedding_model` but did not rebuild the index, so
`index.faiss` is still at the 384-dimensional `multi-qa-MiniLM`
configuration. Rebuild with `python run_local.py --build-index`
(Section 5).

**Port 5000 already in use.** Pass `--port 8000` (or any free port)
to `run_local.py`. Recent versions of macOS use port 5000 for AirPlay
Receiver by default, so this collision is common on fresh installs.

**KEATS login browser times out.** SSO and 2FA together can take up
to about three minutes on a slow network. If the Selenium window
closes before you finish, re-run `python main.py login`. Cookies are
saved encrypted only after a successful session handshake, so a
cancelled login leaves nothing behind.

**`sentence-transformers` or `bert-score` downloading model weights
on first run.** Both libraries lazy-download their models on first
import. The first run of `pytest evaluation/tests` therefore takes
several minutes; subsequent runs hit the local cache.

**`ModuleNotFoundError: No module named 'rag_pipeline'` when running
tests.** Run pytest from the repository root with the virtualenv
active. Every package expects to be importable as a top-level module,
which only happens when the repository root is on `sys.path`.

**`chunks_for_embedding_semantic.jsonl` missing when running
`chunking_strategy_comparison`.** The ablation silently skips if the
semantic-chunked file is absent; the bundle ships only the recursive
chunks. To generate the semantic variant, set
`CHUNK_STRATEGY=semantic` in `keats_scraper/.env` and re-run
`python keats_scraper/main.py process`.

## 13. Tech stack

| Component | Libraries |
|---|---|
| Scraping | `selenium`, `webdriver-manager`, `beautifulsoup4`, `lxml`, `html2text`, `pdfplumber`, `tiktoken`, `cryptography`, `click`, `rich` |
| Embeddings | `sentence-transformers` (`multi-qa-MiniLM-L6-cos-v1`) |
| Retrieval | `faiss-cpu`, `rank-bm25` |
| Generation | `ollama` (the official Python client) |
| Web | `Flask 3.x`, `Jinja2` |
| Evaluation | `rouge-score`, `bert-score`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `pandas` |
| Testing | `pytest`, `pytest-mock`, `pytest-cov`, `responses` |
| Code quality | `mypy` (with per-module strict overrides), `ruff`, `pip-tools` |

Exact pinned versions live in `requirements.lock`.

## 14. Academic metadata and licence

* **Student:** Muhammad Ali Naveed
* **Supervisor:** Dr Jeroen Keppens
* **Module:** 6CCS3PRJ Final Year Individual Project
* **Variant:** 2, Informatics Student Handbook
* **Institution:** King's College London, Department of Informatics

This project is submitted as part of the 6CCS3PRJ Final Year
Individual Project module and is subject to the King's College London
academic regulations governing assessed work. It is intended for
academic review; redistribution or use beyond that scope is not
authorised without the author's written permission.
