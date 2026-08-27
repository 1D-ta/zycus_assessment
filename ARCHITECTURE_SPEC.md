### Product Requirements Document (PRD)

* **Objective:** Build a highly scalable, production-grade LLM tooling suite to automate technical support triage and Technical Account Management (TAM) account synthesis workflows.
* **Scope:** Task 1 (Intelligent Triage), Task 2 (TAM Account Summarizer), Task 3 (Evaluation Harness), and Task 4 (Design Note).
* **Constraints:** Exclusively utilize the provided mock dataset. Absolutely no external data scraping. Must execute cleanly from a standard `pip install -r requirements.txt`.
* **Success Metrics:** 100% deterministic outputs for Task 2 briefs. Zero hallucinations in cited risk evidence. Zero automated disqualifiers triggered.
* **Anti-Patterns (Strictly Forbidden):**
* **No Synchronous Linear Scanning:** Never parse or scan raw JSON arrays synchronously inside API request handlers.
* **No Flat File Caching:** Do not use flat `.json` files inside a local `.cache/` directory to enforce determinism.
* **No Unweighted Lexical Search:** Do not rely purely on basic token set intersection ratios.



---

### Complete Directory Tree & File Blueprint

```text
zycus_ai_support/
├── requirements.txt
├── .env.example
├── README.md
├── DESIGN_NOTE.md
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application and SSE endpoints
│   ├── config.py               # Pydantic BaseSettings for environment variables
│   ├── llm_client.py           # Provider abstraction with mock fallback
│   ├── retrieval.py            # Hybrid RRF search (BM25 + Dense Embeddings)
│   ├── pii_sanitizer.py        # Regex pipeline for data redaction
│   ├── schemas.py              # Strict Pydantic models for I/O
│   ├── triage_agent.py         # Task 1 pipeline logic
│   ├── tam_summarizer.py       # Task 2 pipeline logic
│   └── prompts/
│       ├── CHANGELOG.md        # Prompt versioning tracker
│       ├── triage_v1.txt       # Triage instructions template
│       ├── tam_extract_v1.txt  # Chaining Stage 1: Signal extraction
│       └── tam_synth_v1.txt    # Chaining Stage 2: Narrative synthesis
├── data/
│   ├── tickets.json
│   ├── accounts.json
│   └── knowledge-base/
├── eval/
│   ├── harness.py              # Task 3 Evaluation runner
│   ├── metrics.py              # RAGAS-style scoring functions
│   ├── test_cases_triage.json
│   └── test_cases_tam.json
├── ui/
│   └── streamlit_app.py        # Thin visual frontend
└── .github/
    └── workflows/
        └── ci.yml              # Automated testing and eval trigger

```

---

### System Design & Data Schemas

**Hybrid Retrieval Engine (Task 1)**
The system will deploy Reciprocal Rank Fusion (RRF) combining BM25Okapi for exact-keyword matching and a lightweight dense embedding model (ONNX `all-MiniLM-L6-v2`) for semantic parity.

**PII Redaction Pipeline**
All incoming ticket text passes through `pii_sanitizer.py`. Aggressive regex patterns will replace recognized emails with `[EMAIL]`, standard phone formats with `[PHONE]`, and IP addresses with `[IP]`.

**Two-Stage TAM Prompt Chaining (Task 2)**

1. **Stage 1 (Extraction):** The LLM extracts specific risk signals and their verbatim quotes from the 90-day ticket window.
2. **Code-Level Validation:** The backend executes Python substring checks (`quote in raw_ticket_body`). Quotes failing exact string match are dropped.
3. **Stage 2 (Synthesis):** Only code-verified quotes are passed to the second LLM prompt to draft the executive summary.

**API Schemas (Pydantic)**

| Schema Object | Attributes / Types | Enforcement Rules |
| --- | --- | --- |
| `TriageInput` | `subject` (str), `body` (str) | Both fields required. |
| `TriageOutput` | `product_area` (str), `category` (str), `urgency_tier` (Literal["P1", "P2", "P3", "P4"]), `relevant_kb_doc` (Optional[str]), `routed_team` (str), `draft_response` (str) | LLM constrained via `response_format`. |
| `RiskItem` | `issue` (str), `quote` (str), `ticket_id` (str) | Validated against source arrays natively. |
| `TAMOutput` | `executive_summary` (str), `open_risks` (List[RiskItem]), `talking_points` (List[str]) | Exec summary length constrained. |

---

### Implementation Roadmap

| Phase | Agent Instructions | Milestone / Output |
| --- | --- | --- |
| **1. Ingestion & RAG** | Parse `tickets.json` and `accounts.json` into async memory structures. Construct BM25 and dense vector indexes. Implement Regex PII sanitizer. | `retrieval.py` and `pii_sanitizer.py` validated. |
| **2. LLM Client** | Build `llm_client.py` wrapping the external API with a `DeterministicMockLLM` class for offline CI/CD testing. | Mock provider functional. |
| **3. Task 1 Triage** | Develop the `POST /triage` FastAPI endpoint with Pydantic enforcement. Implement SSE endpoints for token streaming. | `POST /triage` returns correct JSON and streams. |
| **4. Task 2 TAM** | Filter tickets (90-day deterministic window). Implement two-stage chaining and code-level substring validation. | `GET /account/{id}/brief` is deterministic. |
| **5. Evals (Task 3)** | Script test cases. **Crucial:** Wrap the LLM client in an exponential backoff retry block and add an `asyncio.sleep(2)` delay between evaluation iterations to prevent free-tier 429 Rate Limit errors.

 | `eval_report.json` generated successfully. |
| **6. Bonus & UI** | Build `ui/streamlit_app.py`. Finalize GitHub Actions CI to invoke the mock-mode evaluation harness on commit. | Functional UI and `ci.yml`. |

---

### Task 3 Evaluation Suite Blueprint

**Test Definitions**

* **Triage Standard:** Test clear bugs, billing questions, and feature requests.
* **Triage Adversarial:** Input ambiguous text to test confidence scoring drop. Prompt injection attempts embedded in the ticket body.
* **TAM Standard:** Account with dense ticket activity and explicit churn complaints.
* **TAM Adversarial:** Query an orphan `account_id` (expect 404). Query an account with zero tickets in the 90-day window.

**Rate Limiting Mitigation**
The evaluation harness must execute LLM-as-judge calls with a 2-second delay (`asyncio.sleep(2)`) and exponential backoff to handle free-tier API rate limits gracefully during parallel execution.

---

### Task 4 Design Note Outline & Mitigations

* **Over-Engineering Justification:** Explicitly justify the Dense Embedding selection. State: "While BM25 suffices for 9 documents, the ONNX dense embedding was implemented specifically to demonstrate production readiness for the 10x scaling requirement."


* **Failure Modes:** Outline risks of prompt injection, schema deviation, and unverified quoting.
* **Latency Tradeoffs:** Discuss the performance cost of Two-Stage TAM prompting versus the necessity of zero-hallucination quote verification.
* **Data Sensitivity:** Detail the regex layers sanitizing PII prior to LLM transmission.
* **Scaling to 10x:** Highlight transitioning from in-memory JSON to relational indexing, and moving synchronous API routing into decoupled message queues.