<div align="center">

# 🚀 Talk2Docs

**A production-oriented AI document intelligence and agentic conversational backend built with FastAPI, Celery, Docling, LangChain, ChromaDB, PostgreSQL, Redis, and multiple AI-driven retrieval and memory systems.**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge\&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?style=for-the-badge\&logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-316192?style=for-the-badge\&logo=postgresql)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-red?style=for-the-badge)
![Redis](https://img.shields.io/badge/Redis-7.0-DC382D?style=for-the-badge\&logo=redis)
![Celery](https://img.shields.io/badge/Celery-5.x-37814A?style=for-the-badge\&logo=celery)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-orange?style=for-the-badge)
![Cohere](https://img.shields.io/badge/Cohere-Reranking-39594D?style=for-the-badge)
![Nginx](https://img.shields.io/badge/Nginx-Reverse%20Proxy-009639?style=for-the-badge\&logo=nginx)
![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

</div>

---

# 🧠 What is Talk2Docs?

Talk2Docs is an asynchronous, user-isolated document intelligence platform that turns uploaded documents into searchable knowledge and provides grounded AI answers.

It started as a document-processing and RAG backend and evolved into a larger AI system containing:

* Structure-aware document processing
* Multi-index RAG
* Hybrid vector + BM25 retrieval
* Adaptive query transformation
* AI reranking
* Structured source-grounded answers
* Three-tier semantic caching
* Stateful conversations
* Short-term conversational context
* Current-session semantic memory
* Cross-session semantic memory
* Agentic tool selection
* Web search
* Document QA as an internal AI tool
* General intent classification as an internal AI tool
* Asynchronous background processing
* User-isolated storage and retrieval

The system therefore contains two major AI paths:

```text
                    TALK2DOCS
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
        DOCUMENT QA          CONVO AI
              │                   │
              ▼                   ▼
       Advanced RAG          Agentic AI
              │                   │
              ▼                   ▼
       Grounded Answers     Tools + Memory
```

---

# ✨ Highlights

### Document Intelligence System

* Asynchronous document ingestion
* Docling document parsing
* Structure-aware `HybridChunker`
* Raw / Summary / Explanation representations
* Per-user ChromaDB storage
* Persistent document lifecycle tracking
* File type, MIME, size and signature validation

### Advanced RAG

* Vector retrieval
* BM25 lexical retrieval
* Hybrid retrieval
* Reciprocal Rank Fusion
* Multi-Query
* HyDE
* Step-Back prompting
* Advanced query translation
* Query decomposition
* Multi-index retrieval
* Cohere encoder reranking
* Source-grounded structured responses

### Three-Tier Caching

* T1 exact Redis cache
* T2 semantic Redis vector cache
* T3 persistent ChromaDB semantic cache
* Cohere reranking for T3
* Source-version validation
* Asynchronous cache population
* Different cache behavior for AnswerAI and ConvoAI

### Agentic Conversational AI

ConvoAI can dynamically use:

* AnswerAI document search
* Current-session semantic memory
* Global semantic memory
* Web search
* General intent classification

The model decides which tools are required while the application controls the execution loop and tool plumbing.

### Infrastructure

* FastAPI
* Async SQLAlchemy
* PostgreSQL
* Redis
* Celery
* Nginx
* SlowAPI
* JWT authentication
* Structured logging
* Centralized exception handling

---

# 🏗️ High-Level Architecture

Talk2Docs separates the HTTP lifecycle from expensive AI and document-processing workloads.

```text
                              CLIENT
                                │
                                ▼
                             NGINX
                                │
                                ▼
                            FASTAPI
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
        Authentication      Validation       Rate Limiting
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                         Service Layer
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
           Document Flow                 Question Flow
                 │                             │
                 ▼                             ▼
              Celery                    Cache / ConvoAI
                 │                             │
        ┌────────┼────────┐          ┌────────┴─────────┐
        │        │        │          │                  │
        ▼        ▼        ▼          ▼                  ▼
     Docling   BM25   Multi-Index  AnswerAI          ConvoAI
        │        │        │          │                  │
        └────────┴────────┘          │            ┌─────┼─────┐
                 │                  │             │     │     │
                 ▼                  │            LTM  Web   Classifier
             ChromaDB ◄─────────────┘
                 │
                 ▼
             PostgreSQL
```

The architecture follows a thin-route approach:

```text
Route
  ↓
Service
  ↓
AI / Retrieval / Worker Logic
  ↓
APIResponse
  ↓
Centralized Response / Exception Handling
```

---

# 📄 Document Ingestion

The document pipeline is asynchronous.

```text
User Upload
     │
     ▼
FastAPI Route
     │
     ▼
Authentication
     │
     ▼
File Validation
     │
     ├── Filename
     ├── Extension
     ├── MIME Type
     ├── Size
     ├── Content
     └── File Signature
     │
     ▼
Validated File
     │
     ▼
Persistent Storage
     │
     ▼
PostgreSQL Document Record
     │
     ▼
Celery Task
     │
     ▼
Docling Parsing
     │
     ▼
HybridChunker
     │
     ▼
Raw Chunks
     │
     ▼
Embedding Generation
     │
     ▼
RAW VDB
     │
     ▼
Document = READY
     │
     ├──────────────────────┐
     ▼                      ▼
Summary Worker       Explanation Worker
     │                      │
     ▼                      ▼
SUMMARY VDB          EXPLANATION VDB
```

The HTTP request does not wait for expensive parsing, chunking and embedding operations.

Instead, background workers process the document and maintain persistent lifecycle state in PostgreSQL.

---

# 🔐 File Validation

Files are validated before entering the processing pipeline.

Validation includes:

* Filename validation
* Extension validation
* MIME type validation
* Maximum file size
* File-content validation
* Signature / magic-byte validation where applicable
* Authentication
* User ownership

The system does not rely solely on:

```text
document.pdf
```

or:

```text
application/pdf
```

when actual file-content validation is required.

---

# ⚙️ Celery Worker Architecture

Talk2Docs uses Celery to isolate expensive background operations from HTTP requests.

The system contains workers for operations such as:

* Document ingestion
* Parsing
* Embedding
* Summary generation
* Explanation generation
* BM25 construction
* Cache population
* Vector-cache population
* Other AI/background operations

Conceptually:

```text
FastAPI
   │
   ▼
Celery Queue
   │
   ├── Ingestion
   ├── Embeddings
   ├── BM25
   ├── Multi-Index
   ├── Cache
   └── AI Operations
```

Workers communicate using serializable payloads rather than passing ORM objects between processes.

This keeps worker execution independent from SQLAlchemy session state.

---

# 🧠 Multi-Index Document Architecture

Talk2Docs maintains multiple representations of the same document knowledge.

```text
                         RAW CHUNKS
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
             RAW VDB     SUMMARY VDB  EXPLANATION VDB
                │            │            │
                │            │            │
           Original       Summary       Explanation
            Content      Representation Representation
```

Each raw chunk maintains a shared `chunk_id`.

Conceptually:

```text
Raw Chunk
   │
   ├── chunk_id = document_id_0
   │
   ├── Summary Representation
   │
   └── Explanation Representation
```

This allows secondary representations to improve retrieval while still resolving results back to the original raw chunk.

---

# 🔎 Hybrid Retrieval

Talk2Docs combines semantic and lexical retrieval.

```text
                    USER QUESTION
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
         Vector Retrieval      BM25
                │                 │
                └────────┬────────┘
                         ▼
                 Ensemble Retrieval
                         │
                         ▼
                Reciprocal Rank Fusion
                         │
                         ▼
                Candidate Documents
                         │
                         ▼
                    Reranking
```

### Vector Retrieval

Captures semantic similarity.

Useful when the question and document express the same idea using different wording.

### BM25

Captures lexical relevance.

Useful for:

* Exact terminology
* Names
* Identifiers
* Uncommon phrases
* Keyword-heavy questions

The two signals provide:

```text
Semantic Understanding
        +
Lexical Precision
```

---

# ⚡ Global BM25 Architecture

BM25 requires corpus-level statistics.

Rebuilding a user-wide BM25 index from scratch for every query would be wasteful.

Talk2Docs therefore maintains a persistent user-scoped BM25 resource.

```text
User
 │
 ├── Raw VDB
 │    ├── Document A
 │    ├── Document B
 │    └── Document C
 │
 └── Global BM25
      ├── Document A chunks
      ├── Document B chunks
      └── Document C chunks
```

The BM25 resource is versioned and persisted.

Redis can be used for the active cached retriever while persistent pickle artifacts provide a durable fallback.

Conceptually:

```text
New Document
     │
     ▼
Update Corpus
     │
     ▼
Build BM25
     │
     ▼
Versioned Artifact
     │
     ▼
Redis Cache
```

---

# 🧭 Adaptive Query Intelligence

For document questions, Talk2Docs does not blindly execute a single retrieval strategy.

The query can be transformed using techniques such as:

```text
User Question
      │
      ▼
Query / Intent Classification
      │
      ├── NONE
      ├── MULTI_QUERY
      ├── HYDE
      ├── STEP_BACK
      ├── ADVANCED_TRANSLATION
      ├── QUERY_DECOMPOSITION
      └── MULTI_INDEXING
      │
      ▼
Retrieval
      │
      ▼
Reranking
      │
      ▼
AnswerAI
```

### Retrieval Techniques

| Technique            | Purpose                                                               |
| -------------------- | --------------------------------------------------------------------- |
| Multi-Query          | Generates multiple query formulations to improve recall               |
| HyDE                 | Uses a hypothetical semantic representation for retrieval             |
| Step-Back            | Retrieves using a broader conceptual question                         |
| Advanced Translation | Converts the question into retrieval-oriented representations         |
| Query Decomposition  | Splits complex questions into independently retrievable sub-questions |
| Multi-Index          | Searches Raw, Summary and Explanation representations                 |

---

# 🎯 Common Retrieval Contract

Different retrieval techniques can have completely different internal implementations.

However, Talk2Docs deliberately normalizes their output to:

```python
list[LangChainDocument]
```

Therefore:

```text
Multi-Query
HyDE
Step-Back
Translation
Decomposition
Multi-Index
Hybrid Retrieval
        │
        ▼
list[LangChainDocument]
        │
        ▼
Reranking
        │
        ▼
AnswerAI
```

This creates a stable boundary between retrieval and downstream answer generation.

The retrieval system can evolve without requiring AnswerAI to understand every retrieval implementation.

---

# 🎯 AI Reranking

Retrieval is optimized for recall.

Reranking improves precision.

```text
Retriever
   │
   ▼
Candidate Documents
   │
   ▼
Cohere Encoder Reranker
   │
   ├── Candidate 1 → score
   ├── Candidate 2 → score
   ├── Candidate 3 → score
   └── ...
   │
   ▼
Sorted Candidates
   │
   ▼
Top-K
```

Talk2Docs validates reranker output before using it.

Validation includes:

* Schema correctness
* Candidate identity
* Candidate uniqueness
* Candidate completeness
* Score validation
* Ranking validation

---

# 🤖 AnswerAI

AnswerAI is the primary document-grounded answer generation system.

It receives retrieved document context and produces structured output.

```text
User Question
      │
      ▼
Cache
      │
      ├── HIT ──────────────► Cached Answer
      │
      ▼
Query Classification
      │
      ▼
Retrieval Strategy
      │
      ▼
Hybrid / Multi-Index Retrieval
      │
      ▼
Cohere Reranking
      │
      ▼
Top-K Context
      │
      ▼
AnswerAI
      │
      ▼
Structured Grounded Response
```

A simplified response structure contains:

```python
class AnswerModel(BaseModel):
    answer: str
    topic: ShortTopicStr
    citations: list[LocationCitation]
    answer_summary: str
    confidence_score: float
    is_meaning_preserved: bool
```

The result is intended to be:

```text
Human-readable
+
Machine-readable
+
Source-grounded
+
Citation-aware
+
Confidence-aware
```

---


# ⚡ Three-Tier Cache Architecture

Talk2Docs uses a three-tier cache architecture for document-answer workloads.

The architecture was originally developed as part of Talk2Docs and was later extracted into a reusable standalone project:

**TriCacheLLM-MMA:**
https://github.com/mohib-ash/TriCacheLLM_MMA

> **Origin note:** Talk2Docs currently uses the **initial version of TriCacheLLM-MMA**. The standalone project exists to generalize and evolve the caching architecture beyond Talk2Docs.

```text
                    Incoming Question
                           │
                           ▼
                  ┌─────────────────┐
                  │ T1 Exact Redis  │
                  └────────┬────────┘
                           │
                    MISS   │   HIT
                           │    └────────────► Response
                           ▼
                  ┌─────────────────┐
                  │ T2 Semantic     │
                  │ Redis Vector    │
                  └────────┬────────┘
                           │
                    MISS   │   HIT
                           │    └────────────► Response
                           ▼
                  ┌─────────────────┐
                  │ T3 Persistent   │
                  │ Chroma + Cohere │
                  └────────┬────────┘
                           │
                    MISS   │   HIT
                           │    └────────────► Response
                           ▼
                       AI Pipeline
```

## Tier 1: Exact Cache

Redis stores exact request responses.

Purpose:

* Duplicate request protection
* Very fast repeated requests
* Double-click/retry protection
* Avoiding unnecessary AI execution for identical requests

---

## Tier 2: Semantic Redis Cache

The question is embedded and searched using a Redis vector index.

Semantically similar questions can reuse a previous answer even when the wording is different.

```text
Question A
"What is GOTEI 13?"

Question B
"Within Soul Society, what purpose does GOTEI 13 serve?"

          │
          ▼

      Embeddings
          │
          ▼
   Redis Vector Search
          │
          ▼
    Similarity Match
          │
          ▼
    Cached Response
```

This tier provides a faster semantic cache before falling back to the persistent cache.

---

## Tier 3: Persistent Chroma Cache

The persistent cache VDB provides semantic caching beyond the Redis layer.

Candidates are:

1. Retrieved from ChromaDB
2. Validated against cache metadata and source versions
3. Reranked with Cohere
4. Checked against the configured similarity threshold
5. Returned when the candidate satisfies the cache policy

```text
Question
   │
   ▼
Chroma Similarity Search
   │
   ▼
Valid Candidates
   │
   ▼
Cohere Reranking
   │
   ▼
Threshold Validation
   │
   ▼
Cached Response
```

This gives Talk2Docs a persistent semantic cache that survives beyond the short-lived Redis layers.

---

## Why Three Tiers?

Each tier solves a different problem:

| Tier | Technology      | Main Purpose                 |
| ---- | --------------- | ---------------------------- |
| T1   | Redis Exact     | Identical request protection |
| T2   | Redis Vector    | Fast semantic reuse          |
| T3   | Chroma + Cohere | Persistent semantic reuse    |
| Miss | AI Pipeline     | Generate a new answer        |

The result is a progressive cache strategy:

```text
Fastest
   │
   ▼
T1 Exact
   │
   ▼
T2 Semantic
   │
   ▼
T3 Persistent Semantic
   │
   ▼
Full AI Pipeline
   │
   ▼
Slowest
```

## TriCacheLLM-MMA

The three-tier caching architecture was first built and validated inside Talk2Docs.

After proving the design in the application, the caching system was extracted into a standalone reusable project:

**TriCacheLLM-MMA**

* **PyPI:** https://pypi.org/project/TriCacheLLM-MMA/
* **GitHub:** https://github.com/mohib-ash/TriCacheLLM_MMA

The standalone project turns the Talk2Docs-specific implementation into a reusable caching architecture that can be integrated into other LLM applications.

Talk2Docs therefore serves as the **initial production-style integration and validation environment** for the architecture, while TriCacheLLM-MMA is its reusable evolution.



---

# 🔄 Cache Validation

Cached answers are not treated as universally valid.

AnswerAI cache entries can contain information such as:

```text
question
response
document IDs
cache policy
citations
provenance
source versions
```

For document-grounded answers, source versions are checked before reusing cached data.

This prevents an answer generated from an older document version from being blindly reused after the underlying source has changed.

---

# 🗣️ ConvoAI

ConvoAI is the stateful conversational AI system within Talk2Docs.

Unlike AnswerAI, ConvoAI is not simply:

```text
Question → Retrieval → Answer
```

It is an agentic system capable of deciding when external information or stored context is required.

```text
                    User Message
                         │
                         ▼
                     ConvoAI
                         │
                         ▼
                 Tool Selection
                         │
          ┌──────────────┼───────────────┐
          │              │               │
          ▼              ▼               ▼
       Memory          AnswerAI       Web Search
          │              │               │
          ├──────────────┤               │
          │              │               │
          ▼              ▼               ▼
 Current Session    Documents        Current Facts
      LTM
          │
          ▼
   Global Session LTM

          +
          
    General Intent
      Classifier
```

The model decides which tool is appropriate based on the request and available context.

---

# 🧰 ConvoAI Tool System

ConvoAI currently has five major tool capabilities.

## 1. AnswerAI Tool

Used when the user asks about:

* Uploaded documents
* Project-specific knowledge
* Custom document canon
* Indexed content

ConvoAI can invoke AnswerAI internally rather than requiring an external HTTP request.

```text
ConvoAI
   │
   ▼
AnswerAI Tool
   │
   ▼
Document Retrieval
   │
   ▼
Grounded Result
   │
   ▼
ConvoAI
```

---

## 2. Current-Session Semantic LTM

Used when relevant information exists earlier in the current conversation but is no longer present in the immediate context.

The current conversation maintains recent context separately.

```text
Current Conversation
        │
        ├── Latest 5 Q&A
        │       │
        │       └── Recent Context
        │
        └── Older Conversation Data
                │
                ▼
          Current-Session LTM
                │
                ▼
        Semantic Retrieval
```

This allows ConvoAI to retrieve older information from the same conversation without repeatedly placing the entire conversation into the prompt.

---

## 3. Global Semantic LTM

Global LTM allows ConvoAI to retrieve relevant information from previous conversation sessions.

```text
Conversation A
      │
      ├── Memory
      │
      ▼
Conversation B
      │
      ├── New Question
      │
      ▼
Global LTM Search
      │
      ▼
Relevant Information
```

This allows useful information to persist beyond a single conversation ID.

---

## 4. Web Search

ConvoAI can use web search when the request depends on current or external information.

Examples:

```text
Latest API version
Current news
Recent events
Current documentation
Real-time facts
```

The model decides when web search is required rather than blindly searching every question.

---

## 5. General Intent Classifier

Ambiguous or unresolved conversational references can be sent to a general intent classifier.

Examples include:

```text
"What about the other one?"

"No, I meant the thing we discussed earlier."

"Which one was that?"
```

The classifier helps resolve vague references before ConvoAI produces a final answer.

---

# 🔁 ConvoAI Tool Execution

The model handles tool selection while the application controls the actual execution loop.

Conceptually:

```text
User Message
     │
     ▼
ConvoAI Model
     │
     ▼
Tool Call?
     │
 ┌───┴────┐
 │        │
 NO      YES
 │        │
 ▼        ▼
Answer  Execute Tool
          │
          ▼
      ToolMessage
          │
          ▼
      ConvoAI Model
          │
          ▼
      Tool Call?
          │
         ...
          │
          ▼
     Final Response
```

The application limits tool execution rounds to prevent uncontrolled tool loops.

The current implementation uses explicit orchestration rather than hiding execution inside a generic agent executor.

This keeps:

* Tool dispatch
* Error handling
* Logging
* Database access
* Async execution
* Internal service calls

under application control.

---

# 🧠 Conversational Context Model

Talk2Docs separates recent context from semantic long-term memory.

```text
                 Conversation
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    Recent Context          Semantic Memory
       Latest 5                  │
       Q&A                       │
          │               ┌──────┴──────┐
          │               │             │
          │               ▼             ▼
          │         Current Session   Global
          │              LTM            LTM
          │               │             │
          └───────────────┴─────────────┘
                          │
                          ▼
                       ConvoAI
```

This avoids unnecessarily sending an entire conversation history on every request while still allowing older information to be recovered semantically.

---

# 🔐 ConvoAI Caching Policy

Conversational caching is intentionally different from document-answer caching.

### ConvoAI

```text
T1 Exact Cache
      │
      ▼
Duplicate Protection
```

ConvoAI does **not** rely on semantic T2/T3 response caching for normal conversational requests.

The reason is contextual correctness.

Two semantically similar conversational questions can require different answers depending on:

* Previous conversation
* Current topic
* Tool state
* User intent
* Retrieved memory
* Position within the conversation

Therefore ConvoAI uses exact T1 caching primarily for duplicate-request protection.

### AnswerAI

```text
T1 Exact
   ↓
T2 Semantic Redis
   ↓
T3 Semantic Chroma
   ↓
AI Pipeline
```

AnswerAI benefits much more directly from semantic response reuse because document-grounded questions can be validated against their source versions.

---

# 🆔 Conversation IDs

Talk2Docs distinguishes between:

```text
conversation_id
```

and:

```text
request_id
```

### Conversation ID

Identifies a conversational branch/session.

### Request ID

Identifies an individual API operation and is used for request correlation and tracing.

This distinction allows users to maintain multiple ongoing conversations while individual requests remain independently traceable.

---

# 📊 End-to-End Question Architecture

The current system effectively contains two different question pipelines.

## Document Question

```text
                         USER
                          │
                          ▼
                    Question Route
                          │
                          ▼
                   T1 Exact Cache
                          │
                    ┌─────┴─────┐
                   HIT          MISS
                    │             │
                    ▼             ▼
                 Response    Cache Classification
                                  │
                                  ▼
                           T2 Semantic Redis
                                  │
                           ┌──────┴──────┐
                          HIT            MISS
                           │               │
                           ▼               ▼
                        Response      T3 Chroma
                                          │
                                   ┌──────┴──────┐
                                  HIT            MISS
                                   │               │
                                   ▼               ▼
                                Response      Query Classification
                                                   │
                                                   ▼
                                           Retrieval Strategy
                                                   │
                                                   ▼
                                           Hybrid Retrieval
                                                   │
                                                   ▼
                                               Reranking
                                                   │
                                                   ▼
                                                AnswerAI
                                                   │
                                                   ▼
                                           Structured Response
                                                   │
                                                   ▼
                                            Cache Population
```

## Conversational Question

```text
                         USER
                          │
                          ▼
                       ConvoAI
                          │
                          ▼
                   T1 Exact Cache
                          │
                    ┌─────┴─────┐
                   HIT          MISS
                    │             │
                    ▼             ▼
                 Response    Recent Context
                                  │
                                  ▼
                           Agentic Tool Selection
                                  │
             ┌────────────────────┼───────────────────┐
             │                    │                   │
             ▼                    ▼                   ▼
        Current LTM           AnswerAI            Web Search
             │                    │                   │
             └────────────────────┼───────────────────┘
                                  │
                                  ▼
                         General Classifier
                           when required
                                  │
                                  ▼
                           Tool Execution
                                  │
                                  ▼
                           Final Response
```

---

# 📈 Performance

Talk2Docs has been benchmarked at the retrieval-pipeline level.

Current retrieval benchmarks were performed against a single indexed document and should not be interpreted as final large-scale production benchmarks.

| Technique            | Retriever + Classifier | Retrieval Strategy |   Reranker |       Total |
| -------------------- | ---------------------: | -----------------: | ---------: | ----------: |
| Step-Back            |              739.50 ms |          826.49 ms | 1258.61 ms | **2.825 s** |
| HyDE                 |              801.31 ms |         1503.64 ms |  807.36 ms | **3.112 s** |
| Advanced Translation |              739.19 ms |          471.04 ms | 1129.38 ms | **2.340 s** |
| Query Decomposition  |              902.63 ms |          721.46 ms |  673.91 ms | **2.298 s** |
| Multi-Index          |              658.66 ms |         2760.05 ms |  645.60 ms | **4.064 s** |

### Cache Validation

The three cache tiers have also been tested independently:

```text
T1 Exact Redis
      │
      └── PASS

T2 Semantic Redis
      │
      └── PASS

T3 Chroma + Cohere
      │
      └── PASS
```

The T3 validation specifically confirmed:

```text
Chroma similarity search
        ↓
Valid candidates
        ↓
Cohere reranking
        ↓
Cache hit
```

---

# 🔐 Authentication & Security

Talk2Docs uses authenticated, user-scoped access throughout the system.

Security mechanisms include:

* JWT authentication
* OAuth2PasswordBearer
* Redis-backed sessions
* Session revocation
* User ban support
* User-scoped PostgreSQL queries
* User-scoped vector databases
* Rate limiting
* File validation
* File signature validation
* Upload size limits
* Centralized exception handling

Document access is scoped using authenticated user identity rather than trusting arbitrary user-provided identifiers.

Conceptually:

```text
Authenticated User
       +
Request ID
       +
Resource ID
       ↓
Authorized Resource
```

---

# 🚦 Rate Limiting

Talk2Docs uses SlowAPI for API-level rate limiting.

Example:

```python
@limiter.limit("3/minute")
```

Rate limiting protects sensitive endpoints against excessive requests.

---

# 📡 Worker Status Polling

Uploads are asynchronous, so clients can monitor processing status.

Example endpoint:

```text
GET /upload_worker/{task_id}/{request_id}
```

The endpoint combines transient Celery state with persistent PostgreSQL document state.

Example:

```json
{
  "worker": {
    "status": "processing",
    "task_id": "...",
    "state": "STARTED"
  },
  "document": {
    "status": "PROCESSING",
    "failure_reason": null
  },
  "multi_index": {
    "doc_id": 42,
    "summary_status": "PROCESSING",
    "explanation_status": "PENDING"
  }
}
```

This prevents transient worker state from being confused with persistent document state.

I also have a conversiaon_id pooling endpoint which fron-end is expected to call before invoking convoAi:

```python
@router.get("/convo-id")
async def create_convo_id(user_id: int = Depends(get_user_jwt_payload), db: AsyncSession = Depends(get_db)):
    convo_id: str = create_conversation_id(user_id)
    request_id = str(uuid.uuid4())
    
    return APIResponse(
        success=True,
        data={"convo_id": convo_id},
    )
```

---

# 🧱 APIResponse Architecture

Internal services use a common response representation:

```text
success
+
data
+
error_code
+
error_message
```

Conceptually:

```python
APIResponse(
    success=True,
    data=...,
    error_code=None,
    error_message=None
)
```

This provides a predictable boundary between:

* Business-level failures
* Validation failures
* AI failures
* Infrastructure failures
* Unhandled application exceptions

---

# 🛡️ AI Failure & Validation

AI output is not blindly trusted.

Structured AI responses are validated through Pydantic models.

The general approach is:

```text
AI Output
    │
    ▼
Parse
    │
    ▼
Validate
    │
 ┌──┴──┐
 │     │
PASS  FAIL
 │     │
 ▼     ▼
Use   Recovery
        │
        ▼
      Validate
        │
      ┌─┴─┐
      │   │
    PASS FAIL
      │   │
      ▼   ▼
     Use Error
```

The same philosophy is applied to:

* Answer generation
* Query classification
* Reranking
* ConvoAI structured output
* Tool results where validation is required

---

# 📁 Project Structure

A simplified representation:

```text
Talk2Docs/
│
├── Ai/
│   ├── answer_ai.py
│   ├── query_classifier.py
│   ├── retry_logic.py
│   ├── ai_utils.py
│   ├── convo_ai/
│   ├── reranker/
│   └── query_construction/
│       ├── HYDE/
│       ├── multi_query/
│       ├── multi_indexing/
│       ├── query_decomp/
│       ├── step_back/
│       └── advanced_translation/
│
├── celery_worker/
│   ├── celery_app.py
│   └── Tasks/
│       ├── Ai_worker/
│       ├── embedding_worker.py
│       ├── ingestion_worker.py
│       └── chat_worker.py
│
├── routers/
│   ├── Ai/
│   ├── auth/
│   └── users/
│
├── db_tables/
│
├── vector_db/
│
├── docling/
│
├── core/
│   ├── Exceptions/
│   └── rate_limiters/
│
├── utils/
│   ├── logging/
│   └── schemas/
│
├── alembic/
│
├── nginx/
│
└── ...
```

---



# 🛠️ Technology Stack

## Backend

* Python 3.12+
* FastAPI
* Pydantic v2
* SQLAlchemy 2.0 Async
* Alembic

## AI / RAG

* LangChain
* Sentence Transformers
* ChromaDB
* BM25
* Hybrid Retrieval
* Reciprocal Rank Fusion
* Multi-Query
* HyDE
* Step-Back
* Query Decomposition
* Advanced Query Translation
* Multi-Index Retrieval
* Cohere Encoder Reranking
* Structured Pydantic AI output

## Conversational AI

* LangChain tool calling
* Agentic tool orchestration
* Semantic current-session memory
* Semantic cross-session memory
* Web search
* Document QA as an internal tool
* General intent classification

## Document Processing

* Docling
* HybridChunker
* Content validation
* Signature validation

## Background Processing

* Celery
* Redis

## Database & Storage

* PostgreSQL
* ChromaDB
* Redis
* Persistent BM25 artifacts

## Authentication & Security

* JWT
* OAuth2PasswordBearer
* Redis-backed sessions
* Session revocation
* SlowAPI
* User-scoped retrieval

## Infrastructure & Observability

* Nginx
* Structured logging
* Centralized exception handling
* Celery lifecycle tracking
* Persistent document status tracking

---

# 🧪 Engineering Principles

## Separation of Concerns

Routes, services, workers, retrieval systems, AI systems and storage have distinct responsibilities.

```text
HTTP
 ↓
Service
 ↓
AI / Retrieval / Database
 ↓
APIResponse
```

---

## Retrieval Modularity

Retrieval strategies can evolve without forcing AnswerAI to understand their internal implementation.

```text
Retrieval Strategy
       ↓
list[LangChainDocument]
       ↓
Reranker
       ↓
AnswerAI
```

---

## User Isolation

Documents, vector databases, caches and semantic memories are scoped to authenticated users.

```text
User A
 ├── Documents
 ├── VDBs
 ├── Cache
 └── Memory

User B
 ├── Documents
 ├── VDBs
 ├── Cache
 └── Memory
```

---

## Asynchronous Processing

Expensive work is moved away from HTTP request execution whenever possible.

```text
HTTP Request
     │
     ▼
FastAPI
     │
     ▼
Dispatch
     │
     ▼
Celery / Background Task
     │
     ▼
Expensive Work
```

---

## Context-Aware AI

The system does not treat every question as the same type of problem.

A request may require:

```text
Document Retrieval
      OR
Recent Conversation Context
      OR
Current-Session Memory
      OR
Global Memory
      OR
Web Search
      OR
General Intent Resolution
```

The goal is to retrieve the **right information source**, rather than simply retrieving more information.

---

# 📈 Current Progress

## Core Backend

* [x] FastAPI
* [x] Async SQLAlchemy
* [x] PostgreSQL
* [x] Alembic
* [x] JWT authentication
* [x] OAuth2PasswordBearer
* [x] Redis-backed sessions
* [x] Session revocation
* [x] Celery integration
* [x] Rate limiting
* [x] Centralized exception handling
* [x] Structured logging
* [x] Nginx integration

## Document Pipeline

* [x] File validation
* [x] File persistence
* [x] Document metadata
* [x] Asynchronous upload processing
* [x] Docling parsing
* [x] HybridChunker
* [x] Embedding generation
* [x] Raw ChromaDB indexing
* [x] Document state tracking
* [x] Celery retry handling
* [x] Worker status polling

## Multi-Index RAG

* [x] Raw VDB
* [x] Summary VDB
* [x] Explanation VDB
* [x] Summary AI
* [x] Explanation AI
* [x] Shared `chunk_id`
* [x] Multi-index lifecycle tracking
* [x] Background construction
* [x] Summary retrieval
* [x] Explanation retrieval
* [x] Parallel secondary retrieval
* [x] Raw chunk resolution

## Retrieval

* [x] Vector retrieval
* [x] BM25 retrieval
* [x] Hybrid retrieval
* [x] Ensemble retrieval
* [x] Reciprocal Rank Fusion
* [x] Multi-Query
* [x] HyDE
* [x] Step-Back
* [x] Advanced translation
* [x] Query decomposition
* [x] Multi-index retrieval
* [x] Common `list[LangChainDocument]` retrieval contract
* [x] AI reranking
* [x] Reranker validation
* [x] Candidate validation
* [x] Persistent user-scoped BM25

## Caching

* [x] T1 exact Redis cache
* [x] T2 semantic Redis vector cache
* [x] T3 persistent Chroma semantic cache
* [x] Cohere cache reranking
* [x] Cache source-version validation
* [x] Cache provenance metadata
* [x] Asynchronous cache population
* [x] AnswerAI three-tier caching
* [x] ConvoAI T1-only duplicate protection

## Conversational AI

* [x] Conversation IDs
* [x] Recent conversational context
* [x] Latest-five Q&A context
* [x] ConvoAI
* [x] Agentic tool selection
* [x] Manual tool execution loop
* [x] AnswerAI tool
* [x] Web search tool
* [x] Current-session semantic LTM
* [x] Global semantic LTM
* [x] General intent classifier tool
* [x] Structured ConvoAI responses
* [x] Tool execution logging

## Answer Generation

* [x] Document-grounded answers
* [x] Structured Pydantic output
* [x] Source citations
* [x] Verbatim evidence
* [x] Confidence scoring
* [x] Meaning-preservation validation

---

# 🗺️ Future Work

The core document intelligence, caching, and agentic conversation architecture is now implemented.

The next major feature is **tabular document intelligence for CSV and XLSX files**.

Unlike normal document QA, tabular questions should not rely on treating rows as ordinary text chunks. Instead, Talk2Docs will understand the **table schema**, retrieve the relevant structural information, and generate validated Pandas operations against the original file at question time.

## CSV / XLSX Pipeline

The planned architecture is:

```text
                         Upload
                           │
                           ▼
              ┌─────────────────────────┐
              │ Background Tabular      │
              │ Worker                  │
              │                         │
              │ Detect CSV / XLSX       │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Extract Schema +        │
              │ Metadata                │
              │                         │
              │ • Column names          │
              │ • Data types            │
              │ • Table structure       │
              │ • Useful metadata       │
              └────────────┬────────────┘
                           │
                           ▼
                    Index Schema
                           │
                           ▼
                  Schema VDB / Index
                           │
                           │
              Original File Remains
              the Source of Truth
                           │
                           ▼
                     User Question
                           │
                           ▼
                 Retrieve Relevant
                       Schema
                           │
                           ▼
              Load Original File
                 at Runtime
                           │
                           ▼
                    Pandas DataFrame
                           │
                           ▼
              ┌─────────────────────────┐
              │ LLM 1                   │
              │                         │
              │ Question + Schema       │
              │          ↓              │
              │ Pydantic Output         │
              │          ↓              │
              │ Pandas Code             │
              └────────────┬────────────┘
                           │
                           ▼
                      Validation
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
        Column Check    AST Check    Forbidden
        vs Schema       / Allowed    Operations
                        Code Rules
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                  Isolated Execution
                           │
                           ▼
                Execute Pandas Code
                  Against DataFrame
                           │
                           ▼
                Deterministic Result
                           │
                           ▼
              ┌─────────────────────────┐
              │ LLM 2                   │
              │                         │
              │ Original Question       │
              │ + Relevant Schema       │
              │ + Executed Result       │
              │          ↓              │
              │ Final Natural Language  │
              │ Answer                  │
              └─────────────────────────┘
```

## Design Principles

### Original Files Remain the Source of Truth

CSV and XLSX files will be preserved after upload.

The VDB will store **schema and metadata**, not the authoritative table contents.

At question time, the original file will be loaded into a Pandas DataFrame and used for deterministic computation.

```text
Original CSV/XLSX
       │
       ▼
Runtime DataFrame
       │
       ▼
Validated Pandas Operation
       │
       ▼
Deterministic Result
```

This avoids relying on an LLM to perform arithmetic, filtering, aggregation, or other operations over retrieved text chunks.

---

### Schema-First Retrieval

The system will first retrieve the relevant table schema.

For example:

```text
Question:
"What was the average salary of employees in the Finance department?"

Retrieved Schema:

employees
├── employee_id: integer
├── name: string
├── department: string
├── salary: float
└── joining_date: datetime
```

The schema gives the model the structural information it needs to generate the Pandas operation.

---

### LLM 1: Code Generation

The first model will receive:

```text
Original Question
        +
Relevant Table Schema
```

and produce structured output through Pydantic.

Conceptually:

```text
Question + Schema
       │
       ▼
      LLM
       │
       ▼
Pydantic Model
       │
       ▼
Validated Pandas Code
```

The model will **not directly execute its own output**.

---

### Code Validation

Generated Pandas code will pass through a validation layer before execution.

Validation will include:

* Column names must exist in the retrieved schema
* Only permitted operations/functions may be used
* Python AST must satisfy the allowed structure
* Forbidden operations must be rejected
* Unexpected imports or filesystem/network access must be blocked
* Generated code must operate only on the intended DataFrame

The goal is to treat model-generated code as **untrusted input**.

```text
LLM Generated Code
        │
        ▼
     Validate
        │
   ┌────┴────┐
   │         │
Valid      Invalid
   │         │
   ▼         ▼
Execute     Reject
```

---

### Isolated Execution

Only validated Pandas operations will be executed against the runtime DataFrame.

The execution layer will produce a deterministic result.

```text
DataFrame
    +
Validated Pandas Code
    │
    ▼
Execution
    │
    ▼
Deterministic Result
```

The LLM does not calculate the result itself.

It asks Pandas to perform the computation and receives the actual result.

---

### LLM 2: Answer Generation

The final model will receive:

```text
Original Question
        +
Relevant Schema
        +
Executed Result
```

and convert the deterministic result into a natural-language answer.

```text
Question
   +
Schema
   +
Actual Computed Result
        │
        ▼
      LLM 2
        │
        ▼
   Final Answer
```

This separates **computation from explanation**.

The first model determines *what operation should be performed*.

Pandas determines *what the actual result is*.

The second model determines *how that result should be explained to the user*.

## Why This Architecture?

The important difference between ordinary document QA and tabular QA is that a table often requires **computation rather than retrieval**.

For example:

```text
"What was the total revenue in Q3?"

```

should not be answered by hoping the LLM finds a matching sentence in a vector database.

Instead:

```text
Question
   ↓
Schema Retrieval
   ↓
Pandas Code Generation
   ↓
Code Validation
   ↓
Actual DataFrame Execution
   ↓
Deterministic Result
   ↓
Natural Language Answer
```

This allows Talk2Docs to use RAG for **understanding the structure of the data**, while using deterministic execution for **the actual computation**.

## Future Hardening

After the initial CSV/XLSX pipeline is implemented, the next focus will be:

* Stronger generated-code validation
* Execution sandboxing and resource limits
* Better handling of multiple sheets in XLSX
* Large-file handling
* Schema/version validation
* Better error recovery when generated code fails
* Regression and integration testing
* Performance testing
* Production monitoring and observability

The overall goal is to make tabular QA another first-class workload inside Talk2Docs rather than treating spreadsheets as ordinary text documents.


### Production Hardening

* [ ] Larger multi-document benchmarks
* [ ] Load testing
* [ ] Expanded regression test suite
* [ ] More provider failure/recovery testing
* [ ] Expanded observability
* [ ] CI/CD
* [ ] Production deployment
* [ ] Monitoring

### Retrieval & AI

* [ ] Larger-corpus retrieval benchmarking
* [ ] Further model/provider benchmarking
* [ ] Retrieval quality evaluation datasets
* [ ] More advanced agentic workflows
* [ ] Additional tool integrations

### Memory

* [ ] Further memory ranking improvements
* [ ] Memory lifecycle optimization
* [ ] More sophisticated context compression
* [ ] Long-running conversation optimization

### Data Support

* [ ] Advanced CSV/XLSX question-answering pipeline
* [ ] Tabular-data-specific retrieval
* [ ] Additional document formats

---

# ❤️ Why Talk2Docs Exists

Talk2Docs started as a document-processing backend.

It gradually evolved into a system designed around a larger question:

> **How can an AI application decide what information it actually needs before answering?**

The result is no longer simply:

```text
Document
   ↓
Embeddings
   ↓
Vector Search
   ↓
LLM
```

Instead:

```text
                       USER
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
        Document QA              ConvoAI
             │                       │
             ▼                       ▼
       Cache Layers             Tool Selection
             │                       │
             ▼              ┌────────┼─────────┐
      Query Intelligence    │        │         │
             │             LTM     Web      AnswerAI
             ▼              │        │         │
       Hybrid Retrieval     └────────┼─────────┘
             │                       │
             ▼                       ▼
         Reranking             Agentic Context
             │                       │
             ▼                       ▼
        Grounded Answer       Structured Response
```

Talk2Docs is therefore a combination of:

```text
Document Processing
        +
Advanced RAG
        +
Hybrid Retrieval
        +
Multi-Index Retrieval
        +
Three-Tier Semantic Caching
        +
Conversational Memory
        +
Agentic Tool Use
        +
Structured AI Responses
```

---

# 📜 License

MIT License

---

Built with:

**FastAPI · Celery · Docling · LangChain · ChromaDB · PostgreSQL · Redis · Pydantic · Sentence Transformers · Cohere**
