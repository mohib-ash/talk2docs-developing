<div align="center">

# 🚀 Talk2Docs

**A production-oriented AI document intelligence and advanced RAG backend built with FastAPI, Celery, Docling, LangChain, ChromaDB, PostgreSQL, Redis, and multiple AI-driven retrieval techniques.**

![Version](https://img.shields.io/badge/version-v1.0-blue?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-316192?style=for-the-badge&logo=postgresql)](https://www.postgresql.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-red?style=for-the-badge)](https://www.sqlalchemy.org/)
[![Redis](https://img.shields.io/badge/Redis-7.0-DC382D?style=for-the-badge&logo=redis)](https://redis.io/)
[![Celery](https://img.shields.io/badge/Celery-5.x-37814A?style=for-the-badge&logo=celery)](https://docs.celeryq.dev/)
[![Nginx](https://img.shields.io/badge/Nginx-Reverse%20Proxy-009639?style=for-the-badge&logo=nginx)](https://nginx.org/)
[![Cohere](https://img.shields.io/badge/Cohere-Encoder%20Reranking-39594D?style=for-the-badge&logo=cohere)](https://cohere.com/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

Talk2Docs is an asynchronous, user-isolated document intelligence platform
that transforms uploaded documents into searchable knowledge and generates
grounded AI answers.

Unlike a basic PDF → embeddings → LLM pipeline, Talk2Docs combines:

- structure-aware document processing
- hybrid semantic + lexical retrieval
- adaptive query transformation
- multi-index retrieval
- AI reranking
- source-grounded structured answers
- asynchronous Celery processing

When a user asks a question, the system does not blindly perform vector search.
Instead, the query passes through an AI-driven decision pipeline:

```text
User Question
      │
      ▼
Intent Classification and Query Classification
      │
      ▼
Retrieval Technique Selection
      │
      ├── None
      ├── Multi-Query
      ├── HyDE
      ├── Step-Back
      ├── Advanced Translation
      ├── Query Decomposition
      └── Multi-Index Retrieval
      │
      ▼
Retrieval
      │
      ▼
Hybrid Search
(Vector + BM25)
      │
      ▼
Candidate Documents
      │
      ▼
AI Reranking
      │
      ▼
Top-K Relevant Documents
      │
      ▼
Answer Generation
      │
      ▼
Grounded Structured Response
````

The architecture is designed around separation of concerns, asynchronous processing, fault tolerance, user isolation, extensibility, and AI-assisted retrieval. The link between the Query Constructor and Response AI is handled via `List[LangChainDocument]`.

---

## ✨ Highlights

- 📄 Asynchronous document ingestion
- 🧠 Docling + structure-aware chunking
- 📦 Utilizes My own package PyPI: [TriCacheLLM-MMA](https://pypi.org/project/TriCacheLLM-MMA/) | Github: [TriCacheLLM_MMA GitHub](https://github.com/mohib-ash/TriCacheLLM_MMA)
- 🗃️ Raw / Summary / Explanation vector indexes
- 🔍 Hybrid Vector + BM25 retrieval
- 🔀 Reciprocal Rank Fusion
- 🧭 AI-driven retrieval strategy selection
- 🔎 Multi-Query, HyDE, Step-Back & Query Decomposition
- 🎯 AI-powered reranking
- 📌 Source-grounded structured answers
- 🔐 JWT + Redis-backed sessions
- ⚡ Celery + Redis background processing
- 👤 User-isolated document retrieval
- 📊 Persistent document lifecycle tracking
- 📝 Structured logging & centralized errors



---

# 📦 TriCacheLLM-MMA Integration

The core caching, multi-index coordination, and retrieval components of Talk2Docs originated from [TriCacheLLM_MMA GitHub](https://github.com/mohib-ash/TriCacheLLM_MMA) and have since been modularized and published as an independent Python package:

- **PyPI Package:** [TriCacheLLM-MMA](https://pypi.org/project/TriCacheLLM-MMA/)
- **Purpose:** Manages asynchronous caching, chunk-to-representation mapping across Raw, Summary, and Explanation indexes, and optimized retrieval query execution.

---

# 🏗️ High-Level Architecture

Talk2Docs separates the HTTP lifecycle from computationally expensive AI and document-processing workloads.

```text
                           ┌─────────────────┐
                           │     Client      │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │      Nginx      │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │     FastAPI     │
                           │   API Layer     │
                           └────────┬────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
             Authentication                    Request Validation
             Rate Limiting                     Service Layer
                    │                                │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                              ┌───────────┐
                              │  Celery   │
                              │  Queues   │
                              └─────┬─────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
         Ingestion Worker     AI Workers          Other Workers
                │                   │
                ▼                   │
             Docling                │
                │                   │
                ▼                   │
             Chunking               │
                │                   │
                ▼                   │
           Embeddings               │
                │                   │
                ▼                   │
          ChromaDB ◄───────────────┘
                │
                │
                ▼
        PostgreSQL Metadata
```

Redis provides the messaging and transient infrastructure required by Celery, while PostgreSQL remains the persistent source of truth for application and document state.

---

# 📄 Document Ingestion Architecture

The document pipeline begins when a user uploads a file.

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
     ├── File Size
     └── File Content / Signature
     │
     ▼
Validated Upload
     │
     ▼
Celery Task
     │
     ▼
Persistent File Storage
     │
     ▼
PostgreSQL Document Record
     │
     ▼
Docling Parsing
     │
     ▼
Structure-Aware Chunking
     │
     ▼
Raw Chunks
     │
     ▼
Embedding Generation
     │
     ▼
RAW VECTOR DATABASE
     │
     ▼
Document.status = READY
```

The API does not wait for expensive parsing, chunking, or embedding operations.
Instead, the upload request dispatches background work and returns a task identifier.

---

# 🔐 File Validation

Uploaded files are validated before entering the processing pipeline.

Validation includes:

* File name validation
* Extension validation
* MIME type validation
* Maximum file size limits
* File-content validation
* Magic/signature validation where applicable
* User authentication
* User ownership

The system does not rely solely on the file extension or MIME type when content validation is required.

For example, a file claiming to be a PDF should actually contain a valid PDF signature rather than merely having:

```text
document.pdf
```

or:

```text
application/pdf
```

This provides an additional layer of protection against malformed or misleading uploads.

---

# ⚙️ Asynchronous Worker Architecture

Talk2Docs uses Celery to separate expensive processing from HTTP requests.

The general pipeline is:

```text
FastAPI
   │
   ▼
Task 1
   │
   ├── Validate payload
   ├── Save file
   ├── Create Document record
   └── Dispatch processing task
                │
                ▼
             Task 2
                │
                ├── Parse
                ├── Chunk
                ├── Embed
                └── Store in ChromaDB
```

Workers communicate using serializable payloads rather than passing ORM objects between processes.
This keeps workers independently executable and avoids coupling Celery processes to SQLAlchemy session state.
There are more workers in the project for instance new BM25 new creation when new doc arrives, saving data in cache vector database etc.

---

# 🧠 Raw Vector Database

The first vector database created for a document is the **Raw VDB**.
Its purpose is to preserve the original chunk representation for accurate final answer generation.

```text
Parsed Document
      │
      ▼
Chunking
      │
      ▼
Raw Chunk 1 ──► Embedding ──► Raw VDB
Raw Chunk 2 ──► Embedding ──► Raw VDB
Raw Chunk 3 ──► Embedding ──► Raw VDB
...
```

Each chunk contains metadata allowing it to be associated with:

* User
* Document
* Chunk
* Source file
* Page
* Section
* Other retrieval metadata

The Raw VDB is immediately usable for normal question answering once ingestion completes.

---

# 🗂️ Multi-Index Architecture

Talk2Docs does not stop at a single vector representation.
After the Raw VDB becomes available, background workers can construct two additional representations:

```text
                         RAW CHUNKS
                             │
                ┌────────────┼─────────────────┐
                │            │                 │
                ▼            ▼                 ▼
             RAW VDB     SUMMARY VDB     EXPLANATION VDB
                │            │                 │
                │            │                 │
          Original       Summary AI       Explanation AI
           Content       Representation   Representation
```

Each raw chunk can produce three retrieval representations:

**Raw**

**Summary**

**Explanation**

All representations preserve the same `chunk_id`.
This allows secondary indexes to improve retrieval recall while the final
answer-generation stage resolves results back to the original raw chunk.

---

# 🧠 Summary & Explanation AI

The multi-index workers generate two additional representations for each raw chunk.

### Summary Representation

The Summary AI transforms:

```text
Raw Chunk
   │
   ▼
Summary AI
   │
   ▼
Compact semantic representation
```

### Explanation Representation

The Explanation AI transforms:

```text
Raw Chunk
   │
   ▼
Explanation AI
   │
   ▼
Conceptual / explanatory representation
```

The mapping remains strictly:

```text
1 Raw Chunk
     ↓
1 Summary Representation
     ↓
1 Explanation Representation
```

The shared `chunk_id` preserves this relationship.

This makes the additional indexes retrieval-oriented representations rather than independent copies of the document.

---

# 🔎 Retrieval Architecture

Once the document has been indexed, the system can retrieve knowledge using both lexical and semantic search.

```text
                    User Question
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
          Vector Search        BM25 Search
                │                 │
                └────────┬────────┘
                         ▼
                  Hybrid Retrieval
                         │
                         ▼
               Reciprocal Rank Fusion
                         │
                         ▼
                  Candidate Documents
```

The hybrid retriever combines:

### Vector Retrieval

Captures semantic similarity.

Useful when the question and document use different wording but express the same concept.

### BM25 Retrieval

Captures lexical relevance.
Useful when exact terminology, names, identifiers, or uncommon phrases matter.

### Ensemble Retrieval

The two signals are combined using an ensemble retriever.
This gives the system both:

```text
Semantic Understanding
        +
Lexical Precision
```

rather than relying exclusively on one retrieval method.

---

# ⚡ BM25 Architecture

Talk2Docs separates the **lexical retrieval index** from the vector database lifecycle.
A BM25 retriever requires corpus-level statistics such as:

```text
Term Frequency (TF)
Document Frequency (DF)
Inverse Document Frequency (IDF)


This means constructing a BM25 retriever over the entire user corpus every time a new document is uploaded can eventually become wasteful.
The system therefore uses a user-scoped global BM25 corpus as a background-built retrieval resource.

Conceptually:

User
 │
 ├── Raw VDB
 │    ├── Document A
 │    ├── Document B
 │    ├── Document C
 │    └── ...
 │
 └── BM25
      ├── Document A chunks
      ├── Document B chunks
      ├── Document C chunks
      └── ...

The BM25 corpus is scoped to the authenticated user, just like the user's ChromaDB.
This allows the system to maintain a single lexical retrieval resource for the user's document library while still applying document-level filtering when required.
Why BM25 is built separately

The vector retriever can query ChromaDB directly:

Question
   │
   ▼
Chroma Vector Search
   │
   └── metadata filtering

BM25, however, operates over a collection of text documents.

Therefore the system reconstructs the required LangChainDocument objects from the stored Chroma documents:

ChromaDB
   │
   ├── documents
   └── metadatas
          │
          ▼
LangChainDocument
          │
          ▼
BM25Retriever (This will be cached, if not available we use version - 1)

This keeps the responsibilities separate:
ChromaDB
    ↓
Vector Retrieval

BM25 Index
    ↓
Lexical Retrieval

The two are then combined through the hybrid ensemble retriever.
```

***

# 🧭 AI Query Pipeline
The most important part of Talk2Docs is what happens **after the user asks a question**.
The system does not immediately send the question to a retriever.

It first determines what kind of question it is and which retrieval strategy should be used.

```text
                         User Question
                              │
                              ▼
                    ┌───────────────────┐
                    │  Intent Classifier│
                    │        &          │
                    │ Query Classifier  │
                    └─────────┬─────────┘
                              │
                              ▼
        Selected Technique and Intent Classification (merged) 
                              │
         ┌────────────────────┼─────────────────────┐
         │                    │                     │
         ▼                    ▼                     ▼
    Multi-Query             HyDE              Step-Back
         │                    │                     │
         ├──────────────┬─────┴──────────────┬──────┤
         │              │                    │
         ▼              ▼                    ▼
 Advanced Translation  Decomposition   Multi-Index
         │              │                    │
         └──────────────┴──────────┬─────────┘
                                   │
                                   ▼
                              Retrieval
                                   │
                                   ▼
                          Candidate Documents
                                   │
                                   ▼
                              Reranking
                                   │
                                   ▼
                              Top Results
                                   │
                                   ▼
                            Answer Generation
```
---





## ⚡ Post-Generation Lifecycle & Dual-Tier Caching

Once `Answer_ai` successfully compiles and returns the final response, the application enters an optimized post-processing and caching pipeline. To ensure low latency for future requests without blocking the active HTTP response, the system executes caching tasks asynchronously.

```text
               Answer Generation (Successful APIResponse)
                                   │
                                   ▼
                      Pydantic Serialization Check
               (model_dump_json() / dict() / json.dumps())
                                   │
                                   ▼
                   Fetch Corpus Source Versions
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │     Async Background Task (asyncio.create_task) │
          ├────────────────────────┬────────────────────────┤
          │                        │                        │
          ▼                        ▼                        ▼
    Tier 1: Hot Exact Cache    Tier 2: Vector Cache    Structured Logging
    (Redis SETEX - 24h TTL)    (Semantic Redis Store)  (QuestionLogs Enum)
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                       Return Response to Client


```

# 🧭 Query & Intent Classification

Talk2Docs combines intent classification and query-technique classification
into a unified classification stage.

```text
User Question
      │
      ▼
Query & Intent Classifier
      │
      ├── Intent
      │
      └── Retrieval Technique
              │
              ├── NONE
              ├── MULTI_QUERY
              ├── HYDE
              ├── STEP_BACK
              ├── ADVANCED_TRANSLATION
              ├── QUERY_DECOMPOSITION
              └── MULTI_INDEXING
```
The merged implementation is contained in:
>**Ai/query_classifier.py**

The original standalone intent-classification implementation is retained as:

>**Ai/intent_classifier_manul.py**

for reference and comparison.

---

## 🧠 Retrieval Techniques

| Technique | Purpose |
|---|---|
| Multi-Query | Improves recall through multiple query formulations |
| HyDE | Retrieves using a hypothetical semantic representation |
| Step-Back | Retrieves using a broader conceptual question |
| Translation | Transforms queries into retrieval-friendly representations |
| Decomposition | Breaks complex questions into independently retrievable sub-questions |
| Multi-Index | Searches raw, summary and explanation representations |

***

# 🎯 AI Reranking

After retrieval, the system has a set of candidate documents.

Retrieval is optimized for **recall**.

The reranker is responsible for improving **precision** by evaluating the
relevance of each retrieved candidate against the user's query.

```text
Retriever
   │
   ▼
Candidate Documents
   │
   ▼
Cohere Encoder Reranker
   │
   ├── candidate_0 → relevance score
   ├── candidate_1 → relevance score
   ├── candidate_2 → relevance score
   └── ...
   │
   ▼
Sorted Candidates
   │
   ▼
Top-K
```
Talk2Docs currently uses Cohere's encoder-based reranking for the primary
reranking path.
The encoder evaluates query-document relevance and produces relevance scores
used to reorder the retrieved candidates.

The current encoder implementation is located in:
>**re_rank_via_encoder.py**

The project also preserves the original manual AI reranking implementation in:

> **rank\_docs\_manual.py**

---

# 🧱 The API Contract Between QueryClassifier and Response AI

Throughout the retrieval architecture, the system deliberately standardizes retrieval output to:

```python
list[LangChainDocument]
```

This is one of the most important architectural boundaries in the project.

Different retrieval techniques can internally behave very differently:

```text
Multi-Query
HyDE
Step-Back
Translation
Decomposition
Multi-Index
Hybrid Retrieval
```

But the downstream pipeline does not need to know how the documents were retrieved.

They all eventually become:

```text
list[LangChainDocument]
```

This creates a common contract between:

```text
Retrieval
     ↓
Reranking
     ↓
Answer Generation
```

The result is essentially the project's **Great Tranquilizer**:

> No matter how complicated the retrieval strategy becomes, downstream services receive the same boring, beautiful `list[LangChainDocument]`.

This dramatically reduces coupling between retrieval techniques and answer generation.

---

# 🎯 Final Candidate Selection

The overall retrieval stage can therefore be summarized as:

```text
Question
   │
   ▼
Intent Classification
   │
   ▼
Query Technique Classification
   │
   ▼
Technique Execution
   │
   ▼
Hybrid / Multi-Index Retrieval
   │
   ▼
Candidate Documents
   │
   ▼
AI Reranking
   │
   ▼
Top-K Documents
   │
   ▼
Answer AI
```

The answer generator receives only the highest-quality context rather than the entire retrieval pool.

---

# 🤖 Answer Generation

Once the final context is available, Answer AI generates a grounded response.

The final answer is represented using a structured Pydantic schema:

```python
class LocationCitation(BaseModel):
    page_number: int | None = None
    section_heading: str | None = None
    location_fallback: str | None = None
    verbatim_quote: str


class AnswerModel(BaseModel):
    vdb_fetched_answer: str
    topic: ShortTopicStr
    citations: list[LocationCitation] = Field(default_factory=list)
    answer_summary: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    is_meaning_preserved: bool
```

# 📌 Example Answer

A successful response can look like:

```json
{
  "vdb_fetched_answer": "The API Gateway implements a token bucket rate-limiting algorithm using Redis to restrict incoming requests to 100 requests per minute per authenticated user UUID.",
  "topic": "API Rate Limiting",
  "citations": [
    {
      "page_number": 4,
      "section_heading": "3.2 Traffic Management & Throttling",
      "location_fallback": "Chunk #12",
      "verbatim_quote": "The API Gateway implements a token bucket rate-limiting algorithm using Redis to restrict incoming requests to 100 requests per minute per authenticated user UUID."
    }
  ],
  "answer_summary": "The system utilizes a Redis-backed token bucket algorithm to enforce a strict rate limit of 100 requests per minute per user.",
  "confidence_score": 0.99,
  "is_meaning_preserved": true
}
```

The response is therefore simultaneously:

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

# 🔄 Complete End-to-End Architecture

The entire system can be represented as two major pipelines.

## Document Pipeline

```text
                        USER
                         │
                         ▼
                    File Upload
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
                         ▼
                   Celery Queue
                         │
                         ▼
                  Save Document
                         │
                         ▼
                  PostgreSQL Row
                         │
                         ▼
                    Docling
                         │
                         ▼
                     Chunking
                         │
                         ▼
                    Embeddings
                         │
                         ▼
                    RAW VDB
                         │
                         ▼
                  Document READY
                         │
                         ▼
               Background Multi-Index
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         Summary AI           Explanation AI
              │                     │
              ▼                     ▼
         SUMMARY VDB          EXPLANATION VDB
```

## Question Pipeline


The architecture has evolved far beyond a straight-through pipeline. When a user asks a question, it enters a multi-layered routing, context-management, caching, and retrieval workflow designed for low-latency semantic caching and stateful conversation.

```text
                                        USER
                                         │
                                         ▼
                             Question / Payload Entry
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │   Cache VDB Creation Worker           │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │   Conversation vs. Next Question      │
                     │             Classifier                │
                     └───────────────────┬───────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
          [Standalone Question]                       [Conversation Context]
                   │                                           │
                   │                                  ┌────────┴────────┐
                   │                                  │ LTM/STM VDB     │
                   │                                  │ Worker & Pipeline│
                   │                                  └────────┬────────┘
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │    3-Tier Cache Validation Check      │
                     │  (Hot Exact → Semantic → Vector VDB)  │
                     └───────────────────┬───────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
              [CACHE HIT]                                 [CACHE MISS]
                   │                                           │
                   │                                           ▼
                   │                               ┌───────────────────────┐
                   │                               │ Intent & Query Class  │
                   │                               └──────────┬────────────┘
                   │                                          │
                   │                                          ▼
                   │                               ┌───────────────────────┐
                   │                               │ Advanced Retrieval    │
                   │                               │ (HyDE, Step-Back, etc)│
                   │                               └──────────┬────────────┘
                   │                                          │
                   │                                          ▼
                   │                               ┌───────────────────────┐
                   │                               │ Hybrid (Vector + BM25)│
                   │                               └──────────┬────────────┘
                   │                                          │
                   │                                          ▼
                   │                               ┌───────────────────────┐
                   │                               │ AI Reranker & Top-K   │
                   │                               └──────────┬────────────┘
                   │                                          │
                   │                                          ▼
                   │                               ┌───────────────────────┐
                   │                               │ Answer AI Execution   │
                   │                               └──────────┬────────────┘
                   │                                          │
                   │                                          ▼
                   │                               ┌───────────────────────┐
                   │                               │ Structured Response   │
                   │                               └──────────┬────────────┘
                   │                                          │
                   │                                          ▼
                   │                               ┌───────────────────────┐
                   │                               │ Background Push to    │
                   │                               │ Cache VDB & Tier 1/2  │
                   │                               └──────────┬────────────┘
                   │                                          │
                   └─────────────────────┬────────────────────┘
                                         │
                                         ▼
                                        USER

```
---

# ⚡ Performance & Benchmarking

Talk2Docs has been benchmarked at the retrieval-pipeline level to measure the latency contribution of different retrieval techniques.

Current benchmarks were performed against a **single indexed document**. Multi-document and larger-corpus benchmarking is still underway, so these results should be treated as current single-document measurements rather than final production benchmarks.

### Retrieval Pipeline

```text
Query
  │
  ├── Retriever Construction
  └── Intent / Query Classification
             │
             ▼
      Retrieval Strategy
             │
             ▼
         AI Reranker
             │
             ▼
          Answer AI
```
**Current Benchmark:**
| Technique            | Retriever + Classifier | Retrieval Strategy |   Reranker |       Total |
| -------------------- | ---------------------: | -----------------: | ---------: | ----------: |
| Step-Back            |              739.50 ms |          826.49 ms | 1258.61 ms | **2.825 s** |
| HyDE                 |              801.31 ms |         1503.64 ms |  807.36 ms | **3.112 s** |
| Advanced Translation |              739.19 ms |          471.04 ms | 1129.38 ms | **2.340 s** |
| Query Decomposition  |              902.63 ms |          721.46 ms |  673.91 ms | **2.298 s** |
| Multi-Index          |              658.66 ms |         2760.05 ms |  645.60 ms | **4.064 s** |

Benchmark status: Current results use a single-document corpus. Multi-document, larger-corpus, and load-oriented benchmarks are still in progress.

---
# 🔐 Authentication & Security

Talk2Docs uses user-scoped access throughout the system.

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

Document access is never based solely on a user-provided identifier.

Queries are scoped against the authenticated user.

Conceptually:

```text
request_id
    +
authenticated user_id
```

This prevents one user from accessing another user's document simply by knowing a document or request identifier.

---

# 🚦 Rate Limiting

Talk2Docs uses SlowAPI for API-level rate limiting.

Example:

```python
@limiter.limit("3/minute")
```

Rate limiting protects sensitive endpoints from excessive requests and provides an additional layer of API protection.

---

### ChromaDB

Used for:

* Raw vector representations
* Summary representations
* Explanation representations

---

# 📡 Worker Status Polling

Because uploads are asynchronous, clients can monitor processing using:

```text
GET /upload_worker/{task_id}/{request_id}
```

The endpoint combines Celery/Redis worker state with PostgreSQL document state.

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

This prevents transient Celery state from being mistaken for persistent document state.

---

AI services also use structured `APIResponse` objects to distinguish:

```text
success
+
data
+
error_code
+
error_message
```

This prevents business failures from being confused with unhandled application exceptions.

---

# 🔄 AI Failure & Recovery

AI output is not blindly trusted.

Where structured output is used, Pydantic models validate the response.

For example, the reranker validates:

* Schema correctness
* Candidate IDs
* Candidate uniqueness
* Candidate completeness
* Score range
* Ranking order

If parsing fails, recovery mechanisms can attempt to extract or repair structured data.

The general philosophy is:

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
Use   Repair
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

---

# 📁 Project Structure

A simplified representation of the project:

```text
Talk2Docs/
│
├── Ai/
│   ├── ai_utils.py
│   ├── retry_logic.py
│   ├── query_classifier.py
│   ├── answer_ai.py
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
│   └── chroma.py
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

- **Python - 3.12+**
- **FastAPI** - API framework
- **Pydantic v2** - validation and structured AI output
- **SQLAlchemy 2.0 Async** - asynchronous ORM
- **Alembic** - database migrations

## AI / RAG

- **LangChain** - RAG orchestration and retrieval abstractions
- **Sentence Transformers** - embeddings
- **ChromaDB** - vector storage and semantic retrieval
- **BM25** - lexical retrieval
- **Hybrid Retrieval** - vector + BM25
- **Reciprocal Rank Fusion (RRF)** - result fusion
- **LLM Query Transformation** - Multi-Query, HyDE, Step-Back, Translation, and Decomposition
- **Cohere Encoder Reranking** - candidate reranking
- **Structured AI Output** - Pydantic-validated model responses
- **[TriCacheLLM-MMA](https://pypi.org/project/TriCacheLLM-MMA/0.1.8/)** - Custom caching and multi-index retrieval infrastructure package


## Document Processing

- **Docling** - document parsing
- **HybridChunker** - structure-aware chunking
- **Content / Signature Validation** - upload validation

## Background Processing

- **Celery** - asynchronous task execution
- **Redis** - Celery broker/backend and application infrastructure

## Database & Storage

- **PostgreSQL** - persistent application and document metadata
- **ChromaDB** - Raw, Summary, and Explanation vector indexes
- **Redis** - server-side sessions and transient state

## Authentication & Security

- **JWT**
- **OAuth2PasswordBearer**
- **Redis-backed sessions**
- **Session revocation**
- **SlowAPI** - API rate limiting
- **User-scoped document and vector retrieval**

## Infrastructure & Observability

- **Nginx** — reverse proxy
- **Structured application logging**
- **Centralized exception handling**
- **Celery worker lifecycle tracking**
- **Persistent document and multi-index status tracking**

***

# 🧪 Engineering Principles

Talk2Docs is designed as an extensible RAG system rather than a minimal
"chat with documents" application.

The architecture is guided by a few principles:

### Separation of Concerns

API routes, services, workers, retrieval, storage, and AI components have
distinct responsibilities.

### Retrieval Modularity

Retrieval techniques can be added or changed without rewriting the
downstream answer-generation pipeline.

### Common Retrieval Contract

Regardless of the retrieval strategy, downstream components receive:

```python
list[LangChainDocument]
```
---

# 📈 Current Progress

Core Backend  
* ✅ FastAPI application
* ✅ Async SQLAlchemy
* ✅ PostgreSQL
* ✅ Alembic
* ✅ JWT authentication
* ✅ Redis-backed sessions
* ✅ Session revocation
* ✅ Celery integration
* ✅ Rate limiting
* ✅ Centralized exception handling
* ✅ Structured logging
* ✅ Nginx integration


Document Pipeline  
* ✅ File validation
* ✅ File persistence
* ✅ Document metadata
* ✅ Asynchronous upload processing
* ✅ Docling parsing
* ✅ Structure-aware chunking
* ✅ Embedding generation
* ✅ Raw ChromaDB indexing
* ✅ Document state tracking
* ✅ Celery retry handling
* ✅ Worker status polling


Multi-Index Architecture  
* ✅ Raw VDB
* ✅ Summary VDB architecture
* ✅ Explanation VDB architecture
* ✅ Summary AI generation
* ✅ Explanation AI generation
* ✅ Shared `chunk_id` mapping
* ✅ Multi-index status tracking
* ✅ Background multi-index construction
* ✅ Summary hybrid retriever
* ✅ Explanation hybrid retriever
* ✅ Parallel secondary retriever construction
* ✅ Multi-index parallel retrieval


Retrieval & Caching  
* ✅ Vector retrieval
* ✅ BM25 retrieval
* ✅ Hybrid retrieval
* ✅ Ensemble retriever
* ✅ Reciprocal Rank Fusion
* ✅ Multi-Query
* ✅ HyDE
* ✅ Step-Back
* ✅ Advanced query translation
* ✅ Query decomposition
* ✅ Multi-index retrieval
* ✅ Common `list[LangChainDocument]` retrieval contract
* ✅ AI reranking
* ✅ Reranker validation and mapping
* ✅ 3-Tier Cache Engine (Hot Exact, Semantic, Vector VDB)
* ✅ Persistent Global BM25 (Redis + Pickle versioning)


Conversational Routing & Memory  
* ✅ Conversation vs. Next-Question classification
* ✅ LTM/STM VDB worker pipeline architecture
* ✅ Asynchronous background cache population (`asyncio.create_task`)


Answer Generation  
* ✅ Document-grounded answers
* ✅ Structured Pydantic output
* ✅ Source citations
* ✅ Verbatim evidence quotes
* ✅ Confidence scoring
* ✅ Meaning-preservation signal

---

# 🗺️ Roadmap

Phase 1: Core Backend  
* [x] FastAPI
* [x] PostgreSQL
* [x] Async SQLAlchemy
* [x] Alembic
* [x] JWT authentication
* [x] OAuth2PasswordBearer
* [x] Redis-backed sessions
* [x] Session revocation
* [x] Celery
* [x] Redis broker/backend
* [x] Rate limiting
* [x] Centralized exceptions
* [x] Structured logging
* [x] Nginx reverse proxy



Phase 2: Document Intelligence  
* [x] Upload validation
* [x] File persistence
* [x] Document metadata
* [x] Docling parsing
* [x] Structure-aware chunking
* [x] HybridChunker
* [x] Embeddings
* [x] Raw VDB
* [x] User-isolated vector storage
* [x] Persistent processing state
* [x] Celery worker lifecycle
* [x] Worker status polling
* [x] Failure handling
* [x] Retry handling



Phase 3: Hybrid Retrieval  
* [x] Vector retrieval
* [x] BM25 retrieval
* [x] Hybrid retrieval
* [x] Ensemble retrieval
* [x] Reciprocal Rank Fusion
* [x] User/document scoped retrieval
* [x] Common retrieval output contract
* [x] Async retrieval optimization
* [x] Retrieval benchmarking



Phase 4: Advanced RAG  
* [x] Intent classification
* [x] Query classification
* [x] Multi-Query
* [x] HyDE
* [x] Step-Back
* [x] Advanced Translation
* [x] Query Decomposition
* [x] Adaptive retrieval strategy selection
* [x] AI reranking
* [x] Structured reranker validation
* [x] Candidate validation
* [x] Top-K selection



Phase 5: Multi-Index RAG  
* [x] Raw VDB
* [x] Summary VDB
* [x] Explanation VDB
* [x] Summary AI
* [x] Explanation AI
* [x] 1:1 chunk mapping
* [x] Shared `chunk_id`
* [x] Multi-index lifecycle state
* [x] Background construction
* [x] Summary hybrid retrieval
* [x] Explanation hybrid retrieval
* [x] Parallel secondary retrieval
* [x] Parallel multi-index retrieval
* [x] Raw chunk resolution



Phase 6: Retrieval Performance  
* [x] End-to-end benchmarking
* [x] Retrieval latency profiling
* [x] Parallel independent operations
* [x] `asyncio.gather()` optimization
* [x] Synchronous retrieval offloading
* [x] BM25 construction offloading
* [x] Candidate-count optimization
* [x] Reranking optimization
* [x] Model/provider benchmarking
* [x] Multi-index latency benchmarking



Phase 7: Global BM25 & Caching  
* [x] Identify corpus-wide BM25 rebuild bottleneck
* [x] Separate BM25 from vector index lifecycle
* [x] Design user-scoped global BM25
* [x] Design background BM25 construction
* [x] Design BM25 readiness fallback
* [x] Implement persistent global BM25 (`.pkl` master artifacts)
* [x] Add BM25 lifecycle/status tracking & versioning (`v1`/`v2`)
* [x] Integrate new-document synchronization
* [x] Implement 3-Tier Caching (Hot Exact, Semantic, Vector VDB)
* [ ] Benchmark large user libraries



Phase 8: Memory & Conversational Routing  
* [ ] Conversation vs. Next-Question classifier
* [ ] LTM/STM VDB worker pipeline architecture
* [ ] Short-Term Memory (STM) full implementation
* [ ] Long-Term Memory (LTM) full implementation
* [ ] Memory-specific VDB architecture integration
* [ ] Conversation-aware retrieval routing
* [ ] User memory isolation
* [ ] Memory ranking & lifecycle
* [ ] Context-aware answer generation tuning



Phase 9: Production Hardening  
* [ ] Complete route-level logging
* [ ] Complete Celery lifecycle logging
* [ ] Remove temporary/debug comments
* [ ] Logging consistency pass
* [ ] Final sanity test suite
* [ ] Expanded document content validation
* [ ] DOCX content/signature validation
* [ ] Additional file-format validation
* [ ] Observability improvements
* [ ] AI latency profiling
* [ ] Provider benchmarking
* [ ] Failure/recovery testing
* [ ] Production deployment
* [ ] CI/CD
* [ ] Monitoring
* [ ] Load testing
---



# ❤️ Why Talk2Docs Exists

Talk2Docs started as a document-processing backend and evolved into an
end-to-end RAG system combining:

```text
Document Processing
        ↓
Hybrid Retrieval
        ↓
Adaptive Query Intelligence
        ↓
Multi-Index Retrieval
        ↓
Reranking
        ↓
Structured Grounded Answers
```
# 📜 License

MIT License

---

Built with ❤️ using:

**FastAPI · Celery · Docling · LangChain · ChromaDB · PostgreSQL · Redis ·
Pydantic · Sentence Transformers · Cohere**
