import asyncio
import json
import re
import traceback
from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LangChainDocument
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_groq import ChatGroq
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
import re

from Ai.query_and_intent_classifier import QueryClassificationResult, QueryTechnique
from Ai.raw_and_parsed_clean import extract_parsed_data, extract_raw_data
from Ai.retry_logic import check_provider_quota
from core.Exceptions.exceptions import AIServiceException
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES, USER_ERROR_CODES
from utils.logging.helper_log import LogState, log_state
from utils.logging.logEvents import (
    ExceptionLog,
    HyDELog,
    ProviderLog,
    RepairLog,
    SecurityLog,
    ServiceLog,
)
from utils.schemas import APIResponse, QuestionRequest



HYDE_SYSTEM_PROMPT = r"""You are an authoritative technical writer and factual document synthesizer.
YOUR GOAL:
Given a user question, write a single, highly dense hypothetical passage or documentation snippet (100–250 words) that directly answers the question as if it were extracted from a real reference document, textbook, or API guide.

RULES & CONSTRAINTS:
1. ZERO CONVERSATIONAL FILLER: Do NOT include greetings, introductions, or intros like "Here is...", "Based on your question...", or "This document explains...".
2. ABSOLUTE CONFIDENCE: Never say "I don't know", "As an AI...", or express uncertainty. Invent plausible, highly technical phrasing if specifics are unknown.
3. RICH KEYWORD DENSITY: Use relevant domain-specific jargon, API parameters, error codes, architectural patterns, and contextual keywords related to the question.
4. DOCUMENT FORMATTING: Write in clean, structured standard text or technical prose (e.g., standard paragraphs or key-value parameter specifications).

OUTPUT:
Return ONLY the hypothetical raw document passage text."""

HYDE_EXAMPLES = [
    {
        "question": "How does Redis handle token revocation and session invalidation in stateless API architectures?",
        "hypothetical_doc": (
            "In stateless JWT authentication architectures, Redis acts as a high-speed distributed blocklist for token revocation. "
            "Upon receiving a logout request, the server extracts the JWT key identifier (jti) and writes it to Redis with a Time-To-Live (TTL) "
            "matching the token's remaining lifespan. Incoming requests pass through an API gateway middleware that executes an async EXISTS command "
            "against the Redis key store. If the jti is found, the gateway rejects the request with an HTTP 401 Unauthorized status before routing to "
            "upstream microservices."
        ),
    },
    {
        "question": "What is Row-Level Security in PostgreSQL and how is cross-tenant data leakage prevented?",
        "hypothetical_doc": (
            "PostgreSQL Row-Level Security (RLS) restricts database query results based on the executing user's session context. "
            "By executing ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;, security policies are attached using "
            "CREATE POLICY tenant_isolation_policy ON tenants USING (tenant_id = current_setting('app.current_tenant_id'));. "
            "When an ORM or database pool connection opens, it sets the session variable via SET LOCAL app.current_tenant_id = 'tenant_123';. "
            "The PostgreSQL query engine automatically injects these predicates into execution plans, isolating cross-tenant reads and writes "
            "at the storage engine level."
        ),
    },
    {
        "question": "How do thread pool executors handle concurrent file processing and progress tracking in Python?",
        "hypothetical_doc": (
            "Python's concurrent.futures.ThreadPoolExecutor manages concurrent I/O-bound operations by dispatching tasks across a pool of worker threads. "
            "When processing bulk file downloads or parsing media links, tasks are submitted using executor.submit() or mapped via executor.map(). "
            "Progress tracking is integrated using thread-safe wrappers or event callbacks. Iterating over as_completed(futures) yields completed task "
            "futures as they finish, allowing non-blocking aggregation of results, status bar updates, and exception handling without stalling the main process thread."
        ),
    },
    {
        "question": "How to configure Nginx as a reverse proxy for upstream FastAPI services running in WSL?",
        "hypothetical_doc": (
            "Nginx routes inbound external HTTP traffic to upstream FastAPI instances running inside Windows Subsystem for Linux (WSL). "
            "The nginx.conf configuration defines an upstream backend_cluster block specifying local worker endpoints (e.g., 127.0.0.1:8000). "
            "Within the server block listening on port 80/443, a location / directive utilizes proxy_pass http://backend_cluster;. "
            "Key headers including Proxy-Set-Header Host $host; and Proxy-Set-Header X-Real-IP $remote_addr; pass original client metadata "
            "to Uvicorn while enabling WebSocket streaming and CORS preflight handling."
        ),
    },
    {
        "question": "What is hybrid search in vector databases and how does Reciprocal Rank Fusion merge BM25 with dense embeddings?",
        "hypothetical_doc": (
            "Hybrid retrieval combines sparse keyword matching (BM25) with dense vector similarity (such as cosine similarity over embeddings) "
            "to maximize recall and precision. BM25 captures exact terminology, code identifiers, and rare tokens, while dense vector search captures semantic context. "
            "Results from both retrievers are merged using Reciprocal Rank Fusion (RRF). RRF calculates candidate scores using "
            "$RRF\\_Score(d) = \\sum_{m \\in M} \\frac{1}{k + r_m(d)}$, where $r_m(d)$ represents the document's rank in retriever $m$ and $k$ "
            "is a smoothing constant (typically 60), producing a unified candidate list for downstream re-ranking."
        ),
    },
]

_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Question: {question}\n\nHypothetical Document Chunk:"),
    ("ai", "{hypothetical_doc}"),
])

_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=_example_prompt,
    examples=HYDE_EXAMPLES,
)

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", HYDE_SYSTEM_PROMPT),
    _few_shot_prompt,
    ("human", "Question: {question}\n\nHypothetical Document Chunk:"),
])

from utils.config import settings 
model = ChatGroq(
    api_key=settings.api_key,   
    model=settings.model,
)

async def HYDE_fucntion(question: str, user_id: int) -> APIResponse:
    log_state(HyDELog.HYDE_GENERATION_STARTED, function="HYDE_fucntion", user_id=user_id)
    log_state(ServiceLog.AI_SERVICE_STARTED, function="HYDE_fucntion", user_id=user_id)
    
    try:
        log_state(ProviderLog.AI_PROVIDER_REQUEST, function="HYDE_fucntion", user_id=user_id)
        log_state(ProviderLog.AI_PROVIDER_IN_PROCESSING, function="HYDE_fucntion", user_id=user_id)
        
        chain = HYDE_PROMPT | model | StrOutputParser()
        raw_result: str = await chain.ainvoke({"question": question})
        cleaned_result: str = raw_result.strip()
        
        if cleaned_result.startswith("```"):
            cleaned_result = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned_result)
            cleaned_result = re.sub(r"\n?```$", "", cleaned_result).strip()

    except Exception as e:
        log_state(ProviderLog.AI_PROVIDER_FAILURE, level=LogState.EXCEPTION, function="HYDE_fucntion", exc=e, user_id=user_id)
        log_state(ServiceLog.AI_SERVICE_FAILED, function="HYDE_fucntion", user_id=user_id)
        log_state(HyDELog.HYDE_GENERATION_FAILED, function="HYDE_fucntion", user_id=user_id)
        log_state(HyDELog.EXITING_HYDE, function="HYDE_fucntion", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="HYDE_fucntion", user_id=user_id)

        if check_provider_quota(e):
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value,
                error_message="No more tokens left to process this request"
            )

        raise AIServiceException(
            error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
            message="AI processing failed during HYDE_fucntion document generation"
        ) from e 
            
    log_state(ProviderLog.AI_PROVIDER_SUCCESS, level=LogState.INFO, function="HYDE_fucntion", user_id=user_id)
    log_state(ServiceLog.AI_SERVICE_COMPLETED, function="HYDE_fucntion", user_id=user_id)
    log_state(ServiceLog.AI_SERVICE_ENDED, function="HYDE_fucntion", user_id=user_id)
    log_state(HyDELog.HYDE_GENERATION_SUCCESS, function="HYDE_fucntion", user_id=user_id)
    log_state(HyDELog.EXITING_HYDE, function="HYDE_fucntion", user_id=user_id)
    log_state(ServiceLog.EXITING_AI_SERVICE, function="HYDE_fucntion", user_id=user_id)
    
    return APIResponse(
        success=True,
        data=cleaned_result,
        error_code=None,
        error_message=None
    )