import asyncio
import json
import re
import traceback
from collections.abc import Awaitable
from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document as LangChainDocument
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from Ai.ai_utils import safe_retrieve
from Ai.raw_and_parsed_clean import extract_parsed_data, extract_raw_data
from Ai.retry_logic import check_provider_quota
from core.Exceptions.exceptions import AIServiceException
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES, USER_ERROR_CODES
from utils.logging.helper_log import LogState, log_state
from utils.logging.logEvents import (
    ExceptionLog,
    MultiQueryLog,
    ProviderLog,
    QueryDecompositionLog,
    RepairLog,
    SecurityLog,
    ServiceLog,
)
from utils.schemas import APIResponse, QuestionRequest


class QueryDecompositionOutput(BaseModel):
    """Structured output schema for breaking complex, multi-part technical questions
    into distinct, sequential sub-queries for targeted vector database retrieval.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    sub_queries: Annotated[
        list[
            Annotated[
                str,
                Field(
                    min_length=5,
                    max_length=250,
                    description="A distinct, single-topic technical sub-query focusing on an isolated component or logical sub-problem.",
                ),
            ]
        ],
        Field(
            ...,
            min_length=2,
            max_length=5,
            description=(
                "A collection of 2 to 5 independent sub-queries derived from decomposing a complex, multi-part user question. "
                "Each sub-query MUST isolate a single prerequisite concept, component architecture, or specific mechanism. "
                "Do NOT include conversational prefixes, numbering, or explanatory text."
            ),
            examples=[
                [
                    "FastAPI dependency injection yield database session lifecycle",
                    "SQLAlchemy async session engine setup asyncpg",
                    "Handling database transaction rollback in FastAPI middleware",
                ]
            ],
        ),
    ]

    @field_validator("sub_queries")
    @classmethod
    def validate_and_sanitize_sub_queries(cls, sub_queries: list[str]) -> list[str]:
        """Post-processing validator to enforce deduplication, strip LLM numbering artifacts,
        and guarantee non-empty sub-queries.
        """
        sanitized: list[str] = []
        seen: set[str] = set()

        for q in sub_queries:
            cleaned = q.strip()

            # Strip LLM markdown numbering artifacts
            if cleaned and cleaned[0].isdigit() and (cleaned[1:3] in (". ", ") ", ": ")):
                cleaned = cleaned[3:].strip()
            elif cleaned.lower().startswith("sub-query:"):
                cleaned = cleaned[10:].strip()

            cleaned_lower = cleaned.lower()
            if cleaned and cleaned_lower not in seen:
                seen.add(cleaned_lower)
                sanitized.append(cleaned)

        if len(sanitized) < 2:
            raise ValueError(
                f"Query decomposition generated fewer than 2 unique valid sub-queries. Got: {len(sanitized)}"
            )

        return sanitized[:5]


DECOMPOSITION_SYSTEM_PROMPT = r"""You are an expert search query optimization engineer and document retrieval specialist.
YOUR GOAL:
Decompose complex, multi-part technical user questions into 2 to 5 standalone sub-queries. Each sub-query must isolate a single logical aspect or prerequisite component required to answer the complete question.

RULES & CONSTRAINTS:
1. ZERO CONVERSATIONAL FILLER: Do NOT include greetings, introductions, or meta-commentary like "Here are the sub-queries...".
2. LOGICAL ISOLATION: Break compound/multi-part questions down into single-topic sub-queries so vector retrieval can hit precise document chunks.
3. STANDALONE QUALITY: Every sub-query must be fully self-contained with explicit technical nouns (avoid vague pronouns like "it", "this", or "that").
4. NO NUMBERING OR PREFIXES: Return clean, raw search strings only without labels or numbers.
5. NO SEMANTIC LOSS: The combination of all generated sub-queries must cover every requirement present in the original question.

{format_instructions}
"""

DECOMPOSITION_EXAMPLES = [
    {
        "question": "How do I setup JWT authentication in FastAPI with Redis token blacklisting and SQLAlchemy async sessions?",
        "generated_sub_queries": (
            "FastAPI OAuth2PasswordBearer JWT authentication setup dependency\n"
            "Redis key value store JWT token revocation blacklist TTL\n"
            "SQLAlchemy AsyncSession asyncpg engine setup FastAPI lifecycle"
        ),
    },
    {
        "question": "Compare PostgreSQL Row-Level Security with application-level filtering for multi-tenant isolation, including performance trade-offs.",
        "generated_sub_queries": (
            "PostgreSQL Row-Level Security CREATE POLICY tenant_isolation\n"
            "Application level multi tenant data filtering ORM tenant_id\n"
            "Postgres RLS index usage query performance latency benchmarks"
        ),
    },
    {
        "question": "How does Celery process asynchronous tasks with Redis as a broker, and how are dead letter queues handled on task failure?",
        "generated_sub_queries": (
            "Celery distributed task queue worker Redis broker architecture\n"
            "Celery task failure retry policy dead letter queue handling"
        ),
    },
]

_decomp_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Question: {question}\n\nDecomposed Sub-Queries:"),
    ("ai", "{generated_sub_queries}"),
])

_decomp_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=_decomp_example_prompt,
    examples=DECOMPOSITION_EXAMPLES,
)


def _reciprocal_rank_fusion(user_id: int, results_per_query: list[list[LangChainDocument]], k: int = 60) -> list[LangChainDocument] | None:
    log_state(QueryDecompositionLog.RRF_STARTED, function="_reciprocal_rank_fusion", user_id=user_id)
    """Combines multiple ranked document lists into a single consensus-ranked list using RRF."""
    try:
        fused_scores: dict[str, float] = {}
        doc_map: dict[str, LangChainDocument] = {}

        for doc_list in results_per_query:
            for rank, doc in enumerate(doc_list, start=1):
                # Unique identifier using metadata or content hash
                chunk_id = doc.metadata.get("chunk_id") or str(hash((
                    doc.page_content,
                    doc.metadata.get("source", ""),
                    doc.metadata.get("page", ""),
                )))
                """
                    this is either get chunk_id of form embedding's metadata OR of they dont got no id, get all rest and use em to make hash
                    so we have a way to work around
                """


                #if we have not seen a chunk then add it to dict, if we have then move onn 
                if chunk_id not in doc_map:
                    doc_map[chunk_id] = doc
                    fused_scores[chunk_id] = 0.0

                fused_scores[chunk_id] += 1.0 / (k + rank)
                
        reranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        final_docs = [doc_map[doc_id] for doc_id in reranked_ids]
    except Exception as exc:
        log_state(QueryDecompositionLog.RRF_FAILED, level=LogState.EXCEPTION, function="_reciprocal_rank_fusion", exc=exc, user_id=user_id)
        return None
    log_state(QueryDecompositionLog.RRF_SUCCESS, function="_reciprocal_rank_fusion", user_id=user_id)
    return final_docs


async def query_decomposition_function(model: Any, question: str, user_id: int, retriever: EnsembleRetriever, top_n_final: int = 20) -> APIResponse:
    
    log_state(QueryDecompositionLog.QUERY_DECOMPOSITION_STARTED, function="query_decomposition_function", user_id=user_id)
    log_state(ServiceLog.AI_SERVICE_STARTED, function="query_decomposition_function", user_id=user_id)

    # 1. Setup parser and prompt template
    parser = PydanticOutputParser(pydantic_object=QueryDecompositionOutput)

    decomposition_prompt = ChatPromptTemplate.from_messages([
        ("system", DECOMPOSITION_SYSTEM_PROMPT),
        _decomp_few_shot_prompt,
        ("human", "Question: {question}\n\nDecomposed Sub-Queries:"),
    ]).partial(format_instructions=parser.get_format_instructions())

    raw_response = None
    extracted_parsed = None

    try:
        log_state(ProviderLog.AI_PROVIDER_REQUEST, function="query_decomposition_function", user_id=user_id)
        log_state(ProviderLog.AI_PROVIDER_IN_PROCESSING, function="query_decomposition_function", user_id=user_id)

        raw_response = await (decomposition_prompt | model).ainvoke({"question": question})

        cleaned_content = raw_response.content.strip()
        if cleaned_content.startswith("```"):
            cleaned_content = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned_content)
            cleaned_content = re.sub(r"\n?```$", "", cleaned_content).strip()

        extracted_parsed = parser.parse(cleaned_content)

        # FIXED: Success is logged strictly here when both invocation and parsing succeed
        log_state(ProviderLog.AI_PROVIDER_SUCCESS, level=LogState.INFO, function="query_decomposition_function", user_id=user_id)

    except Exception as e:
        if check_provider_quota(e):
            log_state(ProviderLog.AI_PROVIDER_FAILURE, level=LogState.EXCEPTION, function="query_decomposition_function", exc=e, user_id=user_id)
            log_state(ServiceLog.AI_SERVICE_FAILED, function="query_decomposition_function", user_id=user_id)
            log_state(QueryDecompositionLog.QUERY_DECOMPOSITION_FAILED, function="query_decomposition_function", user_id=user_id)
            log_state(QueryDecompositionLog.EXITING_QUERY_DECOMPOSITION, function="query_decomposition_function", user_id=user_id)
            log_state(ServiceLog.EXITING_AI_SERVICE, function="query_decomposition_function", user_id=user_id)

            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value,
                error_message="No more tokens left to process this request",
            )
        else:
            log_state(ProviderLog.AI_PROVIDER_FAILURE, level=LogState.EXCEPTION, function="query_decomposition_function", exc=e, user_id=user_id)
            extracted_parsed = None

    decomposed_queries: list[str] = []

    # Branch 1: Structured parsing succeeded
    if extracted_parsed and extracted_parsed.sub_queries:
        decomposed_queries = extracted_parsed.sub_queries

    # Branch 2: Structured parsing failed -> Fallback to Raw Repair
    else:
        log_state(RepairLog.AI_REPAIR_INITIALIZED, function="query_decomposition_function", user_id=user_id)
        raw = getattr(raw_response, "content", None) if raw_response else None

        if raw is None:
            log_state(ServiceLog.AI_SERVICE_FAILED, level=LogState.WARNING, function="query_decomposition_function", user_id=user_id)
            log_state(RepairLog.AI_REPAIR_INITIALIZATION_STOPPED, level=LogState.WARNING, function="query_decomposition_function", user_id=user_id)
            log_state(QueryDecompositionLog.QUERY_DECOMPOSITION_FAILED, function="query_decomposition_function", user_id=user_id)
            log_state(QueryDecompositionLog.EXITING_QUERY_DECOMPOSITION, function="query_decomposition_function", user_id=user_id)
            log_state(ServiceLog.EXITING_AI_SERVICE, level=LogState.WARNING, function="query_decomposition_function", user_id=user_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
                error_message="Structured output parsing failed and manual parsing came up empty",
            )

        try:
            log_state(RepairLog.AI_REPAIR_STARTED, function="query_decomposition_function", user_id=user_id)
            log_state(RepairLog.AI_REPAIR_IN_PROGRESS, function="query_decomposition_function", user_id=user_id)

            recovered = await extract_raw_data(raw, parser, model, question, QueryDecompositionOutput)

        except Exception as e:
            if check_provider_quota(e):
                log_state(ServiceLog.AI_MY_QUOTA_REACHED, level=LogState.EXCEPTION, function="query_decomposition_function", exc=e, user_id=user_id)
                log_state(RepairLog.AI_REPAIR_PREMATURELY_ENDED, function="query_decomposition_function", user_id=user_id)
                log_state(ServiceLog.AI_SERVICE_FAILED, function="query_decomposition_function", user_id=user_id)
                log_state(QueryDecompositionLog.QUERY_DECOMPOSITION_FAILED, function="query_decomposition_function", user_id=user_id)
                log_state(QueryDecompositionLog.EXITING_QUERY_DECOMPOSITION, function="query_decomposition_function", user_id=user_id)
                log_state(ServiceLog.EXITING_AI_SERVICE, function="query_decomposition_function", user_id=user_id)
                return APIResponse(
                    success=False,
                    data=None,
                    error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value,
                    error_message="No more tokens left to process this request",
                )
            else:
                log_state(RepairLog.AI_REPAIR_PREMATURELY_ENDED, level=LogState.EXCEPTION, function="query_decomposition_function", exc=e, user_id=user_id)
                log_state(ServiceLog.AI_SERVICE_FAILED, function="query_decomposition_function", user_id=user_id)
                log_state(QueryDecompositionLog.QUERY_DECOMPOSITION_FAILED, function="query_decomposition_function", user_id=user_id)
                log_state(QueryDecompositionLog.EXITING_QUERY_DECOMPOSITION, function="query_decomposition_function", user_id=user_id)
                log_state(ServiceLog.EXITING_AI_SERVICE, function="query_decomposition_function", user_id=user_id)
                raise AIServiceException(
                    error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
                    message="AI output recovery process failed during query decomposition",
                ) from e

        if recovered is None or not getattr(recovered, "sub_queries", None):
            log_state(RepairLog.AI_REPAIR_FAILED, function="query_decomposition_function", user_id=user_id)
            log_state(ServiceLog.AI_SERVICE_FAILED, function="query_decomposition_function", user_id=user_id)
            log_state(QueryDecompositionLog.QUERY_DECOMPOSITION_FAILED, function="query_decomposition_function", user_id=user_id)
            log_state(QueryDecompositionLog.EXITING_QUERY_DECOMPOSITION, function="query_decomposition_function", user_id=user_id)
            log_state(ServiceLog.EXITING_AI_SERVICE, function="query_decomposition_function", user_id=user_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.RAW_REPAIR_FAILURE.value,
                error_message="Structured output parsing failed and manual recovery returned no result.",
            )

        log_state(RepairLog.AI_REPAIR_SUCCESS, function="query_decomposition_function", user_id=user_id)
        decomposed_queries = recovered.sub_queries

    # Guarantee the original user prompt is present to prevent context omission
    if question not in decomposed_queries:
        decomposed_queries.append(question)

    # 3. Asynchronous parallel vector retrieval across decomposed query paths
    try:
        log_state(QueryDecompositionLog.DECOMPOSITION_RETRIEVAL_STARTED, function="query_decomposition_function", user_id=user_id)
        retrieval_tasks: list[Awaitable[list[LangChainDocument]]] = [
            safe_retrieve(retriever, query) for query in decomposed_queries
        ]

        decomposition_results: list[list[LangChainDocument]] = await asyncio.gather(*retrieval_tasks)
        log_state(QueryDecompositionLog.DECOMPOSITION_RETRIEVAL_SUCCESS, function="query_decomposition_function", user_id=user_id)

    except Exception as exc:
        log_state(ExceptionLog.NO_RELATED_VECTOR_DATABASE_FOUND, function="query_decomposition_function", user_id=user_id, exc=str(exc))
        log_state(ServiceLog.AI_SERVICE_FAILED, function="query_decomposition_function", user_id=user_id)
        log_state(QueryDecompositionLog.DECOMPOSITION_RETRIEVAL_FAILED, function="query_decomposition_function", user_id=user_id)
        log_state(QueryDecompositionLog.QUERY_DECOMPOSITION_FAILED, function="query_decomposition_function", user_id=user_id)
        log_state(QueryDecompositionLog.EXITING_QUERY_DECOMPOSITION, function="query_decomposition_function", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="query_decomposition_function", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.INTERNAL_SYSTEM_ERROR.value,
            error_message=f"Parallel vector retrieval failed: {str(exc)}",
        )

    # 4. Reciprocal Rank Fusion & Trimming
    fused_documents = _reciprocal_rank_fusion(user_id=user_id, results_per_query=decomposition_results)
    if fused_documents is None:
        log_state(QueryDecompositionLog.QUERY_DECOMPOSITION_FAILED, function="query_decomposition_function", user_id=user_id)
        log_state(ServiceLog.AI_SERVICE_FAILED, function="query_decomposition_function", user_id=user_id)
        log_state(QueryDecompositionLog.EXITING_QUERY_DECOMPOSITION, function="query_decomposition_function", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="query_decomposition_function", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.RRF_FALIURE.value,
            error_message="RRF unexpectedly failed",
        )
    
    final_documents = fused_documents[:top_n_final]

    # 5. Empty retrieval check
    if not final_documents:
        log_state(ExceptionLog.NO_RELATED_DOCUMENT_FOUND, function="query_decomposition_function", user_id=user_id)
        log_state(ServiceLog.AI_SERVICE_FAILED, function="query_decomposition_function", user_id=user_id)
        log_state(QueryDecompositionLog.QUERY_DECOMPOSITION_FAILED, function="query_decomposition_function", user_id=user_id)
        log_state(QueryDecompositionLog.EXITING_QUERY_DECOMPOSITION, function="query_decomposition_function", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="query_decomposition_function", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=USER_ERROR_CODES.NO_RELATED_DOCUMENT_FOUND.value,
            error_message="No relevant document chunks retrieved across decomposed query paths.",
        )

    # 6. Success Contract & Exit Telemetry
    log_state(ServiceLog.AI_SERVICE_COMPLETED, function="query_decomposition_function", user_id=user_id)
    log_state(QueryDecompositionLog.QUERY_DECOMPOSITION_SUCCESS, function="query_decomposition_function", user_id=user_id)
    log_state(QueryDecompositionLog.EXITING_QUERY_DECOMPOSITION, function="query_decomposition_function", user_id=user_id)
    log_state(ServiceLog.EXITING_AI_SERVICE, function="query_decomposition_function", user_id=user_id)

    return APIResponse(success=True, data=final_documents, error_code=None, error_message=None)