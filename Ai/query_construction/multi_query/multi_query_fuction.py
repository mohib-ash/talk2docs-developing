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
    RepairLog,
    SecurityLog,
    ServiceLog,
)
from utils.schemas import APIResponse, QuestionRequest


class MultiQueryOutput(BaseModel):
    """Structured output schema for multi-query retrieval expansion.
    
    Generates diversified, technical search query variations optimized for 
    parallel vector database similarity search and BM25 lexical keyword matching.
    """

    model_config = ConfigDict(
        extra="forbid",  # Enforces strict schema validation for OpenAI/Anthropic function calling
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    queries: Annotated[
        list[
            Annotated[
                str,
                Field(
                    min_length=5,
                    max_length=250,
                    description="A single, standalone search query focused on specific technical jargon, API parameters, or architectural concepts.",
                ),
            ]
        ],
        Field(
            ...,
            min_length=3,
            max_length=4,
            description=(
                "A collection of exactly 3 to 4 distinct, highly targeted search query reformulations. "
                "Each query MUST explore a unique retrieval perspective: "
                "1) Dense semantic intent, "
                "2) Sparse lexical keywords (exact terms, errors, syntax), "
                "3) Component/architectural level abstraction, and "
                "4) Alternative domain terminology or synonyms. "
                "Do NOT include conversational prefixes, numbering ('1.', 'Query:'), or explanation text."
            ),
            examples=[
                [
                    "PostgreSQL Row-Level Security tenant isolation policy syntax",
                    "ALTER TABLE ENABLE ROW LEVEL SECURITY create policy current_setting",
                    "prevent cross-tenant data leaks database session variable isolation",
                    "Postgres multi-tenancy SECURITY DEFINER current_tenant_id filter",
                ]
            ],
        ),
    ]

    @field_validator("queries")
    @classmethod
    def validate_and_sanitize_queries(cls, queries: list[str]) -> list[str]:
        """Post-processing validator to enforce deduplication, strip LLM numbering artifacts,
        and guarantee non-empty queries.
        """
        sanitized: list[str] = []
        seen: set[str] = set()

        for q in queries:
            cleaned = q.strip()

            # Strip accidental LLM markdown numbering (e.g., "1. Query", "Query 1:")
            if cleaned and cleaned[0].isdigit() and (cleaned[1:3] in (". ", ") ", ": ")):
                cleaned = cleaned[3:].strip()
            elif cleaned.lower().startswith("query:"):
                cleaned = cleaned[6:].strip()

            # Deduplicate while preserving rank order
            cleaned_lower = cleaned.lower()
            if cleaned and cleaned_lower not in seen:
                seen.add(cleaned_lower)
                sanitized.append(cleaned)

        if len(sanitized) < 3:
            raise ValueError(
                f"Multi-query expansion generated fewer than 3 unique valid queries. Got: {len(sanitized)}"
            )

        return sanitized[:4]


MULTI_QUERY_SYSTEM_PROMPT = r"""You are an expert search query optimization engineer and document retrieval specialist.
YOUR GOAL:
Given a user question, generate 3 to 4 distinct, highly targeted search query reformulations designed to maximize document chunk recall across both vector similarity search and BM25 lexical keyword search.

RULES & CONSTRAINTS:
1. ZERO CONVERSATIONAL FILLER: Do NOT include greetings, introductions, explanations, or meta-commentary like "Here are the queries...".
2. DIVERSITY OF PERSPECTIVES: Each query MUST target a distinct retrieval angle:
   - Perspective A (Semantic Intent): Focus on core conceptual meaning.
   - Perspective B (Sparse Lexical/Keywords): Focus on exact terminology, project names, technical phrasing, or key noun phrases from the query.
   - Perspective C (Component/Structure): Focus on underlying system components, structural breakdowns, or implementation outlines.
   - Perspective D (Synonyms/Phrasing): Focus on alternative phrasing, domain terminology, or related concepts.
3. NO NUMBERING OR PREFIXES: Do NOT include numbers ("1.", "2."), bullet points, or labels ("Query 1:"). Return raw, clean query strings only.
4. TERM ENRICHMENT: Map human language into specific domain vocabulary, technical concepts, and structural blueprints.
5. ABSOLUTE PRESERVATION OF INTENT: Every variation must target the exact same underlying user question without semantic drift.

{format_instructions}
"""

MULTI_QUERY_EXAMPLES = [
    {
        "question": "What AI and software projects am I planning to build, and what are the main ideas behind them?",
        "generated_queries": json.dumps([
            "software engineering project roadmap and planned AI applications portfolio",
            "planned development ideas architecture feature set tech stack",
            "AI application feature design upcoming personal coding projects",
            "repository planning core concepts system architecture software roadmap"
        ]),
    },
    {
        "question": "How does Redis handle token revocation and session invalidation in stateless API architectures?",
        "generated_queries": json.dumps([
            "Redis JWT blacklist jti expiry time to live TTL session management",
            "Invalidate user tokens stateless API gateway Redis commands",
            "Redis EXISTS check middleware token revocation microservices authentication",
            "Revoking active JSON Web Tokens using Redis distributed cache store"
        ]),
    },
    {
        "question": "What is Row-Level Security in PostgreSQL and how is cross-tenant data leakage prevented?",
        "generated_queries": json.dumps([
            "PostgreSQL current_setting app.current_tenant_id RLS policy syntax",
            "Prevent multi tenant data isolation leaks Postgres database level",
            "CREATE POLICY tenant_isolation_policy USING tenant_id check",
            "Enforce row level security Postgres ORM connection pooling session variable"
        ]),
    },
    {
        "question": "How do thread pool executors handle concurrent file processing and progress tracking in Python?",
        "generated_queries": json.dumps([
            "concurrent futures ThreadPoolExecutor submit map Python file I/O",
            "Python thread pool progress bar track as_completed futures",
            "Non blocking concurrent media file downloader thread pool executor",
            "Handling exceptions worker threads Python ThreadPoolExecutor progress tracking"
        ]),
    },
]

_mq_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Question: {question}\n\nTarget Query Variations:"),
    ("ai", "{generated_queries}"),
])

_mq_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=_mq_example_prompt,
    examples=MULTI_QUERY_EXAMPLES,
)


def _reciprocal_rank_fusion(user_id: int, results_per_query: list[list[LangChainDocument]], k: int = 60) -> list[LangChainDocument] | None:
    log_state(MultiQueryLog.RRF_STARTED, function="_reciprocal_rank_fusion", user_id=user_id)
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

                if chunk_id not in doc_map:
                    doc_map[chunk_id] = doc
                    fused_scores[chunk_id] = 0.0

                fused_scores[chunk_id] += 1.0 / (k + rank)
                
        reranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        final_docs = [doc_map[doc_id] for doc_id in reranked_ids]
    except Exception as exc:
        log_state(MultiQueryLog.RRF_FAILED, level=LogState.EXCEPTION, function="_reciprocal_rank_fusion", exc=exc, user_id=user_id)
        return None
    log_state(MultiQueryLog.RRF_SUCCESS, function="_reciprocal_rank_fusion", user_id=user_id)
    return final_docs


async def multi_query_function(model: Any, question: str, user_id: int, retriever: EnsembleRetriever, top_n_final: int = 20) -> APIResponse:
    log_state(MultiQueryLog.MULTI_QUERY_STARTED, function="multi_query_function", user_id=user_id)
    log_state(ServiceLog.AI_SERVICE_STARTED, function="multi_query_function", user_id=user_id)

    # 1. Setup parser and prompt template
    parser = PydanticOutputParser(pydantic_object=MultiQueryOutput)

    multi_query_prompt = ChatPromptTemplate.from_messages([
        ("system", MULTI_QUERY_SYSTEM_PROMPT),
        _mq_few_shot_prompt,
        ("human", "Question: {question}\n\nTarget Query Variations:"),
    ]).partial(format_instructions=parser.get_format_instructions())

    raw_response = None
    extracted_parsed = None

    try:
        log_state(ProviderLog.AI_PROVIDER_REQUEST, function="multi_query_function", user_id=user_id)
        log_state(ProviderLog.AI_PROVIDER_IN_PROCESSING, function="multi_query_function", user_id=user_id)

        raw_response = await (multi_query_prompt | model).ainvoke({"question": question})

        cleaned_content = raw_response.content.strip()
        if cleaned_content.startswith("```"):
            cleaned_content = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned_content)
            cleaned_content = re.sub(r"\n?```$", "", cleaned_content).strip()

        extracted_parsed = parser.parse(cleaned_content)
        log_state(ProviderLog.AI_PROVIDER_SUCCESS, level=LogState.INFO, function="multi_query_function", user_id=user_id)

    except Exception as e:
        if check_provider_quota(e):
            log_state(ProviderLog.AI_PROVIDER_FAILURE, level=LogState.EXCEPTION, function="multi_query_function", exc=e, user_id=user_id)
            log_state(ServiceLog.AI_SERVICE_FAILED, function="multi_query_function", user_id=user_id)
            log_state(MultiQueryLog.MULTI_QUERY_FAILED, function="multi_query_function", user_id=user_id)
            log_state(MultiQueryLog.EXITING_MULTI_QUERY, function="multi_query_function", user_id=user_id)
            log_state(ServiceLog.EXITING_AI_SERVICE, function="multi_query_function", user_id=user_id)

            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value,
                error_message="No more tokens left to process this request"
            )
        else:
            log_state(ProviderLog.AI_PROVIDER_FAILURE, level=LogState.EXCEPTION, function="multi_query_function", exc=e, user_id=user_id)
            extracted_parsed = None

    expanded_queries: list[str] = []

    # Branch 1: Structured parsing succeeded
    if extracted_parsed and extracted_parsed.queries:
        expanded_queries = extracted_parsed.queries 

    # Branch 2: Structured parsing failed -> Fallback to Raw Repair
    else:
        log_state(RepairLog.AI_REPAIR_INITIALIZED, function="multi_query_function", user_id=user_id)
        raw = getattr(raw_response, "content", None) if raw_response else None

        if raw is None:
            log_state(ServiceLog.AI_SERVICE_FAILED, level=LogState.WARNING, function="multi_query_function", user_id=user_id)
            log_state(RepairLog.AI_REPAIR_INITIALIZATION_STOPPED, level=LogState.WARNING, function="multi_query_function", user_id=user_id)
            log_state(MultiQueryLog.MULTI_QUERY_FAILED, function="multi_query_function", user_id=user_id)
            log_state(MultiQueryLog.EXITING_MULTI_QUERY, function="multi_query_function", user_id=user_id)
            log_state(ServiceLog.EXITING_AI_SERVICE, level=LogState.WARNING, function="multi_query_function", user_id=user_id)
            return APIResponse(success=False, data=None, error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value, error_message="Structured output parsing failed and manual parsing came up empty")

        try:
            log_state(RepairLog.AI_REPAIR_STARTED, function="multi_query_function", user_id=user_id)
            log_state(RepairLog.AI_REPAIR_IN_PROGRESS, function="multi_query_function", user_id=user_id)

            # Target schema correctly set to MultiQueryOutput
            recovered = await extract_raw_data(raw, parser, model, question, MultiQueryOutput)        
        
        except Exception as e:
            if check_provider_quota(e):
                log_state(ServiceLog.AI_MY_QUOTA_REACHED, level=LogState.EXCEPTION, function="multi_query_function", exc=e, user_id=user_id)
                log_state(RepairLog.AI_REPAIR_PREMATURELY_ENDED, function="multi_query_function", user_id=user_id)
                log_state(ServiceLog.AI_SERVICE_FAILED, function="multi_query_function", user_id=user_id)
                log_state(MultiQueryLog.MULTI_QUERY_FAILED, function="multi_query_function", user_id=user_id)
                log_state(MultiQueryLog.EXITING_MULTI_QUERY, function="multi_query_function", user_id=user_id)
                log_state(ServiceLog.EXITING_AI_SERVICE, function="multi_query_function", user_id=user_id)
                return APIResponse(success=False, data=None, error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value, error_message="No more tokens left to process this request")
            else:
                log_state(RepairLog.AI_REPAIR_PREMATURELY_ENDED, level=LogState.EXCEPTION, function="multi_query_function", exc=e, user_id=user_id)
                log_state(ServiceLog.AI_SERVICE_FAILED, function="multi_query_function", user_id=user_id)
                log_state(MultiQueryLog.MULTI_QUERY_FAILED, function="multi_query_function", user_id=user_id)
                log_state(MultiQueryLog.EXITING_MULTI_QUERY, function="multi_query_function", user_id=user_id)
                log_state(ServiceLog.EXITING_AI_SERVICE, function="multi_query_function", user_id=user_id)
                raise AIServiceException(error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value, message="AI output recovery process failed") from e

        if recovered is None or not getattr(recovered, "queries", None):
            log_state(RepairLog.AI_REPAIR_FAILED, function="multi_query_function", user_id=user_id)
            log_state(ServiceLog.AI_SERVICE_FAILED, function="multi_query_function", user_id=user_id)
            log_state(MultiQueryLog.MULTI_QUERY_FAILED, function="multi_query_function", user_id=user_id)
            log_state(MultiQueryLog.EXITING_MULTI_QUERY, function="multi_query_function", user_id=user_id)
            log_state(ServiceLog.EXITING_AI_SERVICE, function="multi_query_function", user_id=user_id)
            return APIResponse(success=False, data=None, error_code=SYSTEM_ERROR_CODES.RAW_REPAIR_FAILURE.value, error_message="Structured output parsing failed and manual recovery returned no result.")

        log_state(RepairLog.AI_REPAIR_SUCCESS, function="multi_query_function", user_id=user_id)
        expanded_queries = recovered.queries


    

    if question not in expanded_queries: 
        expanded_queries.append(question)


    # 3. Asynchronous parallel vector retrieval
    try:
        log_state(MultiQueryLog.MULTI_QUERY_RETRIEVAL_STARTED, function="multi_query_function", user_id=user_id)
    
        retrieval_tasks: list[Awaitable[list[LangChainDocument]]] = [safe_retrieve(retriever, query) for query in expanded_queries] 
        multi_query_results: list[list[LangChainDocument]] = await asyncio.gather(*retrieval_tasks) 
        log_state(MultiQueryLog.MULTI_QUERY_RETRIEVAL_SUCCESS, function="multi_query_function", user_id=user_id)
        

    except Exception as exc:
        log_state(ExceptionLog.NO_RELATED_VECTOR_DATABASE_FOUND, function="multi_query_function", user_id=user_id, exc=str(exc))
        log_state(ServiceLog.AI_SERVICE_FAILED, function="multi_query_function", user_id=user_id)
        log_state(MultiQueryLog.MULTI_QUERY_RETRIEVAL_FAILED, function="multi_query_function", user_id=user_id)
        log_state(MultiQueryLog.MULTI_QUERY_FAILED, function="multi_query_function", user_id=user_id)
        log_state(MultiQueryLog.EXITING_MULTI_QUERY, function="multi_query_function", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="multi_query_function", user_id=user_id)
        return APIResponse(success=False, data=None, error_code=SYSTEM_ERROR_CODES.INTERNAL_SYSTEM_ERROR.value, error_message=f"Parallel vector retrieval failed: {str(exc)}")


    # 4. Reciprocal Rank Fusion & Trimming
    fused_documents = _reciprocal_rank_fusion(user_id=user_id, results_per_query=multi_query_results)
    if fused_documents is None:
        log_state(MultiQueryLog.MULTI_QUERY_FAILED, function="multi_query_function", user_id=user_id)
        log_state(ServiceLog.AI_SERVICE_FAILED, function="multi_query_function", user_id=user_id)
        log_state(MultiQueryLog.EXITING_MULTI_QUERY, function="multi_query_function", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="multi_query_function", user_id=user_id)
        return APIResponse(success=False, data=None, error_code=SYSTEM_ERROR_CODES.RRF_FALIURE.value, error_message="RRF unexpectaly failed")
    final_documents = fused_documents[:top_n_final] #only return 20!

    # 5. Empty retrieval check
    if not final_documents:
        log_state(ExceptionLog.NO_RELATED_DOCUMENT_FOUND, function="multi_query_function", user_id=user_id)
        log_state(ServiceLog.AI_SERVICE_FAILED, function="multi_query_function", user_id=user_id)
        log_state(MultiQueryLog.MULTI_QUERY_FAILED, function="multi_query_function", user_id=user_id)
        log_state(MultiQueryLog.EXITING_MULTI_QUERY, function="multi_query_function", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="multi_query_function", user_id=user_id)
        return APIResponse(success=False, data=None, error_code=USER_ERROR_CODES.NO_RELATED_DOCUMENT_FOUND.value, error_message="No relevant document chunks retrieved across expanded query paths.")

    # 6. Success Contract & Exit Telemetry
    log_state(ServiceLog.AI_SERVICE_COMPLETED, function="multi_query_function", user_id=user_id)
    log_state(MultiQueryLog.MULTI_QUERY_SUCCESS, function="multi_query_function", user_id=user_id)
    log_state(MultiQueryLog.EXITING_MULTI_QUERY, function="multi_query_function", user_id=user_id)
    log_state(ServiceLog.EXITING_AI_SERVICE, function="multi_query_function", user_id=user_id)

    return APIResponse(success=True, data=final_documents, error_code=None, error_message=None)