import asyncio
import json
import traceback
from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional

from langchain_chroma import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
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

from Ai import query_classifier
from Ai.ai_utils import safe_retrieve
# from Ai.intent_classifier import get_user_intent
from Ai.query_classifier import QueryClassificationResult, QueryTechnique
from Ai.query_construction.multi_query.multi_query_fuction import _reciprocal_rank_fusion
from Ai.raw_and_parsed_clean import extract_parsed_data, extract_raw_data
from Ai.retry_logic import check_provider_quota
from core.Exceptions.exceptions import AIServiceException
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES, USER_ERROR_CODES
from utils.logging.helper_log import LogState, log_state
from utils.logging.logEvents import (
    AdvancedTranslationLog,
    ExceptionLog,
    MultiIndexLog,
    MultiQueryLog,
    ProviderLog,
    RepairLog,
    SecurityLog,
    ServiceLog,
    StepBackLog,
)
from utils.schemas import APIResponse, QuestionRequest
from sqlalchemy.ext.asyncio import AsyncSession
from collections.abc import Awaitable


def _resolve_to_original_chunk(doc: LangChainDocument) -> LangChainDocument:
    """Guarantees that a retrieved document chunk holds verbatim original text,
    restoring page_content from metadata if retrieved from Summary or Explanation indices.
    """
    doc_type = doc.metadata.get("doc_type", "raw")  #if u find value of "doc_type" get it else give me "raw" as value, in key_value pair
    
    
    if doc_type != "raw" and "raw_content" in doc.metadata:
        
        return LangChainDocument(
            page_content=doc.metadata["raw_content"],
            metadata={**doc.metadata, "doc_type": "raw"}
        )
    return doc


def _reciprocal_rank_fusion_multi_index(
    user_id: int,
    results_per_query: list[list[LangChainDocument]],
    k: int = 60,
) -> list[LangChainDocument] | None:
    """Fuses ranks across 3 search streams (Raw, Summary, Explanation)
    and strictly maps final hits to original source chunks.
    """
    log_state(MultiQueryLog.RRF_STARTED, function="_reciprocal_rank_fusion", user_id=user_id)
    try:
        fused_scores: dict[str, float] = {}
        doc_map: dict[str, LangChainDocument] = {}

        for docs in results_per_query:
            for rank, doc in enumerate(docs):
                # Unique chunk ID linking Raw, Summary, and Explanation representations
                chunk_id = doc.metadata.get("chunk_id") or str(
                    hash((doc.page_content, tuple(sorted(doc.metadata.items()))))
                )

                if chunk_id not in fused_scores:
                    fused_scores[chunk_id] = 0.0

                # Accumulate RRF score: score = sum(1 / (k + rank))
                fused_scores[chunk_id] += 1.0 / (k + rank + 1)

                # Prioritize holding the true 'raw' document if encountered
                if chunk_id not in doc_map:
                    doc_map[chunk_id] = doc
                else:
                    existing_type = doc_map[chunk_id].metadata.get("doc_type", "")
                    current_type = doc.metadata.get("doc_type", "")
                    if existing_type != "raw" and current_type == "raw":
                        doc_map[chunk_id] = doc

        
        sorted_chunks = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        resolved_original_chunks = [
            _resolve_to_original_chunk(doc_map[chunk_id]) 
            for chunk_id, _ in sorted_chunks
        ] 
        log_state(MultiQueryLog.RRF_SUCCESS, function="_reciprocal_rank_fusion", user_id=user_id)
        return resolved_original_chunks

    except Exception as exc:
        log_state(
            ExceptionLog.NO_RELATED_VECTOR_DATABASE_FOUND,
            function="_reciprocal_rank_fusion_multi_index",
            user_id=user_id,
            exc=str(exc),
        )
        log_state(MultiQueryLog.RRF_FAILED, level=LogState.EXCEPTION, function="_reciprocal_rank_fusion", exc=exc, user_id=user_id)
        return None
    
    
    
async def multi_indexing_function(
    model: Any,
    question: str,
    user_id: int,
    raw_retriever: EnsembleRetriever,          # Index A: Raw Chunks
    summary_retriever: EnsembleRetriever,      # Index B: AI Summaries
    explanation_retriever: EnsembleRetriever,  # Index C: AI Explanations
    top_n_final: int = 20,
) -> APIResponse:
    log_state(MultiIndexLog.MULTI_INDEX_STARTED, function="multi_indexing_function", user_id=user_id)

    try:
        # 1. PARALLEL SEARCHES (Search A, Search B, Search C)
        retrieval_tasks: list[Awaitable[list[LangChainDocument]]] = [
           safe_retrieve(raw_retriever, question),  # Search A
            safe_retrieve(summary_retriever, question),  # Search B
            safe_retrieve(explanation_retriever, question),  # Search C
        ]# the fetch will be form summary but metadata will hold orignal chunk ;)
        parallel_results: list[list[LangChainDocument]] = await asyncio.gather(*retrieval_tasks)

    except Exception as exc:
        log_state(ExceptionLog.NO_RELATED_VECTOR_DATABASE_FOUND, function="multi_indexing_function", user_id=user_id, exc=str(exc))
        log_state(MultiIndexLog.MULTI_INDEX_FAILED, function="multi_indexing_function", user_id=user_id)
        log_state(MultiIndexLog.EXITING_MULTI_INDEX, function="multi_indexing_function", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.INTERNAL_SYSTEM_ERROR.value,
            error_message=f"Multi-index 3-way parallel search failed: {str(exc)}",
        )

    # 2. RRF FUSION & ORIGINAL CHUNK RESOLUTION
    original_chunks = _reciprocal_rank_fusion_multi_index(user_id=user_id, results_per_query=parallel_results)
    if original_chunks is None:
        log_state(MultiIndexLog.MULTI_INDEX_FAILED, function="multi_indexing_function", user_id=user_id)
        log_state(MultiIndexLog.EXITING_MULTI_INDEX, function="multi_indexing_function", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.RRF_FAILURE.value,
            error_message="RRF fusion failed across 3 representation indices",
        )

    final_original_chunks = original_chunks[:top_n_final]

    if not final_original_chunks:
        log_state(ExceptionLog.NO_RELATED_DOCUMENT_FOUND, function="multi_indexing_function", user_id=user_id)
        log_state(MultiIndexLog.MULTI_INDEX_FAILED, function="multi_indexing_function", user_id=user_id)
        log_state(MultiIndexLog.EXITING_MULTI_INDEX, function="multi_indexing_function", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=USER_ERROR_CODES.NO_RELATED_DOCUMENT_FOUND.value,
            error_message="No documents retrieved across Raw, Summary, or Explanation indices.",
        )

    # 3. EXIT TO DOWNSTREAM (Reranker -> Answer AI)
    log_state(MultiIndexLog.MULTI_INDEX_SUCCESS, function="multi_indexing_function", user_id=user_id)
    log_state(MultiIndexLog.EXITING_MULTI_INDEX, function="multi_indexing_function", user_id=user_id)

    return APIResponse(success=True, data=final_original_chunks, error_code=None, error_message=None)