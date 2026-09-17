import cohere
from typing import Optional, Any
from langchain_core.documents import Document as LangChainDocument
from utils.config import settings
from utils.APIResponce_error_code_enum import USER_ERROR_CODES, SYSTEM_ERROR_CODES
from utils.schemas import APIResponse
from utils.logging.logEvents import ProviderLog, ReRanklog, SecurityLog, ServiceLog
from utils.logging.helper_log import log_state, LogState 
from Ai.retry_logic import check_provider_quota
from core.Exceptions.exceptions import AIServiceException


def init_cohere_reranker() -> cohere.ClientV2:
    """Initializes and returns the Cohere V2 client."""
    return cohere.ClientV2(
        api_key=settings.cohere_api_key
    )

client = init_cohere_reranker()
async def cohere_rerank(
    question: str, 
    user_id: int,
    received_docs: list[LangChainDocument], 
    top_k: int,
) -> APIResponse:
    """
    Enterprise wrapper for document reranking using Cohere's native cross-encoder API.
    Retains all core logging, error handling, and APIResponse contracts.
    """
    log_state(ServiceLog.AI_SERVICE_STARTED, function="cohere_rerank", user_id=user_id) 
    
    # 1. Input Validation
    if not question or not question.strip():
        log_state(SecurityLog.EMPTY_INPUT, function="cohere_rerank", user_id=user_id)
        log_state(ServiceLog.AI_SERVICE_FAILED, function="cohere_rerank", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="cohere_rerank", user_id=user_id)
        
        return APIResponse(
            success=False,
            data=None,
            error_code=USER_ERROR_CODES.EMPTY_INPUT.value,
            error_message="Input text is empty"
        )
    
    if not received_docs:
        log_state(ServiceLog.AI_SERVICE_FAILED, function="cohere_rerank", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="cohere_rerank", user_id=user_id)     
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.NO_DATA_WAS_SENT_TO_RANKER.value,
            error_message="Re-ranker got no data, meaning retriever goofed"
        )

    # 2. Extract page contents for Cohere evaluation
    documents_text = [doc.page_content for doc in received_docs]
    
    try:
        log_state(ProviderLog.AI_PROVIDER_REQUEST, function="cohere_rerank", user_id=user_id)
        log_state(ProviderLog.AI_PROVIDER_IN_PROCESSING, function="cohere_rerank", user_id=user_id)
        
        # Call Cohere's native cross-encoder rerank endpoint
        response = client.rerank(
            model=settings.cohere_rerank_model,
            query=question,
            documents=documents_text,
            top_n=min(top_k, len(documents_text)),
        )
        
        
        log_state(ProviderLog.AI_PROVIDER_SUCCESS, level=LogState.INFO, function="cohere_rerank", user_id=user_id)

    except Exception as e:
        if check_provider_quota(e):
            log_state(ProviderLog.AI_PROVIDER_FAILURE, level=LogState.EXCEPTION, function="cohere_rerank", exc=e, user_id=user_id)
            log_state(ServiceLog.AI_SERVICE_FAILED, function="cohere_rerank", user_id=user_id)
            log_state(ServiceLog.EXITING_AI_SERVICE, function="cohere_rerank", user_id=user_id)          
            
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value,
                error_message="No more tokens left to process this request"
            )
        else:
            log_state(ProviderLog.AI_PROVIDER_FAILURE, level=LogState.EXCEPTION, function="cohere_rerank", exc=e, user_id=user_id)
            log_state(ServiceLog.AI_SERVICE_FAILED, function="cohere_rerank", user_id=user_id)
            log_state(ServiceLog.EXITING_AI_SERVICE, function="cohere_rerank", user_id=user_id)   
            
            raise AIServiceException(
                error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
                message="Cohere rerank service request failed"
            ) from e

    # 3. Map Cohere results back to LangChain Documents and enrich metadata
    log_state(ReRanklog.VALIDATING_RERANK_RESULT_STARTED, function="cohere_rerank", user_id=user_id)
    
    reranked_docs: list[LangChainDocument] = []
    try:
        for result in response.results:
            original_doc = received_docs[result.index]
            
            #adding new enrty in metatdat for all in great trequlizer
            updated_metadata = dict(original_doc.metadata or {})
            updated_metadata["rerank_score"] = float(result.relevance_score)

            reranked_docs.append(
                LangChainDocument(
                    page_content=original_doc.page_content,
                    metadata=updated_metadata,
                )
            )
            
        log_state(ReRanklog.VALIDATING_RERANK_RESULT_SUCCESS, function="cohere_rerank", user_id=user_id)

    except Exception as e:
        log_state(ReRanklog.VALIDATING_RERANK_RESULT_FAILURE, level=LogState.EXCEPTION, function="cohere_rerank", exc=e, user_id=user_id)
        log_state(ServiceLog.AI_SERVICE_FAILED, function="cohere_rerank", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="cohere_rerank", user_id=user_id)
        
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
            error_message="Failed to map reranked results to documents."
        )

    log_state(ServiceLog.AI_SERVICE_COMPLETED, function="cohere_rerank", user_id=user_id)
    log_state(ServiceLog.AI_SERVICE_ENDED, function="cohere_rerank", user_id=user_id)
    log_state(ServiceLog.EXITING_AI_SERVICE, function="cohere_rerank", user_id=user_id)
    
    return APIResponse(
        success=True,
        data=reranked_docs,
        error_code=None,
        error_message=None
    )