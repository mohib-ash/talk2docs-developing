from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import asyncio
from typing import Any

from langchain_classic.retrievers.ensemble import EnsembleRetriever

from Ai import query_classifier
from Ai.ai_utils import build_get_retriever
from Ai.query_classifier import QueryClassificationResult, QueryTechnique

from Ai.query_construction.HYDE.HYDE_file import HYDE_fucntion
from Ai.query_construction.HYDE.HYDE_service import use_HYDE
from Ai.query_construction.advanced_translation.advanced_translation_function import advanced_translation_function
from Ai.query_construction.multi_query.multi_query_fuction import multi_query_function
from Ai.query_construction.multi_indexing.multi_indexing_function import multi_indexing_function
from Ai.query_construction.query_decomp.query_decomp_function import query_decomposition_function
from Ai.query_construction.step_back.step_back_function import step_back_function

from db_tables.tables import Document
from utils.schemas import APIResponse, DocumentStatus, MultiIndexStatus
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES
from utils.logging.logEvents import RetrievalStrategyLog
from utils.logging.helper_log import log_state
from redis.asyncio import Redis
from vector_db.chroma import (
    get_user_summary_vdb,
    get_user_explanation_vdb,
)


    
#btw this function acts as a middle man for Answer_Ai and classification techniqyes
async def execute_retrieval_strategy(model, redis_client: Redis, question: str, classification_response: QueryClassificationResult, user_id: int, retriever: EnsembleRetriever, doc_name: list[str], db: AsyncSession) -> APIResponse:
    log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_STARTED, function="execute_retrieval_strategy", user_id=user_id)
    
    technique: QueryTechnique = classification_response.selected_technique
    confidence_score: float = classification_response.confidence_score
    
    #why this has 2? coz it doesnt contain a pydantic
    if technique == QueryTechnique.HYDE:
        result: APIResponse = await HYDE_fucntion(question=question, user_id=user_id)
        if not result.success:
            log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_FAILED, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.FALLING_BACK_TO_DEFAULT_ROUTE, function="execute_retrieval_strategy", user_id=user_id)
            return result #hyde didnt work so now ill go to defualt case and use orignal question
        
        
        result: APIResponse = await use_HYDE(user_id=user_id, hyde_doc=result.data, retriever=retriever)
        if not result.success:
            log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_FAILED, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.FALLING_BACK_TO_DEFAULT_ROUTE, function="execute_retrieval_strategy", user_id=user_id)
            return result #again go defualt case
        
        log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_SUCCESS, function="execute_retrieval_strategy", user_id=user_id)
        log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
        return result ##list[langchaindocument] of 20! senmtaic serch results -> same case is for defualt clase 
        
    
    if technique == QueryTechnique.MULTI_QUERY:
        result: APIResponse = await multi_query_function(
            model=model, 
            user_id=user_id, 
            question=question, 
            retriever=retriever
            )
        if not result.success:
            log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_FAILED, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.FALLING_BACK_TO_DEFAULT_ROUTE, function="execute_retrieval_strategy", user_id=user_id)
            return result 
        log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_SUCCESS, function="execute_retrieval_strategy", user_id=user_id)
        log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
        return result #list[langchaindocument] of 20! senmtaic serch results -> same case is for defualt clase 
    
    
    if technique == QueryTechnique.ADVANCED_TRANSLATION:
        result: APIResponse = await advanced_translation_function(
            model=model, 
            question=question, 
            user_id=user_id, 
            retriever=retriever
        )
        if not result.success:
            log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_FAILED, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.FALLING_BACK_TO_DEFAULT_ROUTE, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
            return result

        log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_SUCCESS, function="execute_retrieval_strategy", user_id=user_id)
        log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
        return result
        

    if technique == QueryTechnique.STEP_BACK:
        result: APIResponse = await step_back_function(
            model=model, 
            question=question, 
            user_id=user_id, 
            retriever=retriever
        )
        if not result.success:
            log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_FAILED, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.FALLING_BACK_TO_DEFAULT_ROUTE, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
            return result

        log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_SUCCESS, function="execute_retrieval_strategy", user_id=user_id)
        log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
        return result
        
        

    if technique == QueryTechnique.MULTI_INDEXING:
        stmt = select(Document).where(Document.user_id == user_id, Document.status == DocumentStatus.READY)
        
        if doc_name:
            stmt = stmt.where(Document.original_filename.in_(doc_name))

        result = await db.execute(stmt)
        target_docs = result.scalars().all()

        if not target_docs or (doc_name and len(target_docs) != len(set(doc_name))):
            log_state(RetrievalStrategyLog.DOCUMENT_NOT_FOUND, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_FAILED, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.DOCUMENT_NOT_FOUND.value,
                error_message="Requested document(s) not found or not in READY state."
            )

        multi_index_ready = all(
            doc.summary_vdb_status == MultiIndexStatus.READY
            and doc.explanation_vdb_status == MultiIndexStatus.READY
            for doc in target_docs
        )

        if multi_index_ready:
            log_state(RetrievalStrategyLog.EXECUTING_MULTI_INDEXING, function="execute_retrieval_strategy", user_id=user_id)

            
            summary_vdb, explain_vdb = await asyncio.gather(
                get_user_summary_vdb(user_id=user_id, db=db),
                get_user_explanation_vdb(user_id=user_id, db=db)
            )  #unlike normal await this one is: u both go i will await for both of u! instead of 1st await finish then 2nd await 

            if summary_vdb and explain_vdb:
                # Concurrently build secondary retrievers
                summary_retriever, explanation_retriever = await asyncio.gather(
                    build_get_retriever(user_vdb=summary_vdb, doc_name=doc_name, k=20, user_id=user_id, db=db, redis_client=redis_client),
                    build_get_retriever(user_vdb=explain_vdb, doc_name=doc_name, k=20, user_id=user_id, db=db, redis_client=redis_client)
                )

                if summary_retriever and explanation_retriever:
                    mi_result: APIResponse = await multi_indexing_function(
                        model=model,
                        question=question,
                        user_id=user_id,
                        raw_retriever=retriever,
                        summary_retriever=summary_retriever,
                        explanation_retriever=explanation_retriever,
                    )

                    if mi_result.success:
                        log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_SUCCESS, function="execute_retrieval_strategy", user_id=user_id)
                        log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
                        return mi_result

            # Any failure above falls through to Tier 2 logging
            log_state(RetrievalStrategyLog.MULTI_INDEX_EXECUTION_FAILED, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.FALLING_BACK_TO_MULTI_QUERY, function="execute_retrieval_strategy", user_id=user_id)
        else:
            log_state(RetrievalStrategyLog.MULTI_INDEX_NOT_READY, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.FALLING_BACK_TO_MULTI_QUERY, function="execute_retrieval_strategy", user_id=user_id)

        #coz if prediction was to do multi-index heavy level vague came so cant let it go normally we do mult-quer
        result: APIResponse = await multi_query_function(
            model=model, 
            user_id=user_id, 
            question=question, 
            retriever=retriever
        )
        
        if not result.success:
            log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_FAILED, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.FALLING_BACK_TO_DEFAULT_ROUTE, function="execute_retrieval_strategy", user_id=user_id)
            return result 

        log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_SUCCESS, function="execute_retrieval_strategy", user_id=user_id)
        log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
        return result
    
    if technique == QueryTechnique.QUERY_DECOMPOSITION:
        result: APIResponse = await query_decomposition_function(
            model=model,
            user_id=user_id,
            question=question,
            retriever=retriever,
        )

        if not result.success:
            log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_FAILED, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
            log_state(RetrievalStrategyLog.FALLING_BACK_TO_DEFAULT_ROUTE, function="execute_retrieval_strategy", user_id=user_id)
            return result

        log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_SUCCESS, function="execute_retrieval_strategy", user_id=user_id)
        log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
        return result

    if technique == QueryTechnique.NONE:
        log_state(RetrievalStrategyLog.FALLING_BACK_TO_DEFAULT_ROUTE,function="execute_retrieval_strategy", user_id=user_id)
        log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)
        return APIResponse(
            success=True,
            data=None,
            error_code=None,
            error_message=None,
        )
    
    # Default / QueryTechnique.NONE fallback
    log_state(RetrievalStrategyLog.RETRIEVAL_STRATEGY_FAILED, function="execute_retrieval_strategy", user_id=user_id)
    log_state(RetrievalStrategyLog.FALLING_BACK_TO_DEFAULT_ROUTE, function="execute_retrieval_strategy", user_id=user_id)
    log_state(RetrievalStrategyLog.EXITING_RETRIEVAL_STRATEGY, function="execute_retrieval_strategy", user_id=user_id)

    return APIResponse(
        success=False,
        data=None,
        error_code=SYSTEM_ERROR_CODES.INTERNAL_SYSTEM_ERROR.value,
        error_message="No valid retrieval strategy was executed."
    )