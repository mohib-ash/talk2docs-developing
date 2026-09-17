import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
from typing import Any
from celery.result import AsyncResult
import httpx
from langchain_chroma import Chroma
from sqlalchemy import select
# from celery_worker.tasks.Ai_worker.Ai_worker_utils import get_worker_result
# from celery_worker.tasks.worker_utils import worker_result_handler
from langchain_community.retrievers import BM25Retriever
from Ai.explain_ai import ExplanationBatchModel, explanation_ai
from Ai.summry_ai import SummaryBatchModel, summry_ai
from core.Exceptions.exceptions import (
ChunkingParsedFileException, 
DocumentNotFoundException, 
EmbeddingChunkedFileException, 
InvalidTask1PayloadException, 
InvalidTask2PayloadException, 
ParsingSavedFileException, 
SavingValidatedFileException, 
TokenizationWorkerStarterException
)
from langchain_core.documents import Document as LangChainDocument
from routers.Ai.ai_services import embedding_model
# from db import AsyncSessionLocal #fastapi db lifecycle
from db import CelerySessionLocal #celery db lifecycle! (celery also has a db session saprate form fastapi!)
from db_tables.tables import BM25Resource, CacheVDBResource, Document
from routers.Ai.ai_services import save_validated_doc_task_service, parse_chunk_embed_saved_doc_task2_service
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES, USER_ERROR_CODES
from celery_worker.celery_app import celery_app
from utils.config import settings
from utils.logging.helper_log import log_state
from utils.logging.logEvents import QuestionLogs, ServiceLog, UploadFileLogs
from utils.schemas import  APIResponse, BM25Status, CacheVDBStatus, DocumentStatus, LogState, MultiIndexStatus, SavedDocumentPayload, passed_vlidation_reponce
from pydantic import ValidationError





#for task 1:
async def save_validated_doc_task1_async(payload: passed_vlidation_reponce):
    user_id = payload.file_payload.user_id if hasattr(payload, "file_payload") else None
    request_id = payload.file_payload.request_id if hasattr(payload, "file_payload") else None
    
    log_state(UploadFileLogs.SAVE_TASK1_ASYNC_STARTED, function="save_validated_doc_task1_async", user_id=user_id, request_id=request_id)
    
    async with CelerySessionLocal() as db:       
        result: dict = await save_validated_doc_task_service(
            payload=payload,
            db=db
        )
        
    log_state(UploadFileLogs.SAVE_TASK1_ASYNC_SUCCESS, function="save_validated_doc_task1_async", user_id=user_id, request_id=request_id)
    return result
    


#task 2:
async def parse_chunk_embed_saved_doc_task2_async(doc_meta_obj: SavedDocumentPayload):
    log_state(UploadFileLogs.TASK_2_ASYNC_STARTED, function="parse_chunk_embed_saved_doc_task2_async", user_id=doc_meta_obj.user_id, request_id=doc_meta_obj.request_id)
    
    async with CelerySessionLocal() as db:        
        await parse_chunk_embed_saved_doc_task2_service( #done, no need to return since task2_parse_saved_doc_service will run and our game is played!
            doc_meta_obj=doc_meta_obj,
            db=db
        )
        
    log_state(UploadFileLogs.TASK_2_ASYNC_SUCCESS, function="parse_chunk_embed_saved_doc_task2_async", user_id=doc_meta_obj.user_id, request_id=doc_meta_obj.request_id)



# celery of task2 
@celery_app.task(bind=True, max_retries=3, name="ai.parse_doc_worker") 
def parse_chunk_embed_saved_doc_task2_inishiator(self, doc_meta_dict: dict):
    # Safely extract user_id and request_id from dict for logging metadata
    user_id = doc_meta_dict.get("user_id")
    request_id = doc_meta_dict.get("request_id")
    
    log_state(UploadFileLogs.TASK_2_STARTED, function="parse_chunk_embed_saved_doc_task2_inishiator", user_id=user_id, request_id=request_id)
    
    try: #see this looks similar to how the first ever celery looked ;)
        log_state(UploadFileLogs.VALIDATING_TASK_2_PAYLOAD, function="parse_chunk_embed_saved_doc_task2_inishiator", user_id=user_id, request_id=request_id)
        doc_meta_obj = SavedDocumentPayload.model_validate(doc_meta_dict)
        
        log_state(UploadFileLogs.RUNNING_TASK_2_ASYNC, function="parse_chunk_embed_saved_doc_task2_inishiator", user_id=user_id, request_id=request_id)
        asyncio.run( 
            parse_chunk_embed_saved_doc_task2_async(
                doc_meta_obj
            )
        )
        
        log_state(UploadFileLogs.TASK_2_SUCCESS, function="parse_chunk_embed_saved_doc_task2_inishiator", user_id=user_id, request_id=request_id)
        
        
    except ValidationError as exc:
        log_state(UploadFileLogs.TASK_2_VALIDATION_ERROR, function="parse_chunk_embed_saved_doc_task2_inishiator", user_id=user_id, request_id=request_id)
        raise InvalidTask2PayloadException( 
            error_code=SYSTEM_ERROR_CODES.INVALID_TASK2_PAYLOAD.value,
            message="The file_validation worker failed b4 asyc wrapper"
        ) from exc
        
    except (
        httpx.TimeoutException,
        ConnectionError,
        ParsingSavedFileException,
        ChunkingParsedFileException,
        EmbeddingChunkedFileException,
    ) as exc:
        log_state(UploadFileLogs.TASK_2_RETRYING, function="parse_chunk_embed_saved_doc_task2_inishiator", user_id=user_id, request_id=request_id, exc=exc)
        raise self.retry(exc=exc, countdown=1)





#celery of task 1
@celery_app.task(bind=True, max_retries=3, name="ai.upload_save_file_worker") 
def save_validated_doc_task(self, validated_file_data: dict):
    user_id = validated_file_data.get("file_payload", {}).get("user_id")
    request_id = validated_file_data.get("file_payload", {}).get("request_id")
    
    log_state(UploadFileLogs.SAVE_TASK_STARTED, function="save_validated_doc_task", user_id=user_id, request_id=request_id)
    
    try:  
        log_state(UploadFileLogs.VALIDATING_TASK_PAYLOAD, function="save_validated_doc_task", user_id=user_id, request_id=request_id)
        payload: passed_vlidation_reponce = passed_vlidation_reponce.model_validate(validated_file_data)  # here we go we made it object again!
        
        log_state(UploadFileLogs.SAVING_DOC_TO_DB, function="save_validated_doc_task", user_id=user_id, request_id=request_id)
        doc_meta_dict: dict = asyncio.run( 
            save_validated_doc_task1_async(
                payload
            )
        )
        
        try:
            log_state(UploadFileLogs.INITIATING_TASK_2, function="save_validated_doc_task", user_id=user_id, request_id=request_id)
            result: AsyncResult = parse_chunk_embed_saved_doc_task2_inishiator.delay(doc_meta_dict) 
            log_state(UploadFileLogs.TASK_2_INITIATED_SUCCESS, function="save_validated_doc_task", user_id=user_id, request_id=request_id)
            
        except Exception as e:
            log_state(UploadFileLogs.TASK_2_INITIATION_FAILED, function="save_validated_doc_task", user_id=user_id, request_id=request_id)
            raise TokenizationWorkerStarterException(
                error_code=SYSTEM_ERROR_CODES.TOKENIZATION_EXCEPTION.value,
                message="Task 2 inishiator failed, thus parsing, chunking, embeding didnt start"
            ) from e 
        
        
    except ValidationError as exc:
        log_state(UploadFileLogs.TASK_VALIDATION_ERROR, function="save_validated_doc_task", user_id=user_id, request_id=request_id)
        raise InvalidTask1PayloadException(  # this one here coz what if --eq(1) try  throws a error ! and it'll take the error and go to bellow expct and retry dw!
            error_code=SYSTEM_ERROR_CODES.INVALID_TASK1_PAYLOAD.value,
            message="The worker received an invalid payload before entering the async wrapper, due to not getting proper pydantic obj->dict"
        ) from exc

        
    except (
        httpx.TimeoutException,
        ConnectionError,
        SavingValidatedFileException,  
        TokenizationWorkerStarterException, 
    ) as exc:
        log_state(UploadFileLogs.TASK_RETRYING, function="save_validated_doc_task", user_id=user_id, request_id=request_id, exc=exc)
        raise self.retry(exc=exc, countdown=1)



# 1 (Sync Wrapper) for bellow multi_index_db_creation_async
@celery_app.task(bind=True, max_retries=3, name="ai.create_rest_vdbs")
def multi_index_db_creation_task(self, doc_id: int, create_summary: bool, create_explanation: bool, user_id: int) -> dict[str, Any]:
    """Synchronous Celery entrypoint managing the event loop lifecycle."""
    log_state(UploadFileLogs.MULTI_INDEX_TASK_STARTED, function="multi_index_db_creation_task", user_id=user_id)
    try:
        return asyncio.run(
            multi_index_db_creation_async(
                task_instance=self,  # Pass self for retries/context
                doc_id=doc_id,
                create_summary=create_summary,
                create_explanation=create_explanation,
                user_id=user_id,
            )
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)



# 2. Asynchronous Execution Helper (this creates and feeds at the same time!)
async def multi_index_db_creation_async(task_instance: Any, doc_id: int, create_summary: bool, create_explanation: bool, user_id: int) -> dict[str, Any]:
    # --- SESSION 1: Quick read to grab what we need, then close immediately ---
    log_state(UploadFileLogs.FETCHING_DOCUMENT_FOR_TASK, function="multi_index_db_creation_async", user_id=user_id, request_id=str(doc_id))
    
    async with CelerySessionLocal() as db:
        document = await db.get(Document, doc_id)
        if document is None:
            log_state(UploadFileLogs.UPLOAD_FAILED, function="multi_index_db_creation_async", user_id=user_id, request_id=str(doc_id))
            raise DocumentNotFoundException(
                error_code=SYSTEM_ERROR_CODES.DOCUMENT_NOT_FOUND.value,
                message="Document not found for multi-index VDB creation.",
            )
        # Pull values into local variables so we don't need the db session open anymore
        collection_name = document.collection_name
        summary_status = document.summary_vdb_status
        explanation_status = document.explanation_vdb_status

    user_chroma_dir = Path(settings.chroma_db_dir) / f"user_{user_id}"

    # 1. Fetch primary vector DB raw chunks
    log_state(UploadFileLogs.FETCHING_RAW_CHUNKS, function="multi_index_db_creation_async", user_id=user_id)
    primary_vdb = Chroma(
        collection_name=collection_name,
        persist_directory=str(user_chroma_dir),
        embedding_function=embedding_model,
    ) 

    raw_data = await asyncio.to_thread(
        primary_vdb.get, 
        where={"doc_id": doc_id}, 
    )

    raw_texts: list[str] = raw_data.get("documents", []) #all the chunks's .page_content in each index curtacy of Chroma's get ;)
    raw_metadatas: list[dict] = raw_data.get("metadatas", [])

    if not raw_texts:
        return {
            "success": False,
            "doc_id": doc_id,
            "message": f"No raw chunks found in Chroma for doc_id: {doc_id}",
        }

    final_summary_status = summary_status
    final_explanation_status = explanation_status

    # SUMMARY INDEX (in futue i could try to make this into a worker, if i wanna)
    if create_summary and summary_status != MultiIndexStatus.READY:
        log_state(UploadFileLogs.BUILDING_SUMMARY_VDB, function="multi_index_db_creation_async", user_id=user_id)
        try:
            summary_response: APIResponse = await summry_ai(raw_texts, user_id)
            if summary_response.success:
                summary_batch: SummaryBatchModel = summary_response.data
                
                summary_docs = [
                    LangChainDocument(
                        page_content=summary.text,
                        metadata={
                            **meta, 
                            "doc_type": "summary",
                            "raw_content": raw_text, 
                        },
                    )
                    for raw_text, meta, summary in zip(
                        raw_texts,
                        raw_metadatas,
                        summary_batch.summaries,
                    )
                ] 

                await asyncio.to_thread(
                    Chroma.from_documents,
                    documents=summary_docs,
                    embedding=embedding_model,
                    collection_name=f"{collection_name}_summary",
                    persist_directory=str(user_chroma_dir),
                )

                final_summary_status = MultiIndexStatus.READY
                log_state(UploadFileLogs.SUMMARY_VDB_SUCCESS, function="multi_index_db_creation_async", user_id=user_id)

            else:
                final_summary_status = MultiIndexStatus.FAILED
                log_state(UploadFileLogs.SUMMARY_VDB_FAILED, function="multi_index_db_creation_async", user_id=user_id)

        except Exception as exc:
            print(f"[SUMMARY VDB FAILED] {type(exc).__name__}: {exc}", flush=True)
            final_summary_status = MultiIndexStatus.FAILED
            log_state(UploadFileLogs.SUMMARY_VDB_FAILED, function="multi_index_db_creation_async", user_id=user_id)

    # EXPLANATION INDEX
    if create_explanation and explanation_status != MultiIndexStatus.READY:
        log_state(UploadFileLogs.BUILDING_EXPLANATION_VDB, function="multi_index_db_creation_async", user_id=user_id)
        try:
            explanation_response: APIResponse = await explanation_ai(raw_texts, user_id)

            if explanation_response.success:
                explanation_batch: ExplanationBatchModel = explanation_response.data

                explanation_docs = [
                    LangChainDocument(
                        page_content=item.explanation,
                        metadata={
                            **meta,  
                            "doc_type": "explanation",
                            "raw_content": raw_text,  
                            "topic": item.topic,      
                            "key_takeaway": item.key_takeaway,
                        },
                    )
                    for raw_text, meta, item in zip(
                        raw_texts,
                        raw_metadatas,
                        explanation_batch.explanations,
                    )
                ]

                await asyncio.to_thread(
                    Chroma.from_documents,
                    documents=explanation_docs,
                    embedding=embedding_model,
                    collection_name=f"{collection_name}_explanation",
                    persist_directory=str(user_chroma_dir),
                )

                final_explanation_status = MultiIndexStatus.READY
                log_state(UploadFileLogs.EXPLANATION_VDB_SUCCESS, function="multi_index_db_creation_async", user_id=user_id)
            else:
                final_explanation_status = MultiIndexStatus.FAILED
                log_state(UploadFileLogs.EXPLANATION_VDB_FAILED, function="multi_index_db_creation_async", user_id=user_id)

        except Exception as exc:
            print(f"[explain VDB FAILED] {type(exc).__name__}: {exc}", flush=True)
            final_explanation_status = MultiIndexStatus.FAILED
            log_state(UploadFileLogs.EXPLANATION_VDB_FAILED, function="multi_index_db_creation_async", user_id=user_id)

    # --- SESSION 2: Open a fresh, short-lived session just to commit the statuses ---
    async with CelerySessionLocal() as db:
        document = await db.get(Document, doc_id)
        if document:
            document.summary_vdb_status = final_summary_status
            document.explanation_vdb_status = final_explanation_status
            await db.commit()

    log_state(UploadFileLogs.MULTI_INDEX_TASK_COMPLETED, function="multi_index_db_creation_async", user_id=user_id)

    return {
        "success": True,
        "doc_id": doc_id,
        "summary_status": final_summary_status.value,
        "explanation_status": final_explanation_status.value,
    }





@celery_app.task(bind=True, max_retries=3, name="retri.creating_global_bm25")
def global_bm25_build_task(self, user_id: int) -> dict[str, Any]:
    """Synchronous Celery entrypoint managing the event loop lifecycle for Global BM25 index creation."""
    log_state(UploadFileLogs.MULTI_INDEX_TASK_STARTED, function="global_bm25_build_task", user_id=user_id)
    try:
        return asyncio.run(
            global_bm25_build_async(
                task_instance=self,  # Pass self for retries/context
                user_id=user_id,
            )
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)


# Asynchronous Worker Implementation 
async def global_bm25_build_async(task_instance: Any, user_id: int) -> dict[str, Any]:
    log_state(UploadFileLogs.FETCHING_DOCUMENT_FOR_TASK, function="global_bm25_build_async", user_id=user_id)
    
    async with CelerySessionLocal() as db:
        # 1. Fetch all documents belonging to this user that are fully READY
        stmt = select(Document).where(
            Document.user_id == user_id,
            Document.status == DocumentStatus.READY
        )
        result = await db.execute(stmt)
        documents = result.scalars().all()

        # Fetch the BM25Resource record to get its current version and state safely
        bm25_stmt = select(BM25Resource).where(BM25Resource.user_id == user_id)
        bm25_result = await db.execute(bm25_stmt)
        bm25_res = bm25_result.scalar_one_or_none()

        if bm25_res is None:
            log_state(UploadFileLogs.UPLOAD_FAILED, function="global_bm25_build_async", user_id=user_id)
            return {
                "success": False,
                "user_id": user_id,
                "message": f"BM25Resource record not found for user_id: {user_id}",
            }
        
        # Extract primitive version value to avoid detached ORM object issues later
        current_version = bm25_res.version

    if not documents:
        log_state(UploadFileLogs.WORKER_SKIPPED_DOC_NOT_READY, function="global_bm25_build_async", user_id=user_id)
        async with CelerySessionLocal() as db:
            bm25_stmt = select(BM25Resource).where(BM25Resource.user_id == user_id)
            bm25_result = await db.execute(bm25_stmt)
            bm25_res_inner = bm25_result.scalar_one_or_none()
            if bm25_res_inner:
                bm25_res_inner.status = BM25Status.READY # Or handle as empty
                await db.merge(bm25_res_inner)
                await db.commit()
        return {
            "success": True,
            "user_id": user_id,
            "message": "No ready documents found for user; BM25 resource initialized empty.",
        }

    user_chroma_dir = Path(settings.chroma_db_dir) / f"user_{user_id}"
    master_docs: list[LangChainDocument] = []

    # 2. Iterate through all collections/documents and pull raw chunks from Chroma
    log_state(UploadFileLogs.FETCHING_RAW_CHUNKS, function="global_bm25_build_async", user_id=user_id)
    
    for doc in documents:
        #why did we do this and not get_user_vbd()?? well coz here! each intter collection_name is changing! document is a list of all docs of user and for each 
        #diff doc we get a diff row which is essettnailly in list!
        primary_vdb = Chroma(
            collection_name=doc.collection_name,
            persist_directory=str(user_chroma_dir),
            embedding_function=embedding_model, 
        )

        raw_data = await asyncio.to_thread(
            primary_vdb.get,
            where={"doc_id": doc.doc_id},
        )

        texts = raw_data.get("documents", [])
        metadatas = raw_data.get("metadatas", [])

        for text, meta in zip(texts, metadatas):
            master_docs.append(
                LangChainDocument(
                    page_content=text,
                    metadata={
                        **meta,
                        "file_name": doc.original_filename, # Attach original file name for subset filtering later!
                        "doc_id": doc.doc_id
                    }
                )
            )

    if not master_docs:
        log_state(UploadFileLogs.UPLOAD_FAILED, function="global_bm25_build_async", user_id=user_id)
        async with CelerySessionLocal() as db:
            bm25_stmt = select(BM25Resource).where(BM25Resource.user_id == user_id)
            bm25_result = await db.execute(bm25_stmt)
            bm25_res_inner = bm25_result.scalar_one_or_none()
            if bm25_res_inner:
                bm25_res_inner.status = BM25Status.FAILED
                bm25_res_inner.failure_reason = "No text chunks found across user documents."
                await db.merge(bm25_res_inner)
                await db.commit()
        return {
            "success": False,
            "user_id": user_id,
            "message": "Master chunks list was empty.",
        }

    # 3. Serialize and save master document corpus AND the pre-built BM25 retriever to disk as versioned pickle files
    log_state(UploadFileLogs.BUILDING_GLOBAL_BM25, function="global_bm25_build_async", user_id=user_id) # Re-using log event or create a custom one
    try:
        user_dir = Path(settings.chroma_db_dir) / f"user_{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate next version increment safely ahead of file write
        next_version = current_version + 1
        #IMPP look paths!
        pickle_path = user_dir / f"master_bm25_docs_v{next_version}.pkl"
        retriever_pickle_path = user_dir / f"master_bm25_retriever_v{next_version}.pkl"
        
        # Temporary paths for atomic write semantics (publish only when complete)
        temp_pickle_path = user_dir / f"master_bm25_docs_v{next_version}.pkl.tmp"
        temp_retriever_pickle_path = user_dir / f"master_bm25_retriever_v{next_version}.pkl.tmp"
        
        # Build the global BM25 retriever object in background so API has zero index-building overhead for global searches!
        master_retriever = await asyncio.to_thread(
            BM25Retriever.from_documents,
            documents=master_docs,
            k=20,
        )

        # Write to temp files first, then atomically rename them to prevent partial/corrupted writes
        def save_pickles_safely():
            with open(temp_pickle_path, "wb") as f:
                pickle.dump(master_docs, f)
            with open(temp_retriever_pickle_path, "wb") as f:
                pickle.dump(master_retriever, f)
            
            # Atomic renames
            temp_pickle_path.replace(pickle_path)
            temp_retriever_pickle_path.replace(retriever_pickle_path)

        await asyncio.to_thread(save_pickles_safely)

        # 4. Update BM25Resource status to READY in DB with the new version and path using a fresh session
        async with CelerySessionLocal() as db:
            bm25_stmt = select(BM25Resource).where(BM25Resource.user_id == user_id)
            bm25_result = await db.execute(bm25_stmt)
            bm25_res_inner = bm25_result.scalar_one_or_none()
            if bm25_res_inner:
                bm25_res_inner.status = BM25Status.READY
                bm25_res_inner.index_path = str(pickle_path)
                bm25_res_inner.version = next_version
                bm25_res_inner.failure_reason = None
                await db.merge(bm25_res_inner)
                await db.commit()

        log_state(UploadFileLogs.WORKER_INITIATED_SUCCESS, function="global_bm25_build_async", user_id=user_id)
        return {
            "success": True,
            "user_id": user_id,
            "total_chunks_indexed": len(master_docs),
            "index_path": str(pickle_path),
            "retriever_index_path": str(retriever_pickle_path),
        }

    except Exception as e:
            log_state(UploadFileLogs.UPLOAD_FAILED, level=LogState.EXCEPTION, function="global_bm25_build_async", user_id=user_id)
            
            # Safely persist failure state to DB so the next poll can retry cleanly
            try:
                async with CelerySessionLocal() as db:
                    bm25_stmt = select(BM25Resource).where(BM25Resource.user_id == user_id)
                    bm25_result = await db.execute(bm25_stmt)
                    bm25_res_inner = bm25_result.scalar_one_or_none()
                    if bm25_res_inner:
                        bm25_res_inner.status = BM25Status.FAILED
                        bm25_res_inner.failure_reason = str(e)
                        await db.merge(bm25_res_inner)
                        await db.commit()
            except Exception as db_err:
                # Fallback print/log if even the db flush fails, completely preventing worker crash propagation
                print(f"Critical error saving BM25 failure state for user {user_id}: {db_err}")

            return {
                "success": False,
                "user_id": user_id,
                "message": f"Global BM25 build failed: {str(e)}",
            }



#this only creates doesnt feed at the same time!
@celery_app.task(bind=True, max_retries=3, name="ai.cache_vdb")
def create_cache_vdb_worker(self, user_id: int):
    log_state(QuestionLogs.CELERY_TASK_STARTED, function="create_cache_vdb_worker", user_id=user_id)
    try:
        return asyncio.run(
            create_cache_vdb_async(
                task_instance=self,
                user_id=user_id,
            )
        )
    except Exception as exc:
        log_state(QuestionLogs.CELERY_TASK_RETRYING, function="create_cache_vdb_worker", user_id=user_id)
        # Fallback retry with a 10-second countdown
        raise self.retry(exc=exc, countdown=10)




async def create_cache_vdb_async(task_instance: Any, user_id: int):
    async with CelerySessionLocal() as db:
        try:
            user_cache_dir = (
                Path(settings.chroma_db_dir)
                / f"user_{user_id}"
            )
            
            # Ensure directory exists safely
            log_state(QuestionLogs.CACHE_VDB_DIR_CREATING, function="create_cache_vdb_async", user_id=user_id)
            await asyncio.to_thread(
                user_cache_dir.mkdir,
                parents=True,
                exist_ok=True,
            )

            # Initialize empty Chroma store wrapped cleanly in a lambda
            log_state(QuestionLogs.CACHE_VDB_STORE_INITIALIZING, function="create_cache_vdb_async", user_id=user_id)
            await asyncio.to_thread(
                lambda: Chroma(
                    collection_name=f"question_cache_{user_id}",
                    embedding_function=embedding_model,
                    persist_directory=str(user_cache_dir),
                )
            )

            # Fetch the DB resource record
            stmt = select(CacheVDBResource).where(
                CacheVDBResource.user_id == user_id
            )
            result = await db.execute(stmt)
            cache_res = result.scalar_one_or_none()
            

            if cache_res is None:
                raise RuntimeError(
                    f"Cache VDB resource record not found in DB for user {user_id}"
                ) #dw we have 3 retries this will just be more info

            # Mark as READY
            log_state(QuestionLogs.CACHE_VDB_MARKED_READY, function="create_cache_vdb_async", user_id=user_id)
            cache_res.status = CacheVDBStatus.READY
            cache_res.vdb_path = str(user_cache_dir)
            cache_res.failure_reason = None
            await db.commit()

            return {
                "status": "READY",
                "user_id": user_id,
            }

        except Exception as exc:
            log_state(QuestionLogs.CACHE_VDB_MARKED_FAILED, function="create_cache_vdb_async", user_id=user_id)
            # If anything blows up, update state to FAILED before bubbling up to Celery retry
            stmt = select(CacheVDBResource).where(
                CacheVDBResource.user_id == user_id
            )
            result = await db.execute(stmt)
            cache_res = result.scalar_one_or_none()

            if cache_res:
                cache_res.status = CacheVDBStatus.FAILED
                cache_res.failure_reason = str(exc)
                await db.commit()

            raise 



from sqlalchemy.ext.asyncio import AsyncSession
async def get_batch_document_versions(doc_ids: list[int], db: AsyncSession) -> dict[int, int]:
    if not doc_ids:
        return {}
    stmt = select(Document.doc_id, Document.version).where(Document.doc_id.in_(doc_ids))
    result = await db.execute(stmt)
    return {row.doc_id: row.version for row in result.all()}


@celery_app.task(bind=True, max_retries=3, name="ai.push_cache_vdb")
def push_responce_in_cache_worker(self, user_id: int, question: str, model_output_dict: dict, doc_name: list[str] | None):
    try:
        return asyncio.run(
            push_response_in_cache_async(
                task_instance=self,
                user_id=user_id,
                question=question,
                model_output_dict=model_output_dict,
                doc_name=doc_name
            )
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)


async def push_response_in_cache_async(task_instance: Any, user_id: int, question: str, model_output_dict: dict, doc_name: list[str] | None) -> str | None:
    log_state(QuestionLogs.CACHE_VDB_PUSH_ASYNC_ENTERED, function="push_response_in_cache_async", user_id=user_id)
    async with CelerySessionLocal() as db:
        try:
            if isinstance(doc_name, str):
                doc_name = [doc_name]
            
            if not doc_name:
                doc_name = None
                
                
            # 1. Extract doc_ids associated with the source documents
            if doc_name is not None:
                stmt = select(Document.doc_id).where(
                    Document.user_id == user_id,
                    Document.original_filename.in_(doc_name)
                )
            else: #if doc_name None aka full corpus serch then all doc's latest version get!
                stmt = select(Document.doc_id).where(
                    Document.user_id == user_id
                )
            
            res = await db.execute(stmt)
            doc_ids = [row.doc_id for row in res.all()]

            # 2. Fetch live versions for source_versions tracking dictionary
            versions_map = await get_batch_document_versions(doc_ids, db)
            source_versions_dict = {int(d_id): ver for d_id, ver in versions_map.items()} #unpacks and also for each doc_id saves latest version in value of key value

            # 3. Serialize model output dictionary back to JSON string payload
            response_json_str = json.dumps(model_output_dict)

            cache_metadata = {
                "user_id": user_id,
                "question": question,
                "llm_response": model_output_dict.get('vdb_fetched_answer', ''),
                "full_llm_response": response_json_str,
                "doc_ids": json.dumps(doc_ids),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_versions": json.dumps(source_versions_dict)
            }

            # 4. Construct LangChain Document object for vector store ingestion matching get_cache expectation
            cache_document = LangChainDocument(
                page_content=question,
                metadata=cache_metadata
            )

            
            # 5. Resolve user's cache VDB path from DB resource record
            stmt = select(CacheVDBResource).where(
                CacheVDBResource.user_id == user_id
            )

            result = await db.execute(stmt)
            cache_res = result.scalar_one_or_none()

            if cache_res is None:
                raise RuntimeError(
                    f"Cache VDB resource record not found for user {user_id}"
                )

            if not cache_res.vdb_path:
                raise RuntimeError(
                    f"Cache VDB path not found for user {user_id}"
                )

            user_cache_vdb_path = Path(cache_res.vdb_path)

            # 6. Initialize user's Chroma cache store instance thread-safely
            user_cache_vdb = await asyncio.to_thread(
                lambda: Chroma(
                    collection_name=f"question_cache_{user_id}",
                    embedding_function=embedding_model,
                    persist_directory=str(user_cache_vdb_path),
                )
            )

            # 7. Push using add_documents so it matches LangChainDocument expectations on read
            await asyncio.to_thread(
                user_cache_vdb.add_documents,
                documents=[cache_document]
            )

            return cache_metadata["created_at"]

        except Exception as exc:
            log_state(ServiceLog.AI_SERVICE_FAILED, level=LogState.WARNING, function="push_response_in_cache_async", user_id=user_id, exc=exc)
            raise