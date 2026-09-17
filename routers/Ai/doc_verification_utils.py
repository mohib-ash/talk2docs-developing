from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from celery_worker.Tasks.Ai_worker.ai_worker import save_validated_doc_task, multi_index_db_creation_task, global_bm25_build_task
from fastapi import UploadFile
from db_tables.tables import BM25Resource, Document
from routers.Ai.ai_services import UPLOAD_DIR, ALLOWED_MIME_TYPES
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES
from utils.logging.helper_log import log_state
from utils.logging.logEvents import UploadFileLogs
from utils.schemas import APIResponse, DocumentStatus, MultiIndexStatus, TokenDataSchema, passed_vlidation_reponce
import secrets
from utils.schemas import BM25Status

from sqlalchemy.ext.asyncio import AsyncSession

async def upload_doc_worker_inishiator(data: passed_vlidation_reponce):
    task = save_validated_doc_task.delay(validated_file_data=data.model_dump())  
    return task.id 







async def multi_index_db_creation_worker_and_bm25_maker_inishiator(user_id: int, request_id: str, db: AsyncSession) -> APIResponse:
    log_state(UploadFileLogs.INITIATING_WORKER, function="multi_index_db_creation_worker_and_bm25_maker_inishiator", user_id=user_id, request_id=request_id)
    stmt = select(Document).where(
        Document.request_id == request_id,
        Document.user_id == user_id,
    )

    result = await db.execute(stmt)
    document = result.scalar_one_or_none()

    # Safety check
    if document is None:
        log_state(UploadFileLogs.UPLOAD_FAILED, function="multi_index_db_creation_worker_and_bm25_maker_inishiator", user_id=user_id, request_id=request_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.DOCUMENT_NOT_FOUND.value,
            error_message="Document not found for multi-index creation.",
        )


    if document.status != DocumentStatus.READY:
        log_state(UploadFileLogs.WORKER_SKIPPED_DOC_NOT_READY, function="multi_index_db_creation_worker_and_bm25_maker_inishiator", user_id=user_id, request_id=request_id)
        return APIResponse(
            success=True, 
            data=None,
            error_code=None,
            error_message=None,
        )  
        
        
    # BM25 RESOURCE CHECK 
    log_state(UploadFileLogs.CHECKING_BM25_STATUS, function="multi_index_db_creation_worker_and_bm25_maker_inishiator", user_id=user_id, request_id=request_id)
    
    bm25_stmt = select(BM25Resource).where(BM25Resource.user_id == user_id)
    bm25_result = await db.execute(bm25_stmt)
    bm25_res = bm25_result.scalar_one_or_none()

    if bm25_res is None:
        log_state(UploadFileLogs.BM25_STATUS_PENDING_INIT, function="multi_index_db_creation_worker_and_bm25_maker_inishiator", user_id=user_id, request_id=request_id)
        bm25_res = BM25Resource(
            user_id=user_id,
            status=BM25Status.PENDING,
            version=0
        )
        db.add(bm25_res)
        await db.commit()
        await db.refresh(bm25_res)
    else:
        log_state(UploadFileLogs.BM25_STATUS_FETCHED_SUCCESS, function="multi_index_db_creation_worker_and_bm25_maker_inishiator", user_id=user_id, request_id=request_id)

    # 1. All targets (VDBs & BM25) already exist 
    if (
        document.summary_vdb_status == MultiIndexStatus.READY
        and document.explanation_vdb_status == MultiIndexStatus.READY
        and bm25_res.status == BM25Status.READY
    ):
        log_state(UploadFileLogs.WORKER_SKIPPED_BOTH_READY, function="multi_index_db_creation_worker_and_bm25_maker_inishiator", user_id=user_id, request_id=request_id)
        return APIResponse(
            success=True,
            data=None,
            error_code=None,
            error_message=None,
        )  


    # 2. Decide which resources actually need to be created
    create_summary = document.summary_vdb_status in (
        MultiIndexStatus.PENDING,
        MultiIndexStatus.FAILED,
    )

    create_explanation = document.explanation_vdb_status in (
        MultiIndexStatus.PENDING,
        MultiIndexStatus.FAILED,
    )


    create_bm25 = bm25_res.status in (
        BM25Status.PENDING,
        BM25Status.FAILED,
        BM25Status.STALE,
    )

    if not create_summary and not create_explanation and not create_bm25:
        log_state(UploadFileLogs.WORKER_SKIPPED_NONE_NEEDED, function="multi_index_db_creation_worker_and_bm25_maker_inishiator", user_id=user_id, request_id=request_id)
        return APIResponse(
            success=True,
            data=None,
            error_code=None,
            error_message=None,
        )

    # 4. Mark only the resources we're actually about to build as PROCESSING
    if create_summary:
        document.summary_vdb_status = MultiIndexStatus.PROCESSING

    if create_explanation:
        document.explanation_vdb_status = MultiIndexStatus.PROCESSING

    if create_bm25:
        bm25_res.status = BM25Status.PROCESSING

    await db.commit()

    # 5. Start background worker(s)
    log_state(UploadFileLogs.QUEUING_MULTI_INDEX_TASK_INISHIATOR, function="multi_index_db_creation_worker_and_bm25_maker_inishiator", user_id=user_id, request_id=request_id)
    
    # Fire multi-index task if summary/explanation needed
    multi_index_task_id = None
    if create_summary or create_explanation:
        task = multi_index_db_creation_task.delay(
            document.doc_id,
            create_summary,
            create_explanation,
            user_id
        )
        multi_index_task_id = task.id


    # Fire global BM25 build task if BM25 needed
    bm25_task_id = None
    if create_bm25:
        log_state(UploadFileLogs.QUEUING_BM25_TASK_INITIATOR, function="multi_index_db_creation_worker_and_bm25_maker_inishiator", user_id=user_id, request_id=request_id)
        bm25_task = global_bm25_build_task.delay(user_id)
        bm25_task_id = bm25_task.id

    log_state(UploadFileLogs.WORKER_INITIATED_SUCCESS, function="multi_index_db_creation_worker_and_bm25_maker_inishiator", user_id=user_id, request_id=request_id)


    return APIResponse(
        success=True,
        data={
            "doc_id": document.doc_id,
            "create_summary": create_summary,
            "create_explanation": create_explanation,
            "create_bm25": create_bm25,
            "multi_index_task_id": multi_index_task_id,
            "bm25_task_id": bm25_task_id,
        },
        error_code=None,
        error_message=None,
    )