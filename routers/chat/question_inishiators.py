from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from celery_worker.Tasks.Ai_worker.ai_worker import create_cache_vdb_worker, create_ltm_vdb_worker
from fastapi import UploadFile
from db_tables.tables import BM25Resource, CacheVDBResource, Document, LTMVDBResource
from routers.Ai.ai_services import UPLOAD_DIR, ALLOWED_MIME_TYPES
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES
from utils.logging.helper_log import log_state
from utils.logging.logEvents import QuestionLogs, UploadFileLogs
from utils.schemas import APIResponse, CacheVDBStatus, DocumentStatus, LTMVDBStatus, MultiIndexStatus, TokenDataSchema, passed_vlidation_reponce
import secrets
from utils.schemas import BM25Status

from sqlalchemy.ext.asyncio import AsyncSession




async def create_cache_vdb_inishiator(user_id: int, db: AsyncSession) -> APIResponse:
    log_state(QuestionLogs.CACHE_VDB_CHECKING_RESOURCE, function="create_cache_vdb_inishiator", user_id=user_id)
    stmt = select(CacheVDBResource).where(
        CacheVDBResource.user_id == user_id,
    )
    
    result = await db.execute(stmt)
    cache_resource = result.scalar_one_or_none()
    
    # 1. If it doesn't exist at all, create it as PENDING
    if cache_resource is None:
        log_state(QuestionLogs.CACHE_VDB_CREATING_PENDING, function="create_cache_vdb_inishiator", user_id=user_id)
        cache_resource = CacheVDBResource(
            user_id=user_id,
            status=CacheVDBStatus.PENDING,
            version=0,
        )
        db.add(cache_resource)
        await db.commit()
        await db.refresh(cache_resource)
    
    # 2. Check if we actually need to spawn a worker
    proceed = cache_resource.status in (
        CacheVDBStatus.PENDING,
        CacheVDBStatus.FAILED,
    )
    
    if not proceed:
        log_state(QuestionLogs.CACHE_VDB_STATUS_SKIPPED, function="create_cache_vdb_inishiator", user_id=user_id)
        return APIResponse(
            success=True,
            data=None,
            error_code=None,
            error_message=None,
        )
    
    # 3. Lock status to PROCESSING so concurrent requests don't duplicate work
    log_state(QuestionLogs.CACHE_VDB_LOCKED_PROCESSING, function="create_cache_vdb_inishiator", user_id=user_id)
    cache_resource.status = CacheVDBStatus.PROCESSING
    await db.commit()
    await db.refresh(cache_resource)

    # 4. Fire the background worker
    log_state(QuestionLogs.CACHE_VDB_WORKER_FIRED, function="create_cache_vdb_inishiator", user_id=user_id)
    task_id = create_cache_vdb_worker.delay(user_id=user_id)
    
    return APIResponse(
        success=True,
        data={
            "task_id": task_id.id if hasattr(task_id, "id") else str(task_id),
            "status": "PROCESSING"
        },
        error_code=None,
        error_message=None,
    )
    

async def create_LTM_vdb_inishiator(user_id: int, db: AsyncSession) -> APIResponse:
    log_state(QuestionLogs.LTM_VDB_CHECKING_RESOURCE, function="create_LTM_vdb_inishiator", user_id=user_id)
    stmt = select(LTMVDBResource).where(
        LTMVDBResource.user_id == user_id,
    )
    
    result = await db.execute(stmt)
    ltm_resource = result.scalar_one_or_none()
    
    # 1. If it doesn't exist at all, create it as PENDING
    if ltm_resource is None:
        log_state(QuestionLogs.LTM_VDB_CREATING_PENDING, function="create_LTM_vdb_inishiator", user_id=user_id)
        ltm_resource = LTMVDBResource(
            user_id=user_id,
            status=LTMVDBStatus.PENDING,
            version=0,
        )
        db.add(ltm_resource)
        await db.commit()
        await db.refresh(ltm_resource)
    
    # 2. Check if we actually need to spawn a worker
    proceed = ltm_resource.status in (
        LTMVDBStatus.PENDING,
        LTMVDBStatus.FAILED,
    )
    
    if not proceed:
        # Already READY or currently PROCESSING -> skip
        log_state(QuestionLogs.LTM_VDB_STATUS_SKIPPED, function="create_LTM_vdb_inishiator", user_id=user_id)
        return APIResponse(
            success=True,
            data=None,
            error_code=None,
            error_message=None,
        )
    
    # 3. Lock status to PROCESSING so concurrent requests don't duplicate work
    log_state(QuestionLogs.LTM_VDB_LOCKED_PROCESSING, function="create_LTM_vdb_inishiator", user_id=user_id)
    ltm_resource.status = LTMVDBStatus.PROCESSING
    await db.commit()
    await db.refresh(ltm_resource)

    # 4. Fire the background worker
    log_state(QuestionLogs.LTM_VDB_WORKER_FIRED, function="create_LTM_vdb_inishiator", user_id=user_id)
    task_id = create_ltm_vdb_worker.delay(user_id=user_id)
    
    return APIResponse(
        success=True,
        data={
            "task_id": task_id.id if hasattr(task_id, "id") else str(task_id),
            "status": "PROCESSING"
        },
        error_code=None,
        error_message=None,
    )