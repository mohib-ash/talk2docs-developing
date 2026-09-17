from celery.result import AsyncResult
from celery.states import SUCCESS, FAILURE, RETRY
from db_tables.tables import Document 
from core.Exceptions.exceptions import DocumentNotFoundException
from sqlalchemy import select
from celery_worker.celery_app import celery_app
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES
from utils.logging.helper_log import log_state
from utils.logging.logEvents import UploadFileLogs
from utils.schemas import APIResponse, TokenDataSchema
from sqlalchemy.ext.asyncio import AsyncSession



def get_worker_redis_status(task_id: str, user_id: int) -> APIResponse:
    log_state(UploadFileLogs.CHECKING_REDIS_WORKER_STATE, function="get_worker_redis_status", user_id=user_id)
    
    async_result = AsyncResult(task_id, app=celery_app)
    state = async_result.state

    if state == SUCCESS:
        log_state(UploadFileLogs.REDIS_WORKER_SUCCESS, function="get_worker_redis_status", user_id=user_id)
        return APIResponse(
            success=True,
            data={
                "status": "completed",
                "task_id": task_id,
                "state": state,
                "failed": False,
                "detail": "The task for validation of doc and initial vdb created (yes both in one, i was nested)"
            },
            error_code=None,
            error_message=None,
        )

    elif state == FAILURE:
        log_state(UploadFileLogs.REDIS_WORKER_FAILED, function="get_worker_redis_status", user_id=user_id)
        return APIResponse(
            success=False,
            data={
                "status": "failed",
                "task_id": task_id,
                "state": state,
                "failed": True,
                "detail": "The task for validation of doc and initial vdb created (yes both in one, i was nested)"
            },
            error_code=SYSTEM_ERROR_CODES.TASK_FAILED.value,
            error_message=str(async_result.result),
        )

    elif state == RETRY:
        log_state(UploadFileLogs.REDIS_WORKER_RETRYING, function="get_worker_redis_status", user_id=user_id)
        return APIResponse(
            success=True,
            data={
                "status": "retrying",
                "task_id": task_id,
                "state": state,
                "failed": False,
                "detail": "The task for validation of doc and initial vdb created (yes both in one, i was nested)"
            },
            error_code=None,
            error_message=None,
        )

    else:
        log_state(UploadFileLogs.REDIS_WORKER_PROCESSING, function="get_worker_redis_status", user_id=user_id)
        return APIResponse(
            success=True,
            data={
                "status": "processing",
                "task_id": task_id,
                "state": state,
                "failed": False,
                "detail": "The task for validation of doc and initial vdb created (yes both in one, i was nested)"
            },
            error_code=None,
            error_message=None,
        )




async def get_document_by_request_id(
    request_id: str,
    db: AsyncSession,
    user_jwt_payload: TokenDataSchema,
) -> dict:
    user_id = user_jwt_payload.user_id

    log_state(
        UploadFileLogs.FETCHING_DOCUMENT_BY_REQUEST_ID,
        function="get_document_by_request_id",
        user_id=user_id,
        request_id=request_id,
    )

    stmt = select(Document).where(
        Document.request_id == request_id,
        Document.user_id == user_id,
    )

    result = await db.execute(stmt)
    document = result.scalar_one_or_none()

    # Task 1 may have been queued but not saved yet.
    if document is None:
        log_state(
            UploadFileLogs.DOCUMENT_PENDING_SAVE_STATE,
            function="get_document_by_request_id",
            user_id=user_id,
            request_id=request_id,
        )

        return {
            "doc_id": None,
            "status": "PENDING_SAVE",
            "failure_reason": None,
        }

    log_state(
        UploadFileLogs.DOCUMENT_FOUND_STATE,
        function="get_document_by_request_id",
        user_id=user_id,
        request_id=request_id,
    )

    return {
        "doc_id": document.doc_id,
        "status": document.status,
        "failure_reason": document.failure_reason,
    }
    
def worker_result_handler(result: APIResponse):
    return result.data