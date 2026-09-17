from celery_worker.Tasks.worker_utils import get_worker_redis_status, get_document_by_request_id, worker_result_handler
from utils.logging.helper_log import log_state
from utils.logging.logEvents import UploadFileLogs
from utils.schemas import TokenDataSchema, APIResponse
from sqlalchemy.ext.asyncio import AsyncSession

async def redis_and_db_worker_status(task_id: str, request_id: str, db: AsyncSession, user_jwt_payload: TokenDataSchema) -> APIResponse:
    user_id = user_jwt_payload.user_id
    log_state(
        UploadFileLogs.REDIS_AND_DB_STATUS_CHECK_STARTED,
        function="redis_and_db_worker_status",
        user_id=user_id,
        request_id=request_id
    )

    
    worker_response: APIResponse = get_worker_redis_status(task_id=task_id, user_id=user_id) 
    document: dict  = await get_document_by_request_id(request_id=request_id, db=db, user_jwt_payload=user_jwt_payload)

    data =  {
        "worker_1": worker_result_handler(worker_response),
        "doc_in_worker_1": {
            "doc_id": document["doc_id"],
            "status": document["status"],
            "failure_reason": document["failure_reason"],
        },
    } 

    log_state(
        UploadFileLogs.REDIS_AND_DB_STATUS_CHECK_SUCCESS,
        function="redis_and_db_worker_status",
        user_id=user_jwt_payload.user_id,
        request_id=request_id
    )

    return APIResponse(
        success=True,
        data=data,
        error_code=None,
        error_message=None
    )