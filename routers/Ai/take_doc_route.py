from fastapi import (
    APIRouter,
    Depends,
    Request,
    Response,
    UploadFile,
    File
)
from routers.Ai.ai_route_utils import redis_and_db_worker_status
from core.Exceptions.exceptions import UploadingException
from core.rate_limiters.limiter_file import limiter
from core.rate_limiters.limiter_utils import RateLimits
from Oauth2 import get_user_jwt_payload
from db import get_db
from utils.ai_responce_handler import handle_service_response
from utils.logging.helper_log import log_state
from utils.logging.logEvents import UploadFileLogs
from utils.schemas import All_worker_starter_responce, TokenDataSchema, APIResponse, DataToFrontEndAfterUploadingRoute, passed_vlidation_reponce
from sqlalchemy.ext.asyncio import AsyncSession
from routers.Ai.doc_verification_utils import multi_index_db_creation_worker_and_bm25_maker_inishiator, upload_doc_worker_inishiator
from routers.Ai.ai_services import file_validation_service
from db_tables.tables import BM25Resource, Document
from sqlalchemy import select   
router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("/upload-doc", response_model=All_worker_starter_responce)
@limiter.limit(RateLimits.AI.FILE_UPLOAD) 
async def upload_doc(
    request: Request, response: Response, 
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user_jwt_payload: TokenDataSchema = Depends(get_user_jwt_payload) 
):
    user_id = user_jwt_payload.user_id
    log_state(UploadFileLogs.UPLOAD_SERVICE_STARTED, function="upload_doc", user_id=user_id)
    
    log_state(UploadFileLogs.VALIDATING_FILE, function="upload_doc", user_id=user_id)
    result: APIResponse = await file_validation_service(file=file, user_jwt_payload=user_jwt_payload, db=db)
    
    if not result.success:
        log_state(UploadFileLogs.VALIDATION_FAILED, function="upload_doc", user_id=user_id)
        
        
        
    data: passed_vlidation_reponce = handle_service_response(result, UploadingException)
    log_state(UploadFileLogs.VALIDATION_SUCCESS, function="upload_doc", user_id=user_id)
    
    
    log_state(UploadFileLogs.INITIATING_WORKER, function="upload_doc", user_id=user_id)
    task_id = await upload_doc_worker_inishiator(data=data)
    
    log_state(UploadFileLogs.WORKER_INITIATED_SUCCESS, function="upload_doc", user_id=user_id)
    
    log_state(UploadFileLogs.UPLOAD_SUCCESS, function="upload_doc", user_id=user_id)
    log_state(UploadFileLogs.EXITING_UPLOAD_SERVICE, function="upload_doc", user_id=user_id)
    
    return All_worker_starter_responce(
        task_id=task_id,
        doc_upload_api_responce=DataToFrontEndAfterUploadingRoute(
            request_id=data.file_payload.request_id,
            user_id=user_id
        )
    )





@router.get("/upload_worker/{task_id}/{request_id}")
async def get_upload_worker_result(
    task_id: str,
    request_id: str,
    db: AsyncSession = Depends(get_db),
    user_jwt_payload: TokenDataSchema = Depends(get_user_jwt_payload),
):
    log_state(UploadFileLogs.POLL_WORKER_REQUESTED, function="get_upload_worker_result", user_id=user_jwt_payload.user_id, request_id=request_id)
    result: APIResponse = await redis_and_db_worker_status(task_id=task_id, request_id=request_id, db=db, user_jwt_payload=user_jwt_payload)
    result_data: dict = result.data 

    
    log_state(UploadFileLogs.INITIATING_MULTI_INDEX_CHECK, function="get_upload_worker_result", user_id=user_jwt_payload.user_id, request_id=request_id)
    two_vdb_and_bm25_task_id: APIResponse = await multi_index_db_creation_worker_and_bm25_maker_inishiator(user_id=user_jwt_payload.user_id, request_id=request_id, db=db)
    res: dict | None = handle_service_response(two_vdb_and_bm25_task_id, UploadingException) #if we get None means processing is happening if get data means ggz if nothing we've raised properly ;)
    
    if res is not None:
        bm25_task_id = res["bm25_task_id"] 
        multi_index_task_id = res["multi_index_task_id"]
    

    # Re-fetch document because the initiator may have changed
    # summary/explanation statuses to PROCESSING.
    log_state(UploadFileLogs.REFRESHING_DOCUMENT_STATUS, function="get_upload_worker_result", user_id=user_jwt_payload.user_id, request_id=request_id)

    stmt = select(Document).where(Document.request_id == request_id, Document.user_id == user_jwt_payload.user_id)
    db_result = await db.execute(stmt)
    document = db_result.scalar_one_or_none()

    
    if document:
        result_data["multi_index_worker_2"] = {
            "doc_id": document.doc_id,
            "summary_status": document.summary_vdb_status.value,
            "explanation_status": document.explanation_vdb_status.value,
        }
    else:
        result_data["multi_index_worker_2"] = {
            "summary_status": "STARTED",
            "explanation_status": "STARTED",
        }

    # Fetch and log BM25 resource state during poll
    log_state(UploadFileLogs.CHECKING_BM25_STATUS, function="get_upload_worker_result", user_id=user_jwt_payload.user_id, request_id=request_id)

    bm25_stmt = select(BM25Resource).where(BM25Resource.user_id == user_jwt_payload.user_id)
    db_res = await db.execute(bm25_stmt)
    user_bm25_table_obj = db_res.scalar_one_or_none()
    
    
    if user_bm25_table_obj:
        log_state(UploadFileLogs.BM25_STATUS_FETCHED_SUCCESS, function="get_upload_worker_result", user_id=user_jwt_payload.user_id, request_id=request_id)
        result_data["bm25_worker_3"] = {
            "doc_id": document.doc_id,
            "status": user_bm25_table_obj.status.value,
            "version": user_bm25_table_obj.version,
            "index_path": user_bm25_table_obj.index_path,
            "failure_reason": user_bm25_table_obj.failure_reason,
        }
    else:
        log_state(UploadFileLogs.BM25_STATUS_PENDING_INIT, function="get_upload_worker_result", user_id=user_jwt_payload.user_id, request_id=request_id)
        result_data["bm25_worker_3"] = {
            "status": "STARTED"
        }
        
    log_state(UploadFileLogs.POLL_WORKER_SUCCESS, function="get_upload_worker_result", user_id=user_jwt_payload.user_id, request_id=request_id)
    return result_data


