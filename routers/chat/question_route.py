import asyncio
import json
import hashlib
from sqlalchemy import select
from langchain_groq import ChatGroq

from Ai.Talk2Doc import Answer_ai, AnswerModel
from Ai.cache_classifier import get_classification
from Ai.convo_classifier import classify_conversation_intent
from core.Exceptions.exceptions import NoVectorDatabaseException
from db_tables.tables import Document
from routers.chat.cache_system import multi_tier_cache_system
from routers.chat.question_inishiators import create_cache_vdb_inishiator
from routers.chat.question_services import get_cached, get_redis_vector_cached, save_to_redis_vector_cache
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES
from utils.logging.helper_log import log_state
from utils.logging.logEvents import ProviderLog, QuestionLogs
from utils.schemas import QuestionRequest
from fastapi import (
    APIRouter,
    Depends,
    Request,
    Response
)
from core.rate_limiters.limiter_file import limiter
from core.rate_limiters.limiter_utils import RateLimits
from Oauth2 import get_user_jwt_payload
from db import get_db
from utils.ai_responce_handler import handle_service_response
from utils.schemas import All_worker_starter_responce, TokenDataSchema, APIResponse, DataToFrontEndAfterUploadingRoute, passed_vlidation_reponce
from sqlalchemy.ext.asyncio import AsyncSession
from vector_db.chroma import get_user_vdb, get_user_cache_vdb
from langchain_chroma import Chroma
from core.Exceptions.exceptions import AIServiceException
from utils.config import settings 
from redis.asyncio import Redis
from core.redis import get_redis, get_redis_binary
from routers.chat.covo_classidier_service import check_new_question_and_brach
from routers.chat.cache_system_utils import get_corpus_source_versions_by_id, generate_cache_key

router = APIRouter(tags=["Question"])

model = ChatGroq(
    api_key=settings.api_key,   
    model=settings.model,
    temperature=0.1,             
    max_tokens=4096,            
    max_retries=2,               
    reasoning_effort="low",
    model_kwargs={"response_format": {"type": "json_object"}} 
)









@router.post("/question", response_model=AnswerModel) 
@limiter.limit(RateLimits.AI.ASK_QUESTION) 
async def ask_question(user_payload: QuestionRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db), user_jwt_payload: TokenDataSchema = Depends(get_user_jwt_payload), normal_redis_instance: Redis = Depends(get_redis), bytes_redis: Redis = Depends(get_redis_binary)) -> All_worker_starter_responce:
    
    user_id = user_jwt_payload.user_id
    log_state(QuestionLogs.QUESTION_ROUTE_STARTED, function="ask_question", user_id=user_id)
    log_state(QuestionLogs.CHECKING_USER_VDB, function="ask_question", user_id=user_id)
    
    user_vdb: Chroma | None = await get_user_vdb(user_id=user_id, db=db)
    
    if user_vdb is None:
        log_state(QuestionLogs.VDB_NOT_FOUND, function="ask_question", user_id=user_id)
        log_state(QuestionLogs.SERVICE_FAILED, function="ask_question", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="ask_question", user_id=user_id)
        raise NoVectorDatabaseException(
            error_code=SYSTEM_ERROR_CODES.NO_RELATED_VECTOR_DATABSE_FOUND.value,
            message="Bro tryna chat, but aint uploded shit yet man..."
        ) 
        
        
    
    ans: APIResponse = await create_cache_vdb_inishiator(user_id=user_id, db=db) 
    task_id = (ans.data["task_id"] if ans and ans.data else "Cache_vbd was created, first time task ran.. no more .delay moving forward as such no more task_Id")
    
    
    #convo route (comming soon)
    """
    #i purposefully placed the convo_classifer here coz, it'll give time to the db to be created! thus lower caller wont None lower chances of that now!
    #(IN FUTURE IT'D BE BETTER TO LET BY BUTTON USER DECIDE IF HE WANNA ASK NEXT Q OR SATY IN CONVO CHAT OR EXIT THAT TO GO TO NEXT QUSTION BY BUTTON!)
    log_state(QuestionLogs.CHECKING_CONVO_INTENT, function="ask_question", user_id=user_id)
    res: APIResponse = await check_new_question_and_brach(question=user_payload.question, user_id=user_id) #internal logs for new skipped coz this service aint made yet!
    if res.success and res.data == "continue_chat":
        log_state(QuestionLogs.CONVO_INTENT_CONTINUE, function="ask_question", user_id=user_id)
        return handle_service_response(res, AIServiceException)
    else:
        pass #pass coz we didnt go to continue chat ai branch!
    """ 
        
    
    doc_names = getattr(user_payload, "doc_name", None) 
    cache_key = generate_cache_key(user_id=user_id, question=user_payload.question, doc_name=doc_names)
    
    log_state(QuestionLogs.CHECKING_MULTI_TIER_CACHE, function="ask_question", user_id=user_id)
    cache_res: APIResponse = await multi_tier_cache_system(question=getattr(user_payload, "question", None), user_id=user_id, db=db, model=model, normal_redis_instance=normal_redis_instance, doc_names=doc_names, cache_key=cache_key)
    
    cache_classification_verdict: str | None = None
    
    if cache_res.data:
        cache_classification_verdict = cache_res.data["verdict"]
    
    if cache_res.success and cache_res.data:
        log_state(QuestionLogs.CACHE_HIT_SUCCESS, function="ask_question", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="ask_question", user_id=user_id)
        cache_classification_verdict = cache_res.data["verdict"]
        return cache_res.data["data"]
    
    
    if not cache_res.success:
        log_state(QuestionLogs.CACHE_MISS_FALLTHROUGH, function="ask_question", user_id=user_id)
    

    log_state(QuestionLogs.INVOKING_AI_SERVICE, function="ask_question", user_id=user_id)
    cache_policy = cache_classification_verdict
    result: APIResponse = await Answer_ai(model=model, user_id=user_id, user_raw_vdb=user_vdb, user_payload=user_payload, db=db, normal_redis_instance=normal_redis_instance, bytes_redis=bytes_redis, cache_policy=cache_policy)
    
    if result.success:
        log_state(QuestionLogs.QUESTION_SUCCESS, function="ask_question", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="ask_question", user_id=user_id)
        
        if result.data:
            if hasattr(result.data, "model_dump_json"):
                tier1_data = result.data.model_dump_json()
            elif hasattr(result.data, "dict"):
                tier1_data = json.dumps(result.data.dict())
            else:
                tier1_data = json.dumps(result.data)
                
            source_versions = await get_corpus_source_versions_by_id(user_id=user_id, db_session=db, doc_name=user_payload.doc_name)            
            log_state(QuestionLogs.BACKGROUND_CACHE_POPULATION_FIRED, function="ask_question", user_id=user_id)

            
            async def run_both():
                await asyncio.gather(
                    normal_redis_instance.setex(cache_key, 86400, tier1_data),
                    save_to_redis_vector_cache(user_id=user_id, question=user_payload.question,response_payload=result.data, redis_client=normal_redis_instance, doc_name=user_payload.doc_name, source_versions=source_versions)
                    )
            asyncio.create_task(
               run_both() 
            )
            
    else:
        log_state(QuestionLogs.QUESTION_FAILED, function="ask_question", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="ask_question", user_id=user_id)
        
    log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="ask_question", user_id=user_id)
    return handle_service_response(result, AIServiceException)





