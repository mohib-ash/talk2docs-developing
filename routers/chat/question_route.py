import asyncio
import json
import hashlib
from sqlite3 import IntegrityError
from typing import Optional
from uuid import uuid4
import uuid
from sqlalchemy import select, text
from langchain_groq import ChatGroq

from Ai.Talk2Doc import Answer_ai, AnswerModel

from Ai.convo_ai.convo_ai import ConvoAnswerModel, continue_convo
from Ai.convo_classifier import classify_conversation_intent
from core.Exceptions.exceptions import NoVectorDatabaseException
from db_tables.tables import AiResponse, ConversationData, Document
from routers.chat.cache_system import multi_tier_cache_system
from routers.chat.question_inishiators import create_cache_vdb_inishiator, create_LTM_vdb_inishiator
from routers.chat.question_services import get_cached, get_redis_vector_cached, save_to_redis_vector_cache
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES, USER_ERROR_CODES
from utils.logging.helper_log import log_state
from utils.logging.logEvents import ProviderLog, QuestionLogs, SecurityLog, ServiceLog
from utils.schemas import ConvoRequest, QuestionRequest
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
from routers.chat.cache_system_utils import get_corpus_source_versions_by_id, generate_cache_key

@router.post("/question", response_model=AnswerModel) 
@limiter.limit(RateLimits.AI.ASK_QUESTION) 
async def ask_question(user_payload: QuestionRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db), user_jwt_payload: TokenDataSchema = Depends(get_user_jwt_payload), normal_redis_instance: Redis = Depends(get_redis), bytes_redis: Redis = Depends(get_redis_binary)) -> APIResponse:
    
    user_id = user_jwt_payload.user_id
    log_state(QuestionLogs.QUESTION_ROUTE_STARTED, function="ask_question", user_id=user_id)
    log_state(QuestionLogs.CHECKING_USER_VDB, function="ask_question", user_id=user_id)
    convo_id = None 
    
    
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
        
    ans2: APIResponse = await create_LTM_vdb_inishiator(user_id=user_id, db=db)
    task_id = (ans.data["task_id"] if ans and ans.data else "LTM_vbd was created, first time task ran.. no more .delay moving forward as such no more task_Id")
    
        
    
    ai_type = "answer_ai"
    doc_names = getattr(user_payload, "doc_name", None) 
    cache_key = generate_cache_key(user_id=user_id, question=user_payload.question, doc_name=doc_names, convo_id=convo_id)
    
    log_state(QuestionLogs.CHECKING_MULTI_TIER_CACHE, function="ask_question", user_id=user_id)
    cache_res: APIResponse = await multi_tier_cache_system(question=getattr(user_payload, "question", None), 
                                                           user_id=user_id, 
                                                           db=db, 
                                                           model=model, 
                                                           normal_redis_instance=normal_redis_instance, 
                                                           doc_names=doc_names, 
                                                           cache_key=cache_key, 
                                                           ai_type=ai_type,
                                                           convo_id=convo_id
                                                           )
    
    cache_classification_verdict: str | None = None
    if cache_res.data:
        cache_classification_verdict = cache_res.data["cache_verdict"]
    
    
    if cache_res.success and cache_res.data:
        log_state(QuestionLogs.CACHE_HIT_SUCCESS, function="ask_question", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="ask_question", user_id=user_id)
        return cache_res.data["data"] 
    
    
    if not cache_res.success:
        log_state(QuestionLogs.CACHE_MISS_FALLTHROUGH, function="ask_question", user_id=user_id)
        

    log_state(QuestionLogs.INVOKING_AI_SERVICE, function="ask_question", user_id=user_id)
    result: APIResponse = await Answer_ai(model=model, 
                                          user_id=user_id, 
                                          user_raw_vdb=user_vdb, 
                                          user_payload=user_payload, 
                                          db=db, 
                                          bytes_redis=bytes_redis, 
                                          cache_policy=cache_classification_verdict,
                                          convo_id=convo_id
                                          )
    
    if result.success:
        log_state(QuestionLogs.QUESTION_SUCCESS, function="ask_question", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="ask_question", user_id=user_id)
        
        if result.data:
            
            # Handle Pydantic model serialization safely for Tier 1  (coz pydantic obj on redis nuh uh )
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
                    # 1. Save to Tier 1 Hot Exact Cache
                    normal_redis_instance.setex(cache_key, 86400, tier1_data),
                    
                    # 2. Save to Tier 2 Redis Vector Cache (this handles that internally)
                    save_to_redis_vector_cache(convo_id=convo_id, user_id=user_id, question=user_payload.question,response_payload=result.data, redis_client=normal_redis_instance, doc_name=user_payload.doc_name, source_versions=source_versions, ai_type=ai_type)
                    )
                
            asyncio.create_task(
               run_both() 
            ) 
            
    else:
        log_state(QuestionLogs.QUESTION_FAILED, function="ask_question", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="ask_question", user_id=user_id)
        
    log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="ask_question", user_id=user_id)
    return handle_service_response(result, AIServiceException)





def create_conversation_id(user_id: int) -> str:
    raw = f"{user_id}:{uuid.uuid4()}"
    return str(hashlib.sha256(raw.encode()).hexdigest())

#front-end should call this b4 calling \convo and pass it in http req (THE IDEA IS IF KEY COMES! NEW CONVO SESSION, IF KEY DON COME SAME QNA SESSION)
@router.get("/convo-id")
async def create_convo_id(user_id: int = Depends(get_user_jwt_payload), db: AsyncSession = Depends(get_db)):
    convo_id: str = create_conversation_id(user_id)
    request_id = str(uuid.uuid4())
    
    return APIResponse(
        success=True,
        data={"convo_id": convo_id},
    )


@router.post("/convo", response_model=ConvoAnswerModel) 
@limiter.limit(RateLimits.AI.CONVO) 
async def start_convo(user_payload: ConvoRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db), user_jwt_payload: TokenDataSchema = Depends(get_user_jwt_payload), normal_redis_instance: Redis = Depends(get_redis), bytes_redis: Redis = Depends(get_redis_binary)) -> APIResponse:
    
    user_id = user_jwt_payload.user_id
    question = user_payload.new_question
    doc_names = getattr(user_payload, "doc_name", None)
    
    
    if not question or not question.strip():
        log_state(SecurityLog.EMPTY_INPUT, function="start_convo", user_id=user_id)
        log_state(ServiceLog.AI_SERVICE_FAILED, function="start_convo", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="start_convo", user_id=user_id)

        return APIResponse(
            success=False,
            data=None,
            error_code=USER_ERROR_CODES.EMPTY_INPUT.value,
            error_message="Question input is empty"
        )
    
    
    user_vdb: Chroma | None = await get_user_vdb(user_id=user_id, db=db)
    if user_vdb is None:
        log_state(QuestionLogs.VDB_NOT_FOUND, function="ask_question", user_id=user_id)
        log_state(QuestionLogs.SERVICE_FAILED, function="ask_question", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="ask_question", user_id=user_id)
        raise NoVectorDatabaseException(
            error_code=SYSTEM_ERROR_CODES.NO_RELATED_VECTOR_DATABSE_FOUND.value,
            message="Bro tryna chat, but aint uploded shit yet man..."
        ) 
        
    
    convo_id: Optional[str] = user_payload.convo_id
    starter_question = user_payload.last_question
    starter_answer = user_payload.last_model_answer
    
    
    if convo_id:
        conversation = ConversationData(
            convo_id=convo_id,
            user_id=user_id,
            starter_question=starter_question,
            starter_answer=starter_answer,
        )
        db.add(conversation)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.DATABASE_ERROR.value,
                error_message="Conversation could not be created."
            )

    
    
    if not convo_id:
        try:
            stmt = (
                select(ConversationData.convo_id)
                .where(ConversationData.user_id == user_id)
                .order_by(text("created_at DESC"))
                .limit(1)
        )
            result = await db.execute(stmt)
            db_response = result.scalar_one_or_none()
            
            convo_id = db_response
        except Exception:
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.DATABASE_ERROR.value,
                error_message="Conversation could not be created."
            )
        
    
    
    #At this point convo_id is gurantee to exits (FUTURE: TODO: make a way to have multiple convos ongoing like gpt! on side!)
    
    
    log_state(QuestionLogs.QUESTION_ROUTE_STARTED, function="start_convo", user_id=user_id)
    log_state(QuestionLogs.CHECKING_USER_VDB, function="start_convo", user_id=user_id)
    
    user_vdb: Chroma | None = await get_user_vdb(user_id=user_id, db=db) 
    if user_vdb is None:
        log_state(QuestionLogs.VDB_NOT_FOUND, function="start_convo", user_id=user_id)
        log_state(QuestionLogs.SERVICE_FAILED, function="start_convo", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="start_convo", user_id=user_id)
        raise NoVectorDatabaseException(
            error_code=SYSTEM_ERROR_CODES.NO_RELATED_VECTOR_DATABSE_FOUND.value,
            message="Bro tryna chat, but aint uploded shit yet man..."
        ) 
        
        
    #Why this here? well if user clicks this straight up and if thhese doesnt exist make em, if they exist no harm
    ans: APIResponse = await create_cache_vdb_inishiator(user_id=user_id, db=db) 
    ans2: APIResponse = await create_LTM_vdb_inishiator(user_id=user_id, db=db)
    
        
    
    ai_type = "continue_convo"    
    cache_key = generate_cache_key(user_id=user_id, question=user_payload.new_question, doc_name=doc_names, convo_id=convo_id) 
    
    log_state(QuestionLogs.CHECKING_MULTI_TIER_CACHE, function="start_convo", user_id=user_id)
    cache_res: APIResponse = await multi_tier_cache_system(
        question=question,
        user_id=user_id,
        db=db,
        model=model,
        normal_redis_instance=normal_redis_instance,
        doc_names=None,
        cache_key=cache_key,
        ai_type=ai_type,
        convo_id=convo_id,
    )
    
    cache_classification_verdict: str | None = None
    if cache_res.data:
        cache_classification_verdict = cache_res.data["cache_verdict"]
    
    
    if cache_res.success and cache_res.data:
        log_state(QuestionLogs.CACHE_HIT_SUCCESS, function="start_convo", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="start_convo", user_id=user_id)
        return cache_res.data["data"] 
    
    
    if not cache_res.success:
        log_state(QuestionLogs.CACHE_MISS_FALLTHROUGH, function="start_convo", user_id=user_id)
        
    
    # FALLBACK: FULL AI GENERATION PIPELINE (Cache Miss / Non-cacheable)
    log_state(QuestionLogs.INVOKING_AI_SERVICE, function="start_convo", user_id=user_id)
    extra = {
        "convo_id": convo_id,
        "starter_question": starter_question,
        "starter_answer": starter_answer,
        "doc_names": doc_names
    }
    result: APIResponse = await continue_convo(question=getattr(user_payload, "new_question", None), 
                                               db=db, 
                                               user_id=user_id, 
                                               cache_policy=cache_classification_verdict, 
                                               extra=extra,
                                               model_json=model,
                                               user_raw_vdb=user_vdb,
                                               bytes_redis=bytes_redis
                                               )
    
    if result.success:
        log_state(QuestionLogs.QUESTION_SUCCESS, function="start_convo", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="start_convo", user_id=user_id)
        
        if result.data:
            
            # Handle Pydantic model serialization safely for Tier 1  (coz pydantic obj on redis nuh uh )
            if hasattr(result.data, "model_dump_json"):
                tier1_data = result.data.model_dump_json()
            elif hasattr(result.data, "dict"):
                tier1_data = json.dumps(result.data.dict())
            else:
                tier1_data = json.dumps(result.data)
                
            
            
            log_state(QuestionLogs.BACKGROUND_CACHE_POPULATION_FIRED, function="start_convo", user_id=user_id)
            # Fire both Tier 1 (Exact) and Tier 2 (Vector) concurrently in the background!
            async def run_both():
                await asyncio.gather(
                    # 1. Save to Tier 1 Hot Exact Cache
                    normal_redis_instance.setex(cache_key, 86400, tier1_data),
                    
                    # 2. Save to Tier 2 Redis Vector Cache (this handles that internally)
                    save_to_redis_vector_cache(user_id=user_id, 
                                               question=user_payload.new_question, 
                                               response_payload=result.data, 
                                               redis_client=normal_redis_instance, 
                                               doc_name=None, 
                                               source_versions=None, 
                                               ai_type=ai_type,
                                               convo_id=convo_id
                                               )
                    )
            asyncio.create_task(
               run_both() 
            ) 
            
    else:
        log_state(QuestionLogs.QUESTION_FAILED, function="start_convo", user_id=user_id)
        log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="start_convo", user_id=user_id)
        
    log_state(QuestionLogs.EXITING_QUESTION_ROUTE, function="start_convo", user_id=user_id)
    return handle_service_response(result, AIServiceException)
