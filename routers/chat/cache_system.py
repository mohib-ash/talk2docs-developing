import asyncio
import json
from typing import Any, List, Optional
from langchain_chroma import Chroma
from redis.asyncio import Redis
from db_tables.tables import AiResponse
from routers.chat.question_services import get_cached, get_redis_vector_cached, save_to_redis_vector_cache
from Ai.cache_classifier import get_cache_classification
from routers.chat.question_services import get_redis_vector_cached
from utils.logging.helper_log import log_state
from utils.logging.logEvents import QuestionLogs
from utils.schemas import APIResponse, QuestionRequest
from sqlalchemy.ext.asyncio import AsyncSession
from routers.chat.cache_system_utils import get_corpus_source_versions_by_id
from vector_db.chroma import get_user_cache_vdb



async def multi_tier_cache_system(convo_id: str, ai_type: str, question: str, user_id: int, normal_redis_instance: Redis, db: AsyncSession, model: Any, cache_key: str, doc_names: list[str] | None = None) -> APIResponse:
    # TIER 1: SLAP-BACK (Redis Key-Value) -> Stops double-enter spam instantly
    cached_payload = await normal_redis_instance.get(cache_key)
    if cached_payload:
        log_state(QuestionLogs.CACHE_TIER1_EXACT_HIT, function="multi_tier_cache_system", user_id=user_id)
        data = {
            "cache_verdict": None,
            "data": json.loads(cached_payload)
        }
        return APIResponse(
            data=data, # Instantly returns the AnswerModel dict! (yes we de-seralize tear-1 dw!) str->obj
            success=True,
            error_code=None,
            error_message=None
        )
    
    
    
    # i dont want T2/T3 cache for covo_ai so ill early exit coz wt if user ask (what do you think about it? very high occurace chance yet not asked on same topic...):
    # i feel like ill drop cache T2,T3 for it tho ill keep the groundwork and still store reponces in cache_vdb just coz if i need in future convo_ai msgs!
    # BTW FOR CONVO_AI I TESTED ALL 3 TIERS THE WORK!
    if convo_id:
        print("Exiting early coz only T1 in covo_ai")
        return APIResponse(
            success=False,
            data=data,
            error_code=None,
            error_message=None
        )
    
    
    cache_policy = "non_cacheable"
    log_state(QuestionLogs.CACHE_CLASSIFICATION_AI, function="multi_tier_cache_system", user_id=user_id)
    
    cache_classifier = await get_cache_classification(question=question, user_id=user_id)
    if cache_classifier.success and cache_classifier.data:
        cache_policy = cache_classifier.data.responce.strip().lower()  #its not scoped! coz it was assigned outside first!
    
    
    
    #early exist! if still its non_cachable for both cases
    if cache_policy == "non_cacheable":
        log_state(QuestionLogs.CACHE_POLICY_NON_CACHEABLE, function="multi_tier_cache_system", user_id=user_id)
        data = {
            "cache_verdict": cache_policy,
            "data": None
        }
        return APIResponse(
            success=False,
            data=data,
            error_code=None,
            error_message=None
        )
    
    

    
    
    # TIER 2: HOT-CACHE in memory redis similarity serch
    #Blazing fast RAM-based vector similarity search (skips disk I/O)
    if cache_policy == "cacheable":
        source_versions = await get_corpus_source_versions_by_id(user_id=user_id, db_session=db, doc_name=doc_names) #in future come down and only run this if ai_type="answer_ai" (hold till i make csv/xlsx covo)
        
        log_state(QuestionLogs.CACHE_POLICY_CACHEABLE, function="multi_tier_cache_system", user_id=user_id)
        redis_vector_result: APIResponse = await get_redis_vector_cached(
            question=question, 
            user_id=user_id, 
            redis_client=normal_redis_instance, 
            doc_name=doc_names, 
            source_versions=source_versions, 
            ai_type=ai_type, 
            convo_id=convo_id
        )
        
        if redis_vector_result.success:
            log_state(QuestionLogs.CACHE_TIER2_VECTOR_HIT, function="multi_tier_cache_system", user_id=user_id)
            #if we found cached answer in hot cache we put said QnA in slapback! to pupulate tier-1
            asyncio.create_task(
                normal_redis_instance.setex(cache_key, 86400, json.dumps(redis_vector_result.data))
            )
            data = {
                "cache_verdict": cache_policy,
                "data": redis_vector_result.data
            }
            return APIResponse(
                data=data,
                success=True,
                error_code=None,
                error_message=None
            )
        


    
        # TIER 3: COLD SIMILARITY SEARCH (Persistent Chroma VDB Cache)
        cache_vdb: Chroma | None = await get_user_cache_vdb(user_id=user_id, db=db) 
        if cache_vdb is not None:
            chroma_result: APIResponse = await get_cached(question=question, cache_vdb=cache_vdb, db=db, user_id=user_id, ai_type=ai_type, convo_id=convo_id)
            
            if chroma_result.success:
                log_state(QuestionLogs.CACHE_TIER3_CHROMA_HIT, function="multi_tier_cache_system", user_id=user_id)
                source_versions: dict | None = chroma_result.data["source_versions"]
                cached_answer: dict = chroma_result.data["answer"]
                
                # Fire backfill in the background (Non-blocking!)
                async def run_both():
                    await asyncio.gather(
                        normal_redis_instance.setex(cache_key, 86400, json.dumps(cached_answer)), 
                        
                        save_to_redis_vector_cache(
                            user_id=user_id,
                            question=question,
                            response_payload=chroma_result.data["answer"],
                            redis_client=normal_redis_instance,
                            doc_name=doc_names,
                            source_versions=source_versions,
                            ai_type=ai_type,
                            convo_id=convo_id
                        ),
                        return_exceptions=True  
                    )

                asyncio.create_task(run_both())
                data = {"cache_verdict": cache_policy, "data": chroma_result.data["answer"]}
                return APIResponse(success=True, data=data, error_code=None, error_message=None)
            else: #treat not finding cache_vdb as fail of cache and run full bs!
                data = {
                    "cache_verdict": cache_policy,
                    "data": None
                }
                log_state(QuestionLogs.CACHE_SYSTEM_MISS_FALLBACK, function="multi_tier_cache_system", user_id=user_id)
                return APIResponse(
                    success=False,
                    data=data,
                    error_code=None,
                    error_message=None
                )
        else:
            data = {
                "cache_verdict": cache_policy,
                "data": None
            }
            log_state(QuestionLogs.CACHE_VDB_NOT_FOUND, function="multi_tier_cache_system", user_id=user_id)
            return APIResponse(
                    success=False,
                    data=data,
                    error_code=None,
                    error_message=None
                )

