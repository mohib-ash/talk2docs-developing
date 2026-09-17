import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import cohere
from groq import BaseModel
from langchain_chroma import Chroma
from redis import Redis
from sqlalchemy import select
from Ai.re_ranker_via_encoder import cohere_rerank
from celery_worker.Tasks.Ai_worker.ai_worker import save_validated_doc_task, multi_index_db_creation_task, global_bm25_build_task
from fastapi import UploadFile
from db_tables.tables import BM25Resource, Document
from routers.Ai.ai_services import UPLOAD_DIR, ALLOWED_MIME_TYPES
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES
from utils.config import settings
from utils.logging.helper_log import log_state
from utils.logging.logEvents import QuestionLogs, UploadFileLogs
from utils.schemas import APIResponse, DocumentStatus, MultiIndexStatus, TokenDataSchema, passed_vlidation_reponce
import secrets
from utils.schemas import BM25Status
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document as LangChainDocument
import random
import json
import asyncio
from typing import Any





async def get_batch_document_versions(doc_ids: list[int], db: AsyncSession) -> dict[int, int]:
    if not doc_ids:
        return {}
    stmt = select(Document.doc_id, Document.version).where(Document.doc_id.in_(doc_ids))
    result = await db.execute(stmt)
    return {row.doc_id: row.version for row in result.all()}






#it always receive the cachable questions! (as such if success=False means go Answer_ai)
async def get_cached(model: Any, question: str, cache_vdb: Chroma, db: AsyncSession, user_id: int) -> APIResponse:
    log_state(QuestionLogs.CHROMA_SIMILARITY_SEARCH_STARTED, function="get_cached", user_id=user_id)
    try:
        docs_and_scores: list[tuple[LangChainDocument, float]] = await asyncio.to_thread(
            cache_vdb.similarity_search_with_score,
            query=question,
            k=10
        )
    except Exception as exc:
        log_state(QuestionLogs.CHROMA_SIMILARITY_SEARCH_ERROR, function="get_cached", exc=str(exc), user_id=user_id)
        return APIResponse(success=False, data=None, error_code=None, error_message=str(exc))

    #No candidates -> miss
    if not docs_and_scores:
        log_state(QuestionLogs.CHROMA_CANDIDATES_NOT_FOUND, function="get_cached", user_id=user_id)
        return APIResponse(success=False, data=None, error_code=None, error_message=None)


    # 5. Deterministic validation (Batch fetch all doc IDs to prevent N+1 queries)
    all_doc_ids = set() #gets all the doc_ids form "source_versions" metadata field!
    for doc, _ in docs_and_scores:
        
        if raw_versions:
            cached_versions = json.loads(raw_versions) if isinstance(raw_versions, str) else raw_versions
            
            for doc_id_str in cached_versions:
                all_doc_ids.add(int(doc_id_str)) 


    live_versions_map = {}
    if all_doc_ids:
        log_state(QuestionLogs.CHROMA_BATCH_VERSION_FETCH, function="get_cached", user_id=user_id)
        live_versions_map: dict[int, int] = await get_batch_document_versions(doc_ids=list(all_doc_ids), db=db)



    valid_candidates = []
    for doc, score in docs_and_scores:
        raw_versions: dict[int, int] = doc.metadata.get("source_versions")
        if not raw_versions:
            continue
            
        cached_versions: dict[int, int] = json.loads(raw_versions) if isinstance(raw_versions, str) else raw_versions
        is_stale = False
        
        
        
        for doc_id_str, cached_version in cached_versions.items():
            doc_id = int(doc_id_str) 
            live_version = live_versions_map.get(doc_id) 
            
            if live_version is None or live_version != cached_version: 
                is_stale = True
                break
                
        if not is_stale: #if they match append em
            valid_candidates.append(doc)

    # 6. Nothing valid -> miss
    if not valid_candidates:
        log_state(QuestionLogs.CHROMA_NO_VALID_CANDIDATES, function="get_cached", user_id=user_id)
        return APIResponse(success=False, data=None, error_code=None, error_message=None)

    log_state(QuestionLogs.CHROMA_VALID_CANDIDATES_FOUND, function="get_cached", user_id=user_id)
    

    # 7. the great trenqulizer
    documents_for_rerank = [
        LangChainDocument(
            page_content=(
                f"Cached Question:\n{doc.metadata.get('question')}\n\n"
                f"Cached Answer:\n{doc.metadata.get('llm_response')}"
            ),
            metadata=doc.metadata
        )
        for doc in valid_candidates
    ]

    log_state(QuestionLogs.CHROMA_RERANK_STARTED, function="get_cached", user_id=user_id)
    rerank_response = await cohere_rerank(
        question=question,
        user_id=user_id,
        received_docs=documents_for_rerank,
        top_k=3
    )

    if not rerank_response.success or not rerank_response.data:
        log_state(QuestionLogs.CHROMA_RERANK_FAILED, function="get_cached", user_id=user_id)
        return APIResponse(success=False, data=None, error_code=None, error_message=None)

    # 8. Take highest reranked candidate from our reranker response
    best_doc = rerank_response.data[0]
    best_score = best_doc.metadata.get("rerank_score", 0.0) #in cohore i add this metadata to already existing dw


    # 9. Relevance gate & 10% exploration refresh (Deliberate Miss rate)
    RELEVANCE_THRESHOLD = 0.85
    DELIBERATE_MISS_RATE = 0.10

    # Low relevance -> Weak candidate -> Deliberate Miss
    if best_score < RELEVANCE_THRESHOLD:
        log_state(QuestionLogs.CHROMA_RELEVANCE_THRESHOLD_FAILED, function="get_cached", user_id=user_id)
        return APIResponse(success=False, data=None, error_code=None, error_message=None)

    # High relevance, but hit the 10% exploration roll -> Forced Refresh -> Deliberate Miss
    if random.random() < DELIBERATE_MISS_RATE: #random.random() gives a number between 0 and 1.
        log_state(QuestionLogs.CHROMA_DELIBERATE_MISS_TRIGGERED, function="get_cached", user_id=user_id)
        return APIResponse(success=False, data=None, error_code=None, error_message=None)

    
    # 10. Return stored AnswerAI payload on HIT
    log_state(QuestionLogs.CHROMA_CACHE_HIT, function="get_cached", user_id=user_id)
    raw_response = best_doc.metadata.get("full_llm_response")
    parsed_answer = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
    
    raw_versions = best_doc.metadata.get("source_versions")
    versions = (
        json.loads(raw_versions)
        if isinstance(raw_versions, str)
        else (raw_versions or {})
    )
    winning_source_versions = {
        int(k): int(v)
        for k, v in versions.items()
    }
    
    internal_payload = {
        "answer": parsed_answer,
        "source_versions": winning_source_versions
    }
    return APIResponse(
        success=True,
        data=internal_payload,
        error_code=None,
        error_message=None,
    )
    






import asyncio
import hashlib
import json
from typing import Any
import numpy as np
from pydantic import BaseModel
from redis.asyncio import Redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import (
    IndexDefinition,
    IndexType,
)
from redis.commands.search.query import Query
from redis.exceptions import ResponseError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from utils.embedding_model import embedding_model


def normalize_cache_scope(doc_name: str | list[str] | None) -> str:
    """Creates a deterministic, sorted, deduplicated scope token to guarantee isolation."""
    if isinstance(doc_name, str):
        docs = [doc_name]
    elif doc_name:
        docs = sorted(set(doc_name))
    else:
        return "global"
    return "__".join(docs)


async def generate_embedding(text: str) -> list[float]:
    return await asyncio.to_thread(embedding_model.embed_query, text)


# alr now this exists for each user
async def create_redis_vector_index(user_id: int, redis_client: Redis, dim: int = 384):
    flag_key = f"meta:index_initialized:{user_id}" 
    
    log_state(QuestionLogs.REDIS_INDEX_CHECK_STARTED, function="create_redis_vector_index", user_id=user_id)
    if await redis_client.exists(flag_key):
        log_state(QuestionLogs.REDIS_INDEX_EXISTS_CACHED, function="create_redis_vector_index", user_id=user_id)
        return 
    
    index_name = f"idx:user_cache:{user_id}"
    schema = (
        VectorField(
            "vector",                  
            "HNSW",                      
            {
                "TYPE": "FLOAT32",      
                "DIM": dim,              
                "DISTANCE_METRIC": "COSINE", 
                "M": 16,                  
                "EF_CONSTRUCTION": 200,   
                "EF_RUNTIME": 10          
            }
        ),
        TagField("scope"), 
    )
    
    try:
        log_state(QuestionLogs.REDIS_INDEX_CREATION_STARTED, function="create_redis_vector_index", user_id=user_id)
        await redis_client.ft(index_name).create_index(
            schema, 
            definition=IndexDefinition(
                prefix=[f"cache:doc:{user_id}:"], 
                index_type=IndexType.HASH 
            )
        )

        log_state(QuestionLogs.REDIS_INDEX_CREATED_SUCCESS, function="create_redis_vector_index", user_id=user_id)
    except ResponseError as e:
        if "Index already exists" not in str(e):
            log_state(QuestionLogs.REDIS_INDEX_CREATION_ERROR, function="create_redis_vector_index", exc=str(e), user_id=user_id)
            raise e
        log_state(QuestionLogs.REDIS_INDEX_ALREADY_EXISTS, function="create_redis_vector_index", user_id=user_id)

    await redis_client.set(flag_key, "1") 
    log_state(QuestionLogs.REDIS_INDEX_INITIALIZATION_COMPLETE, function="create_redis_vector_index", user_id=user_id)



async def save_to_redis_vector_cache(user_id: int, question: str, response_payload: dict | BaseModel | Any, redis_client: Redis, doc_name: str | list[str] | None = None, source_versions: dict[str, str] | None = None):
    log_state(QuestionLogs.REDIS_VECTOR_SAVE_STARTED, function="save_to_redis_vector_cache", user_id=user_id)
    try:
        # 0. Ensure the RediSearch index exists for this user before saving!
        await create_redis_vector_index(user_id=user_id, redis_client=redis_client)
        log_state(QuestionLogs.REDIS_VECTOR_INDEX_ENSURED, function="save_to_redis_vector_cache", user_id=user_id)
        
        # 1. Generate the embedding for the question
        vector = await generate_embedding(question)
        log_state(QuestionLogs.REDIS_VECTOR_EMBEDDING_GENERATED, function="save_to_redis_vector_cache", user_id=user_id)
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        
        # Handle Pydantic v2 serialization cleanly
        if isinstance(response_payload, BaseModel):
            payload_data = response_payload.model_dump_json()
        else:
            payload_data = json.dumps(response_payload)
        log_state(QuestionLogs.REDIS_VECTOR_PAYLOAD_SERIALIZED, function="save_to_redis_vector_cache", user_id=user_id)

        # 2. Build deterministic scope and hash it into the entry ID to prevent collisions across documents
        normalized_question = question.strip().lower()
        scope = normalize_cache_scope(doc_name) #the a.pdf__b.pdf
        
        scope_payload = f"{normalized_question}:{scope}"
        entry_id = hashlib.sha256(scope_payload.encode("utf-8")).hexdigest()
        
        redis_key = f"cache:doc:{user_id}:{entry_id}"
        
        # Ensure source_versions values are strictly standardized to dict[str, str]
        normalized_versions = {int(k): int(v) for k, v in (source_versions or {}).items()}

        # 3. Save it as a Redis Hash (Including strict string-typed source_versions for provenance validation)
        await redis_client.hset( 
            redis_key,
            mapping={
                "vector": vector_bytes,                            
                "response_data": payload_data,                     
                "scope": scope,                                    
                "source_versions": json.dumps(normalized_versions) 
            }
        )
        log_state(QuestionLogs.REDIS_VECTOR_HASH_SAVED, function="save_to_redis_vector_cache", user_id=user_id)
        await redis_client.expire(redis_key, 86400) # 24 hours
        log_state(QuestionLogs.REDIS_VECTOR_TTL_SET, function="save_to_redis_vector_cache", user_id=user_id)

    except Exception as e:
        log_state(QuestionLogs.REDIS_VECTOR_SAVE_ERROR, function="save_to_redis_vector_cache", exc=str(e), user_id=user_id)
        raise
    


def escape_redis_tag(value: str) -> str:
    special_chars = r",.<>{}[]\"':;!@#$%^&*()-+=~|"
    return "".join(
        f"\\{char}" if char in special_chars or char == " " else char
        for char in value
    )


async def get_redis_vector_cached(question: str, user_id: int, redis_client: Redis, top_k: int = 1, doc_name: str | list[str] | None = None, source_versions: dict[int, int] | None = None) -> APIResponse:
    log_state(QuestionLogs.REDIS_VECTOR_SEARCH_STARTED, function="get_redis_vector_cached", user_id=user_id)
    try:
        # 1. Generate embedding for the incoming question 
        log_state(QuestionLogs.REDIS_VECTOR_GENERATING_EMBEDDING, function="get_redis_vector_cached", user_id=user_id)
        question_vector: list[float] = await generate_embedding(question) 
        
        # Redis expects raw bytes for float32 vector queries
        vector_bytes = np.array(question_vector, dtype=np.float32).tobytes()
        
        # 2. Define your RediSearch index name for this user 
        index_name = f"idx:user_cache:{user_id}"
        
        
        # 3. Construct the KNN query syntax with exact scope pre-filtering
        scope = normalize_cache_scope(doc_name) 
        log_state(QuestionLogs.REDIS_VECTOR_SCOPE_NORMALIZED, function="get_redis_vector_cached", user_id=user_id)
        
        
        escaped_scope = escape_redis_tag(scope)
        query_str = (
            f"(@scope:{{{escaped_scope}}})"
            f"=>[KNN {top_k} @vector $vec AS vector_score]"
        )
        print(f"DEBUG doc_name = {doc_name!r}")
        print(f"DEBUG scope = {scope!r}")
        print(f"DEBUG escaped_scope = {escaped_scope!r}")
        print(f"DEBUG query_str = {query_str!r}")
        
        
        # 4. Execute the FT.SEARCH command via redis-py
        query = (
            Query(query_str)
            .return_fields("response_data", "vector_score", "source_versions") 
            .sort_by("vector_score") 
            .dialect(2) 
        )
        
        log_state(QuestionLogs.REDIS_VECTOR_QUERY_EXECUTING, function="get_redis_vector_cached", user_id=user_id)
        results = await redis_client.ft(index_name).search( #index_name = f"idx:user_cache:{user_id}" !! see this what i was sayin in prefix
            query, 
            query_params={"vec": vector_bytes}
        )
        
        
        # 5. Check if we found a match, evaluate similarity score, and validate provenance strictly
        if results and results.docs:
            doc = results.docs[0]
            score = float(doc.vector_score)
            
            # Fail fast on geometric distance miss
            if score > 0.20:
                log_state(QuestionLogs.REDIS_VECTOR_SCORE_MISMATCH, function="get_redis_vector_cached", user_id=user_id)
                return APIResponse(success=False, data=None) #Cache-miss
            
            # Mandatory fail-closed provenance validation provenance==origan!
            if not source_versions:
                log_state(QuestionLogs.REDIS_VECTOR_INCOMING_PROVENANCE_MISSING, function="get_redis_vector_cached", user_id=user_id)
                return APIResponse(success=False, data=None) #cache-miss
            
            if not hasattr(doc, "source_versions") or not doc.source_versions:
                log_state(QuestionLogs.REDIS_VECTOR_STORED_PROVENANCE_MISSING, function="get_redis_vector_cached", user_id=user_id)
                return APIResponse(success=False, data=None) # Missing provenance on cache hit -> MISS
            
            normalized_incoming = {int(k): int(v) for k, v in source_versions.items()}
            versions = json.loads(doc.source_versions) 
            stored_versions = {
                int(k): int(v)
                for k, v in versions.items()
            }
            
            if stored_versions != normalized_incoming:
                log_state(QuestionLogs.REDIS_VECTOR_VERSION_MISMATCH, function="get_redis_vector_cached", user_id=user_id)
                return APIResponse(success=False, data=None) # Corpus version mismatch -> MISS
            
            # If all checks pass, parse response and return hit
            log_state(QuestionLogs.REDIS_VECTOR_HIT_FOUND, function="get_redis_vector_cached", user_id=user_id)
            response_data = json.loads(doc.response_data) #the full model answer!
            return APIResponse(success=True, data=response_data)
                
        log_state(QuestionLogs.REDIS_VECTOR_CACHE_MISS, function="get_redis_vector_cached", user_id=user_id)
        return APIResponse(success=False, data=None)

    except Exception as e:
        # If Redis index doesn't exist yet or query fails, gracefully fallback to Tier 3 (Chroma)
        log_state(QuestionLogs.REDIS_VECTOR_ERROR, function="get_redis_vector_cached", exc=str(e), user_id=user_id)
        return APIResponse(success=False, data=None)