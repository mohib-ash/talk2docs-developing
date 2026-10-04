from redis.asyncio import Redis
from langchain_community.retrievers import BM25Retriever
from pathlib import Path
from langchain_chroma import Chroma
from sqlalchemy import select
from db_tables.tables import BM25Resource, CacheVDBResource, Document, LTMVDBResource
from utils.config import settings
from routers.Ai.ai_services import embedding_model
from sqlalchemy.ext.asyncio import AsyncSession

from utils.logging.helper_log import log_state
from utils.logging.logEvents import QuestionLogs
import pickle
import asyncio

from utils.schemas import CacheVDBStatus, LTMVDBStatus, LogState


async def get_user_collection_name(user_id: int, db: AsyncSession) -> str | None:

    stmt = (
        select(Document.collection_name)
        .where(Document.user_id == user_id)
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()



async def get_user_vdb(user_id: int, db: AsyncSession) -> Chroma | None:
    log_state(QuestionLogs.GETTING_USER_VDB, function="get_user_vdb", user_id=user_id)
    user_vdb_path = Path(settings.chroma_db_dir) / f"user_{user_id}"
    log_state(QuestionLogs.FETCHING_COLLECTION_NAME, function="get_user_vdb", user_id=user_id)
    collection_name = await get_user_collection_name(user_id=user_id, db=db)
    
    if collection_name:
        log_state(QuestionLogs.VDB_FOUND, function="get_user_vdb", user_id=user_id)
        return Chroma(
            persist_directory=str(user_vdb_path),
            collection_name=collection_name,
            embedding_function=embedding_model,
        )
    else:
        log_state(QuestionLogs.VDB_NOT_FOUND, function="get_user_vdb", user_id=user_id)
        return None


async def get_user_summary_vdb(user_id: int, db: AsyncSession) -> Chroma | None:
    log_state(QuestionLogs.CHECKING_USER_VDB, function="get_user_summary_vdb", user_id=user_id)
    
    user_vdb_path = Path(settings.chroma_db_dir) / f"user_{user_id}"
    collection_name = await get_user_collection_name(user_id=user_id, db=db)
    
    if collection_name:
        log_state(QuestionLogs.VDB_FOUND, function="get_user_summary_vdb", user_id=user_id)
        return Chroma(
            persist_directory=str(user_vdb_path),
            collection_name=f"{collection_name}_summary",
            embedding_function=embedding_model,
        )
    else:
        log_state(QuestionLogs.VDB_NOT_FOUND, function="get_user_summary_vdb", user_id=user_id)
        return None



async def get_user_explanation_vdb(user_id: int, db: AsyncSession) -> Chroma | None:
    log_state(QuestionLogs.CHECKING_USER_VDB, function="get_user_explanation_vdb", user_id=user_id)
    
    user_vdb_path = Path(settings.chroma_db_dir) / f"user_{user_id}"
    collection_name = await get_user_collection_name(user_id=user_id, db=db)
    
    if collection_name:
        log_state(QuestionLogs.VDB_FOUND, function="get_user_explanation_vdb", user_id=user_id)
        return Chroma(
            persist_directory=str(user_vdb_path),
            collection_name=f"{collection_name}_explanation",
            embedding_function=embedding_model,
        )
    else:
        log_state(QuestionLogs.VDB_NOT_FOUND, function="get_user_explanation_vdb", user_id=user_id)
        return None


async def get_user_cache_vdb(user_id: int, db: AsyncSession) -> Chroma | None:
    log_state(QuestionLogs.CHECKING_USER_VDB, function="get_user_cache_vdb", user_id=user_id)
    
    # 1. Fetch the cache VDB resource record from the DB
    stmt = select(CacheVDBResource).where(CacheVDBResource.user_id == user_id)
    result = await db.execute(stmt)
    cache_res = result.scalar_one_or_none()
    
    # 2. Verify it exists, is READY, and has a valid path stored
    if cache_res is None or cache_res.status != CacheVDBStatus.READY or not cache_res.vdb_path:
        log_state(QuestionLogs.VDB_NOT_FOUND, function="get_user_cache_vdb", user_id=user_id)
        return None
    
    log_state(QuestionLogs.VDB_FOUND, function="get_user_cache_vdb", user_id=user_id)
    
    # 3. Return the Chroma instance pointing to the user's isolated directory
    return await asyncio.to_thread(
        lambda: Chroma(
            persist_directory=cache_res.vdb_path,
            collection_name=f"question_cache_{user_id}",
            embedding_function=embedding_model,
        )
    )


async def get_user_ltm_vdb(user_id: int, db: AsyncSession) -> Chroma | None:
    log_state(QuestionLogs.CHECKING_USER_LTM_VDB, function="get_user_ltm_vdb", user_id=user_id)
    
    # 1. Fetch the LTM VDB resource record from the DB
    stmt = select(LTMVDBResource).where(LTMVDBResource.user_id == user_id)
    result = await db.execute(stmt)
    ltm_res = result.scalar_one_or_none()
    
    # 2. Verify it exists, is READY, and has a valid path stored
    if ltm_res is None or ltm_res.status != LTMVDBStatus.READY or not ltm_res.vdb_path:
        log_state(QuestionLogs.LTM_VDB_NOT_FOUND, function="get_user_ltm_vdb", user_id=user_id)
        return None
    
    log_state(QuestionLogs.LTM_VDB_FOUND, function="get_user_ltm_vdb", user_id=user_id)
    
    # 3. Return the Chroma instance pointing to the user's isolated LTM directory
    return await asyncio.to_thread(
        lambda: Chroma(
            persist_directory=ltm_res.vdb_path,
            collection_name=f"ltm_memory_{user_id}",
            embedding_function=embedding_model,
        )
    )


async def get_user_bm25_retriever(
    user_id: int, 
    db: AsyncSession, 
    redis_client: Redis, 
    doc_name: list[str] | str | None = None, 
    k: int = 20
) -> BM25Retriever | None:
    """
    Loads the user's pre-built global BM25 retriever if no specific documents are requested (doc_name is None).
    If specific documents ARE requested (doc_name is provided), it loads the raw master docs, 
    filters them down to the selected files, and builds a targeted BM25Retriever on the fly.
    Cached in Redis per version for both paths!
    """
    # Normalize doc_name to a list of strings if it's passed as a single string
    if isinstance(doc_name, str):
        doc_name = [doc_name]

    bm25_stmt = select(BM25Resource).where(BM25Resource.user_id == user_id)
    bm25_result = await db.execute(bm25_stmt)
    bm25_res = bm25_result.scalar_one_or_none()

    user_dir: Path = Path(settings.chroma_db_dir) / f"user_{user_id}"

    def load_pickle(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Determine current version for cache keys
    if bm25_res:
        current_version: int = bm25_res.version
    else:
        current_version = 1


    # PATH 1: NO DOC_NAME (Global Search -> Pre-built Retriever) 
    if not doc_name:
        if bm25_res:
            retriever_pickle_path: Path = Path(bm25_res.index_path).parent / f"master_bm25_retriever_v{current_version}.pkl" if bm25_res.index_path else (user_dir / f"master_bm25_retriever_v{current_version}.pkl")
        else:
            retriever_pickle_path = user_dir / f"master_bm25_retriever_v1.pkl"

        redis_key: str = f"bm25_retriever:user_{user_id}:v{current_version}"
        cached_bytes: bytes | None = await redis_client.get(redis_key)
        
        if cached_bytes:
            print(f"[CACHE HIT] Loaded global BM25 retriever from Redis for user {user_id} (v{current_version})")
            return await asyncio.to_thread(pickle.loads, cached_bytes)
        
        print(f"[CACHE MISS] Loading global BM25 retriever from disk/file for user {user_id} (v{current_version})")
        master_retriever: BM25Retriever | None = None
        if retriever_pickle_path.exists():
            try:
                master_retriever = await asyncio.to_thread(load_pickle, retriever_pickle_path)
                serialized_retriever: bytes = await asyncio.to_thread(pickle.dumps, master_retriever)
                await redis_client.set(redis_key, serialized_retriever, ex=86400)
                return master_retriever
            except Exception as e:
                log_state(QuestionLogs.VDB_NOT_FOUND, level=LogState.EXCEPTION, function="get_user_bm25_retriever", user_id=user_id)

        # Fallback loop for global retriever versions
        fallback_version: int = current_version - 1
        while fallback_version >= 1:
            fallback_redis_key: str = f"bm25_retriever:user_{user_id}:v{fallback_version}"
            cached_fallback_bytes: bytes | None = await redis_client.get(fallback_redis_key)
            
            if cached_fallback_bytes:
                print(f"[CACHE HIT - FALLBACK] Loaded fallback global BM25 retriever from Redis for user {user_id} (v{fallback_version})")
                return await asyncio.to_thread(pickle.loads, cached_fallback_bytes)

            fallback_path: Path = user_dir / f"master_bm25_retriever_v{fallback_version}.pkl"
            if fallback_path.exists():
                try:
                    log_state(QuestionLogs.CHECKING_USER_VDB, function="get_user_bm25_retriever", user_id=user_id)
                    print(f"[CACHE MISS - FALLBACK] Loading fallback global BM25 retriever from disk for user {user_id} (v{fallback_version})")
                    master_retriever = await asyncio.to_thread(load_pickle, fallback_path)
                    serialized_fallback: bytes = await asyncio.to_thread(pickle.dumps, master_retriever)
                    await redis_client.set(fallback_redis_key, serialized_fallback, ex=86400)
                    return master_retriever
                except Exception as fallback_err:
                    log_state(QuestionLogs.VDB_NOT_FOUND, level=LogState.EXCEPTION, function="get_user_bm25_retriever", user_id=user_id)
            
            fallback_version -= 1

        return None

    # PATH 2: SPECIFIC DOC_NAME PROVIDED (Filtered Search -> Build on the fly) [tho we build on the fly the LangChainDocs are alredy made!]
    if bm25_res:
        pickle_path = Path(bm25_res.index_path) if bm25_res.index_path else (user_dir / f"master_bm25_docs_v{current_version}.pkl")
    else:
        pickle_path = user_dir / f"master_bm25_docs_v1.pkl"

    redis_key = f"bm25_docs:user_{user_id}:v{current_version}"
    cached_bytes = await redis_client.get(redis_key)
    
    master_docs = [] 
    if cached_bytes:
        print(f"[CACHE HIT] Loaded master docs from Redis for user {user_id} (v{current_version})")
        master_docs = await asyncio.to_thread(pickle.loads, cached_bytes)
    elif pickle_path.exists():
        print(f"[CACHE MISS] Loading master docs from disk for user {user_id} (v{current_version})")
        try:
            master_docs = await asyncio.to_thread(load_pickle, pickle_path)
            serialized_docs = await asyncio.to_thread(pickle.dumps, master_docs)
            await redis_client.set(redis_key, serialized_docs, ex=86400)
        except Exception as e:
            log_state(QuestionLogs.VDB_NOT_FOUND, level=LogState.EXCEPTION, function="get_user_bm25_retriever", user_id=user_id)
    else:
        fallback_version = current_version - 1
        while fallback_version >= 1:
            fallback_redis_key = f"bm25_docs:user_{user_id}:v{fallback_version}"
            cached_fallback_bytes = await redis_client.get(fallback_redis_key)
            
            if cached_fallback_bytes:
                print(f"[CACHE HIT - FALLBACK] Loaded fallback master docs from Redis for user {user_id} (v{fallback_version})")
                master_docs = await asyncio.to_thread(pickle.loads, cached_fallback_bytes)
                break

            fallback_path = user_dir / f"master_bm25_docs_v{fallback_version}.pkl"
            if fallback_path.exists():
                try:
                    log_state(QuestionLogs.CHECKING_USER_VDB, function="get_user_bm25_retriever", user_id=user_id)
                    print(f"[CACHE MISS - FALLBACK] Loading fallback master docs from disk for user {user_id} (v{fallback_version})")
                    master_docs = await asyncio.to_thread(load_pickle, fallback_path)
                    serialized_fallback = await asyncio.to_thread(pickle.dumps, master_docs)
                    await redis_client.set(fallback_redis_key, serialized_fallback, ex=86400)
                    break
                except Exception as fallback_err:
                    log_state(QuestionLogs.VDB_NOT_FOUND, level=LogState.EXCEPTION, function="get_user_bm25_retriever", user_id=user_id)
            
            fallback_version -= 1

    if not master_docs:
        return None
    
    filtered_docs = [doc for doc in master_docs if doc.metadata.get("file_name") in doc_name] #so we only use the ones which we need ;)

    if not filtered_docs:
        return None
    try:
        targeted_retriever = await asyncio.to_thread(
            BM25Retriever.from_documents,
            documents=filtered_docs,
            k=k,
        )
        return targeted_retriever
    except Exception:
        return None