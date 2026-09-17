#two user have same doc name? that wont be isse each user gets his own vdb!
import asyncio
from langchain_chroma import Chroma
from utils.logging.helper_log import log_state
from utils.logging.logEvents import RetriverLog
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LangChainDocument
import time
from sqlalchemy.ext.asyncio import AsyncSession
from vector_db.chroma import get_user_bm25_retriever 
from redis.asyncio import Redis

embedding_lock = asyncio.Lock()
async def safe_retrieve(retriever, query: str) -> list[LangChainDocument]:
  async with embedding_lock:
    return await retriever.ainvoke(query)



def format_tiered_context(docs: list[LangChainDocument]) -> str:
    """Formats up to 5 retrieved docs into explicit Tier hierarchy blocks."""

    tier_labels = [
        "TIER 1: PRIMARY TRUTH (Highest Relevance)",
        "TIER 2: SUPPORTING HELPER (Secondary Relevance)",
        "TIER 3: GENERAL CONTEXT (Background 1)",
        "TIER 4: GENERAL CONTEXT (Background 2)",
        "TIER 5: GENERAL CONTEXT (Background 3)",
    ]

    formatted_chunks = []
    for idx, doc in enumerate(docs[:5]):
        label = tier_labels[idx] 
        
        metadata = doc.metadata or {}
        file_name = metadata.get("file_name", "Unknown Source")

        rerank_score = metadata.get("rerank_score")
        quote = metadata.get("key_evidence_quote")
        reasoning = metadata.get("rerank_reasoning")


        extra_lines = []
        if rerank_score is not None:
            extra_lines.append(f"Relevance Score: {rerank_score:.2f}")
        if quote:
            extra_lines.append(f"Key Evidence: \"{quote}\"")
        if reasoning:
            extra_lines.append(f"Reasoning: {reasoning}")


        extra_info_str = ("\n" + "\n".join(extra_lines)) if extra_lines else ""

        chunk_entry = (
            f"=== [{label}] ===\n" 
            f"Source File: {file_name}\n"
            f"{extra_info_str}\n" 
            f"Content: {doc.page_content}" 
        )
        formatted_chunks.append(chunk_entry)
    return "\n\n".join(formatted_chunks) 


async def build_get_retriever(user_vdb: Chroma, user_id: int, db: AsyncSession, redis_client: Redis, doc_name: list[str] | None = None, k: int = 20) -> EnsembleRetriever | VectorStoreRetriever | None:
    log_state(RetriverLog.BUILDING_RETRIVER_STARTED, function="build_get_retriever", user_id=user_id)
    if isinstance(doc_name, str): 
        doc_name = [doc_name]
    
    where_filter = {"file_name": {"$in": doc_name}} if doc_name and len(doc_name) > 0 else None 
    
    
    log_state(RetriverLog.FETCHING_DOCS_FOR_BM25, function="build_get_retriever", user_id=user_id)
    bm25_retriever: BM25Retriever | None = await get_user_bm25_retriever(user_id=user_id, db=db, redis_client=redis_client, doc_name=doc_name, k=k)

    
    search_kwargs = {"k": k}
    if where_filter:
        search_kwargs["filter"] = where_filter

    try:
        log_state(RetriverLog.BUILDING_VECTOR_RETRIVER, function="build_get_retriever", user_id=user_id)
        vector_retriever = user_vdb.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )
    except Exception:
        log_state(RetriverLog.BUILDING_VECTOR_RETRIVER_FAILURE, function="build_get_retriever", user_id=user_id)
        return None

    log_state(RetriverLog.BUILDING_VECTOR_RETRIVER_SUCCESS, function="build_get_retriever", user_id=user_id)

    
    if not bm25_retriever:
        log_state(RetriverLog.COULD_NOT_FIND_DOCS_FOR_BM25, function="build_get_retriever", user_id=user_id)
        log_state(RetriverLog.BUILDING_RETRIVER_SUCCESS, function="build_get_retriever", user_id=user_id)
        log_state(RetriverLog.EXITING_RETRIVER_BUILDER, function="build_get_retriever", user_id=user_id)
        return vector_retriever

    log_state(RetriverLog.BUILDING_BM25, function="build_get_retriever", user_id=user_id)
    log_state(RetriverLog.BUILDING_BM25_SUCCESS, function="build_get_retriever", user_id=user_id)

    try:
        log_state(RetriverLog.CREATING_HYBRID_RETRIVER, function="build_get_retriever", user_id=user_id)
        hybrid_retirver = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5],
            c=60,
        )
    except Exception:
        log_state(RetriverLog.CREATING_HYBRID_RETRIVER_FAILURE, function="build_get_retriever", user_id=user_id)
        return vector_retriever 
        
    log_state(RetriverLog.CREATING_HYBRID_RETRIVER_SUCCESS, function="build_get_retriever", user_id=user_id)
    log_state(RetriverLog.BUILDING_RETRIVER_SUCCESS, function="build_get_retriever", user_id=user_id)
    log_state(RetriverLog.EXITING_RETRIVER_BUILDER, function="build_get_retriever", user_id=user_id)
    return hybrid_retirver


