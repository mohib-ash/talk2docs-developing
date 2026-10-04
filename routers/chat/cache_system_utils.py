import hashlib
import json
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from db_tables.tables import Document
from sqlalchemy import select



async def get_corpus_source_versions_by_id(user_id: int, db_session: AsyncSession, doc_name: str | list[str] | None = None) -> dict[int, int]:

    # 1. Normalize target documents based on scope request
    target_docs = None
    if isinstance(doc_name, str):
        target_docs = [doc_name]
    elif doc_name:
        target_docs = list(set(doc_name))

    # 2. Query Document ID and Version strictly filtered by user_id
    stmt = select(Document.doc_id, Document.version).where(Document.user_id == user_id)
    
    # 3. Apply scoping filter if specific document names were requested
    if target_docs is not None:
        stmt = stmt.where(Document.original_filename.in_(target_docs))

    # 4. Execute query asynchronously
    result = await db_session.execute(stmt)
    rows = result.all()

    # 5. Format as {doc_id: version} with integer keys and integer values
    return {int(row.doc_id): int(row.version) for row in rows}



def generate_cache_key(user_id: int, question: str, doc_name: list[str] | str | None, convo_id: Optional[str] = None) -> str:
    normalized_q = question.strip().lower()

    if isinstance(doc_name, str):
        doc_name = [doc_name]

    sorted_docs = sorted(doc_name) if doc_name else []

    # convo_id:
    #   None          -> answer_ai
    #   actual ID     -> continue_convo
    payload = f"{user_id}:{normalized_q}:{json.dumps(sorted_docs)}:{convo_id}"
    hash_sig = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"ai_hot_cache:{user_id}:{hash_sig}"

