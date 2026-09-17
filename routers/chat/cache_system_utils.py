import hashlib
import json
from sqlalchemy.ext.asyncio import AsyncSession
from db_tables.tables import Document
from sqlalchemy import select




async def get_corpus_source_versions_by_id(user_id: int, db_session: AsyncSession, doc_name: str | list[str] | None = None) -> dict[int, int]:
    """
    Fetches current document versions from PostgreSQL for the requested scope.
    Returns a standardized dictionary mapping document IDs to versions: {12: 4, 18: 2}
    """
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



#i plan to make doc_name none or str, coz front-end will have list of user's uploaded documents too talk to, if he doesnt chose then he talks to all thus None
def generate_cache_key(user_id: int, question: str, doc_name: list[str] | str | None) -> str:
    # Normalize the question for consistent exact-match hits
    normalized_q = question.strip().lower()
    
    # Handle doc_name if passed as a single string instead of list
    if isinstance(doc_name, str):
        doc_name = [doc_name]

    # Sort doc_names so order doesn't break the cache key hash
    sorted_docs = sorted(doc_name) if doc_name else [] #sort coz if user asks in a,b,c then in c,b,a -> cache will create new! we dont want that!
    
    # Construct a unique payload string combining user, question, and scoped documents
    payload = f"{user_id}:{normalized_q}:{json.dumps(sorted_docs)}"
    
    # Hash it using SHA-256 to keep the Redis key clean and fixed-length
    hash_sig = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    
    return f"ai_hot_cache:{user_id}:{hash_sig}"

