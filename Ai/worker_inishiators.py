import asyncio
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from langchain_chroma import Chroma
from sqlalchemy import select
from celery_worker.Tasks.Ai_worker.ai_worker import push_responce_in_cache_worker, push_responce_in_LTM_worker
from fastapi import UploadFile
from db_tables.tables import BM25Resource, Document
from routers.Ai.ai_services import UPLOAD_DIR, ALLOWED_MIME_TYPES
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES
from utils.logging.helper_log import log_state
from utils.logging.logEvents import ServiceLog, UploadFileLogs
from utils.schemas import APIResponse, CacheVDBStatus, DocumentStatus, LogState, MultiIndexStatus, TokenDataSchema, passed_vlidation_reponce
import secrets
from utils.schemas import BM25Status
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
import json
from langchain_core.documents import Document as LangChainDocument
from sqlalchemy import select
from db_tables.tables import Document





async def push_responce_in_cache_inishiator(model_output: Any, question: str,  user_id: int, doc_name: list[str] | None, ai_type: str, convo_id: Optional[str] = None) -> APIResponse:
    model_dict = model_output.model_dump() if hasattr(model_output, "model_dump") else model_output
    task = push_responce_in_cache_worker.delay(
        user_id=user_id,
        question=question,
        model_output_dict=model_dict,
        doc_name=doc_name,
        ai_type=ai_type,
        convo_id=convo_id
    )
    
    return APIResponse(
        success=True,
        data={
            "task_id": task.id if hasattr(task, "id") else str(task.id),
            "status": "PROCESSING"
        },
        error_code=None,
        error_message=None,
    )

async def push_responce_in_LTM_inishiator(model_output: Any, question: str, user_id: int, ai_type: str, convo_id: str) -> APIResponse:
    model_dict = model_output.model_dump() if hasattr(model_output, "model_dump") else model_output
    
    task = push_responce_in_LTM_worker.delay(
        user_id=user_id,
        question=question,
        model_output_dict=model_dict,
        ai_type=ai_type,
        convo_id=convo_id
    )
    return APIResponse(
        success=True,
        data={
            "task_id": task.id if hasattr(task, "id") else str(task.id),
            "status": "PROCESSING"
        },
        error_code=None,
        error_message=None,
    )



