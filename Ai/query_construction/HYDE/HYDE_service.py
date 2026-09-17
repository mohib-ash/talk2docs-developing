from Ai import query_classifier
from Ai.ai_utils import build_get_retriever
from Ai.query_classifier import QueryClassificationResult, QueryTechnique
from Ai.query_construction.HYDE import HYDE_file
from utils.APIResponce_error_code_enum import USER_ERROR_CODES, SYSTEM_ERROR_CODES
from typing import Annotated, Any, Dict, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import BaseModel, Field, StringConstraints, field_validator, ConfigDict, model_validator
from Ai.retry_logic import check_provider_quota
from langchain_core.output_parsers import PydanticOutputParser
from core.Exceptions.exceptions import AIServiceException
from utils.logging.logEvents import ExceptionLog, HyDELog, ProviderLog, RepairLog, SecurityLog, ServiceLog
from utils.schemas import APIResponse, QuestionRequest
# from Ai.intent_classifier import  get_user_intent
from pydantic import ValidationError
from Ai.raw_and_parsed_clean import extract_raw_data, extract_parsed_data
from utils.logging.helper_log import log_state, LogState 
from langchain_chroma import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LangChainDocument
import asyncio
import json 
import traceback 
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession
from Ai.ai_utils import safe_retrieve


async def use_HYDE(user_id: int, hyde_doc: str, retriever: EnsembleRetriever) -> APIResponse:
    log_state(HyDELog.HYDE_RETRIEVAL_STARTED, function="use_HYDE", user_id=user_id)
    
    #Retrieve chunks using the HyDE passage
    try: #try for rertiver fail not hyd_doc file
        retrieved_docs: list[LangChainDocument] = await safe_retrieve(retriever, hyde_doc)
    except Exception as e:
        log_state(HyDELog.HYDE_RETRIEVAL_FAILED, level=LogState.EXCEPTION, function="use_HYDE", exc=e, user_id=user_id)
        log_state(HyDELog.EXITING_HYDE_RETRIVER, function="use_HYDE", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.HYDE_RETRIVER_FAILURE.value,
            error_message="Retriver failed to get results for hyde_docs"
        )
    
    #upper was retiver expction, this is if that thang came empty
    if not retrieved_docs:
        log_state(HyDELog.HYDE_RETRIEVAL_FAILED, function="use_HYDE", user_id=user_id)
        log_state(HyDELog.EXITING_HYDE_RETRIVER, function="use_HYDE", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.NO_DATA_FOUND_BY_RETRIVER.value,
            error_message="No matching text chunks retrieved."
        )

    log_state(HyDELog.HYDE_RETRIEVAL_SUCCESS, function="use_HYDE", user_id=user_id)
    log_state(HyDELog.EXITING_HYDE_RETRIVER, function="use_HYDE", user_id=user_id)
    return APIResponse(
        success=True,
        data=retrieved_docs,
        error_code=None,
        error_message=None
    )