import asyncio
import json
import re
import traceback
from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LangChainDocument
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from Ai.raw_and_parsed_clean import extract_parsed_data, extract_raw_data
from Ai.retry_logic import check_provider_quota
from core.Exceptions.exceptions import AIServiceException
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES, USER_ERROR_CODES
from utils.logging.helper_log import LogState, log_state
from utils.logging.logEvents import (
    AdvancedTranslationLog,
    ExceptionLog,
    MultiQueryLog,
    ProviderLog,
    RepairLog,
    SecurityLog,
    ServiceLog,
)
from utils.schemas import APIResponse, QuestionRequest
from sqlalchemy.ext.asyncio import AsyncSession
from collections.abc import Awaitable
from Ai.ai_utils import safe_retrieve

class TranslationOutput(BaseModel):
    """Structured output schema for cross-lingual and domain query translation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    translated_query: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description=(
            "The exact, normalized search query translated into concise, technical English. "
            "Must preserve all specific technical entities, error codes, database parameters, "
            "and framework names."
        ),
    )


TRANSLATION_SYSTEM_PROMPT = """You are an expert technical translator and search query optimization engineer.
YOUR GOAL:
Translate non-English user queries or heavy technical slang into precise, technical English optimized for document chunk retrieval.

RULES:
1. Preserve all exact code terms, SQL keywords, API parameters, and framework names.
2. Output clear, direct, technical English without introductory filler or conversational fluff.
3. Maintain 100% of the original question's core technical intent.

{format_instructions}
"""


async def advanced_translation_function(
    model: Any, question: str, user_id: int, retriever: EnsembleRetriever
) -> APIResponse:
    log_state(
        AdvancedTranslationLog.ADVANCED_TRANSLATION_STARTED,
        function="advanced_translation_function",
        user_id=user_id,
    )
    log_state(
        ServiceLog.AI_SERVICE_STARTED,
        function="advanced_translation_function",
        user_id=user_id,
    )

    parser = PydanticOutputParser(pydantic_object=TranslationOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", TRANSLATION_SYSTEM_PROMPT),
            ("human", "Query to translate: {question}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    raw_response = None
    extracted_parsed = None

    # 1. LLM Invocation & Parsing
    try:
        log_state(
            ProviderLog.AI_PROVIDER_REQUEST,
            function="advanced_translation_function",
            user_id=user_id,
        )
        raw_response = await (prompt | model).ainvoke({"question": question})

        cleaned_content = raw_response.content.strip()
        if cleaned_content.startswith("```"):
            cleaned_content = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned_content)
            cleaned_content = re.sub(r"\n?```$", "", cleaned_content).strip()

        extracted_parsed = parser.parse(cleaned_content)

        # FIXED: Moved inside the try block so it only fires on true provider & parsing success[cite: 10]
        log_state(
            ProviderLog.AI_PROVIDER_SUCCESS,
            function="advanced_translation_function",
            user_id=user_id,
        )

    except Exception as e:
        if check_provider_quota(e):
            log_state(
                ProviderLog.AI_PROVIDER_FAILURE,
                level=LogState.EXCEPTION,
                function="advanced_translation_function",
                exc=e,
                user_id=user_id,
            )
            log_state(
                ServiceLog.AI_SERVICE_FAILED,
                function="advanced_translation_function",
                user_id=user_id,
            )
            log_state(
                AdvancedTranslationLog.ADVANCED_TRANSLATION_FAILED,
                function="advanced_translation_function",
                user_id=user_id,
            )
            log_state(
                AdvancedTranslationLog.EXITING_ADVANCED_TRANSLATION,
                function="advanced_translation_function",
                user_id=user_id,
            )
            log_state(
                ServiceLog.EXITING_AI_SERVICE,
                function="advanced_translation_function",
                user_id=user_id,
            )

            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value,
                error_message="No more tokens left to process this request",
            )
        else:
            log_state(
                ProviderLog.AI_PROVIDER_FAILURE,
                level=LogState.EXCEPTION,
                function="advanced_translation_function",
                exc=e,
                user_id=user_id,
            )
            extracted_parsed = None

    target_query: str | None = (
        extracted_parsed.translated_query if extracted_parsed else None
    )

    # 2. Raw Repair Fallback Path
    if not target_query:
        log_state(
            RepairLog.AI_REPAIR_INITIALIZED,
            function="advanced_translation_function",
            user_id=user_id,
        )
        raw = getattr(raw_response, "content", None) if raw_response else None

        if not raw:
            log_state(
                ServiceLog.AI_SERVICE_FAILED,
                function="advanced_translation_function",
                user_id=user_id,
            )
            log_state(
                AdvancedTranslationLog.ADVANCED_TRANSLATION_FAILED,
                function="advanced_translation_function",
                user_id=user_id,
            )
            log_state(
                AdvancedTranslationLog.EXITING_ADVANCED_TRANSLATION,
                function="advanced_translation_function",
                user_id=user_id,
            )
            log_state(
                ServiceLog.EXITING_AI_SERVICE,
                function="advanced_translation_function",
                user_id=user_id,
            )
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
                error_message="Structured translation parsing failed.",
            )

        try:
            log_state(
                RepairLog.AI_REPAIR_STARTED,
                function="advanced_translation_function",
                user_id=user_id,
            )
            log_state(
                RepairLog.AI_REPAIR_IN_PROGRESS,
                function="advanced_translation_function",
                user_id=user_id,
            )

            recovered = await extract_raw_data(
                raw, parser, model, question, TranslationOutput
            )

            if not recovered or not getattr(recovered, "translated_query", None):
                log_state(
                    RepairLog.AI_REPAIR_FAILED,
                    function="advanced_translation_function",
                    user_id=user_id,
                )
                log_state(
                    ServiceLog.AI_SERVICE_FAILED,
                    function="advanced_translation_function",
                    user_id=user_id,
                )
                log_state(
                    AdvancedTranslationLog.ADVANCED_TRANSLATION_FAILED,
                    function="advanced_translation_function",
                    user_id=user_id,
                )
                log_state(
                    AdvancedTranslationLog.EXITING_ADVANCED_TRANSLATION,
                    function="advanced_translation_function",
                    user_id=user_id,
                )
                log_state(
                    ServiceLog.EXITING_AI_SERVICE,
                    function="advanced_translation_function",
                    user_id=user_id,
                )
                return APIResponse(
                    success=False,
                    data=None,
                    error_code=SYSTEM_ERROR_CODES.RAW_REPAIR_FAILURE.value,
                    error_message="Translation repair recovery failed.",
                )

            log_state(
                RepairLog.AI_REPAIR_SUCCESS,
                function="advanced_translation_function",
                user_id=user_id,
            )
            target_query = recovered.translated_query
        except Exception as e:
            if check_provider_quota(e):
                log_state(
                    ServiceLog.AI_MY_QUOTA_REACHED,
                    level=LogState.EXCEPTION,
                    function="advanced_translation_function",
                    user_id=user_id,
                    exc=e,
                )
                log_state(
                    RepairLog.AI_REPAIR_PREMATURELY_ENDED,
                    function="advanced_translation_function",
                    user_id=user_id,
                )
                log_state(
                    ServiceLog.AI_SERVICE_FAILED,
                    function="advanced_translation_function",
                    user_id=user_id,
                )
                log_state(
                    ServiceLog.EXITING_AI_SERVICE,
                    function="advanced_translation_function",
                    user_id=user_id,
                )

                return APIResponse(
                    success=False,
                    data=None,
                    error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value,
                    error_message="No more tokens left to process this request",
                )
            else:
                log_state(
                    RepairLog.AI_REPAIR_PREMATURELY_ENDED,
                    level=LogState.EXCEPTION,
                    function="advanced_translation_function",
                    user_id=user_id,
                    exc=e,
                )
                log_state(
                    ServiceLog.AI_SERVICE_FAILED,
                    function="advanced_translation_function",
                    user_id=user_id,
                )
                log_state(
                    ServiceLog.EXITING_AI_SERVICE,
                    function="advanced_translation_function",
                    user_id=user_id,
                )

                raise AIServiceException(
                    error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
                    message="AI output recovery process failed during query translation",
                ) from e

    # 3. Vector & BM25 Retrieval using Translated Query
    try:
        log_state(
            AdvancedTranslationLog.TRANSLATION_RETRIEVAL_STARTED,
            function="advanced_translation_function",
            user_id=user_id,
        )
        retrieved_docs: list[LangChainDocument] = await safe_retrieve(retriever, target_query)
    except Exception as exc:
        log_state(
            AdvancedTranslationLog.TRANSLATION_RETRIEVAL_FAILED,
            level=LogState.EXCEPTION,
            function="advanced_translation_function",
            exc=exc,
            user_id=user_id,
        )
        log_state(
            AdvancedTranslationLog.EXITING_ADVANCED_TRANSLATION,
            function="advanced_translation_function",
            user_id=user_id,
        )
        log_state(
            ServiceLog.AI_SERVICE_FAILED,
            function="advanced_translation_function",
            user_id=user_id,
        )
        log_state(
            ServiceLog.EXITING_AI_SERVICE,
            function="advanced_translation_function",
            user_id=user_id,
        )
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.INTERNAL_SYSTEM_ERROR.value,
            error_message=f"Retrieval failed after translation: {str(exc)}",
        )

    if not retrieved_docs:
        log_state(
            AdvancedTranslationLog.TRANSLATION_RETRIEVAL_FAILED,
            function="advanced_translation_function",
            user_id=user_id,
        )
        log_state(
            ServiceLog.AI_SERVICE_FAILED,
            function="advanced_translation_function",
            user_id=user_id,
        )
        log_state(
            AdvancedTranslationLog.EXITING_ADVANCED_TRANSLATION,
            function="advanced_translation_function",
            user_id=user_id,
        )
        log_state(
            ServiceLog.EXITING_AI_SERVICE,
            function="advanced_translation_function",
            user_id=user_id,
        )
        return APIResponse(
            success=False,
            data=None,
            error_code=USER_ERROR_CODES.NO_RELATED_DOCUMENT_FOUND.value,
            error_message="No documents found using translated query.",
        )

    log_state(
        AdvancedTranslationLog.TRANSLATION_RETRIEVAL_SUCCESS,
        function="advanced_translation_function",
        user_id=user_id,
    )

    # 4. Exit Lifecycle
    log_state(
        ServiceLog.AI_SERVICE_COMPLETED,
        function="advanced_translation_function",
        user_id=user_id,
    )
    log_state(
        AdvancedTranslationLog.ADVANCED_TRANSLATION_SUCCESS,
        function="advanced_translation_function",
        user_id=user_id,
    )
    log_state(
        AdvancedTranslationLog.EXITING_ADVANCED_TRANSLATION,
        function="advanced_translation_function",
        user_id=user_id,
    )
    log_state(
        ServiceLog.EXITING_AI_SERVICE,
        function="advanced_translation_function",
        user_id=user_id,
    )
    return APIResponse(
        success=True, data=retrieved_docs, error_code=None, error_message=None
    )