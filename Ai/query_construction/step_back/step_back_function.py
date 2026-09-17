import asyncio
import json
import re
import traceback
from collections.abc import Awaitable
from enum import Enum
from typing import Annotated, Any

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document as LangChainDocument
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from Ai.ai_utils import safe_retrieve
from Ai.query_construction.multi_query.multi_query_fuction import _reciprocal_rank_fusion
from Ai.raw_and_parsed_clean import extract_parsed_data, extract_raw_data
from Ai.retry_logic import check_provider_quota
from core.Exceptions.exceptions import AIServiceException
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES, USER_ERROR_CODES
from utils.logging.helper_log import LogState, log_state
from utils.logging.logEvents import (
    ExceptionLog,
    ProviderLog,
    RepairLog,
    ServiceLog,
    StepBackLog,
)
from utils.schemas import APIResponse


class StepBackOutput(BaseModel):
    """Structured output schema for step-back abstract query generation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    step_back_question: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description=(
            "A high-level, generic concept or principle question derived from the raw specific prompt. "
            "Strips out raw stack traces, local variable values, and hyper-specific line numbers to focus "
            "on underlying system architecture, protocols, or fundamental framework rules."
        ),
        examples=[
            "What causes SSL connection timeouts and EOF errors during PostgreSQL database handshakes?"
        ],
    )


STEP_BACK_SYSTEM_PROMPT = """You are an expert search query optimization engineer and systems architect.
YOUR GOAL:
Take a hyper-specific technical question or raw error stack trace and generate a high-level "step-back" question about the core underlying principles, concepts, or system mechanics.

RULES:
1. ABSTRACT AWAY NOISE: Remove file paths, port numbers, local variable names, and line numbers.
2. FOCUS ON PRINCIPLES: Ask about the foundational protocol, framework concept, or underlying operational mechanics.
3. Keep it clear, concise, and direct. Zero conversational filler.

{format_instructions}
"""

STEP_BACK_EXAMPLES = [
    {
        "question": "psycopg2.OperationalError: SSL SYSCALL error: EOF detected on port 5432",
        "step_back_question": "What causes SSL connection handshakes to fail with unexpected EOF errors in PostgreSQL connection pools?",
    },
    {
        "question": "FastAPI throwing 422 Unprocessable Entity on RequestValidationError for nested Pydantic v2 body schema",
        "step_back_question": "How does FastAPI handle body payload validation errors and customize RequestValidationError responses using Pydantic?",
    },
]

_sb_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Specific Question: {question}"),
    ("ai", "{step_back_question}"),
])

_sb_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=_sb_example_prompt,
    examples=STEP_BACK_EXAMPLES,
)


async def step_back_function(model: Any, question: str, user_id: int, retriever: EnsembleRetriever, top_n_final: int = 20) -> APIResponse:
    log_state(StepBackLog.STEP_BACK_STARTED, function="step_back_function", user_id=user_id)
    log_state(ServiceLog.AI_SERVICE_STARTED, function="step_back_function", user_id=user_id)

    # 1. Setup parser and prompt template
    parser = PydanticOutputParser(pydantic_object=StepBackOutput)

    step_back_prompt = ChatPromptTemplate.from_messages([
        ("system", STEP_BACK_SYSTEM_PROMPT),
        _sb_few_shot_prompt,
        ("human", "Specific Question: {question}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    raw_response = None
    extracted_parsed = None

    try:
        log_state(ProviderLog.AI_PROVIDER_REQUEST, function="step_back_function", user_id=user_id)
        log_state(ProviderLog.AI_PROVIDER_IN_PROCESSING, function="step_back_function", user_id=user_id)

        raw_response = await (step_back_prompt | model).ainvoke({"question": question})

        cleaned_content = raw_response.content.strip()
        if cleaned_content.startswith("```"):
            cleaned_content = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned_content)
            cleaned_content = re.sub(r"\n?```$", "", cleaned_content).strip()

        extracted_parsed = parser.parse(cleaned_content)

    except Exception as e:
        log_state(ProviderLog.AI_PROVIDER_FAILURE, level=LogState.EXCEPTION, function="step_back_function", exc=e, user_id=user_id)
        log_state(ServiceLog.AI_SERVICE_FAILED, function="step_back_function", user_id=user_id)
        log_state(StepBackLog.STEP_BACK_FAILED, function="step_back_function", user_id=user_id)
        log_state(StepBackLog.EXITING_STEP_BACK, function="step_back_function", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="step_back_function", user_id=user_id)

        if check_provider_quota(e):
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value,
                error_message="No more tokens left to process this request",
            )
        raise AIServiceException(
            error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
            message="AI processing failed during step-back prompt generation",
        ) from e

    log_state(ProviderLog.AI_PROVIDER_SUCCESS, level=LogState.INFO, function="step_back_function", user_id=user_id)

    step_back_query: str | None = extracted_parsed.step_back_question if extracted_parsed else None

    # Fallback to raw repair if primary output parsing failed
    if not step_back_query:
        log_state(RepairLog.AI_REPAIR_INITIALIZED, function="step_back_function", user_id=user_id)
        raw = getattr(raw_response, "content", None) if raw_response else None

        if not raw:
            log_state(ServiceLog.AI_SERVICE_FAILED, level=LogState.WARNING, function="step_back_function", user_id=user_id)
            log_state(RepairLog.AI_REPAIR_INITIALIZATION_STOPPED, level=LogState.WARNING, function="step_back_function", user_id=user_id)
            log_state(StepBackLog.STEP_BACK_FAILED, function="step_back_function", user_id=user_id)
            log_state(StepBackLog.EXITING_STEP_BACK, function="step_back_function", user_id=user_id)
            log_state(ServiceLog.EXITING_AI_SERVICE, level=LogState.WARNING, function="step_back_function", user_id=user_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
                error_message="Structured output parsing failed during step-back generation.",
            )

        try:
            log_state(RepairLog.AI_REPAIR_STARTED, function="step_back_function", user_id=user_id)
            recovered = await extract_raw_data(raw, parser, model, question, StepBackOutput)
        except Exception as e:
            if check_provider_quota(e):
                log_state(ServiceLog.AI_MY_QUOTA_REACHED, level=LogState.EXCEPTION, function="step_back_function", exc=e, user_id=user_id)
                log_state(RepairLog.AI_REPAIR_PREMATURELY_ENDED, function="step_back_function", user_id=user_id)
                log_state(ServiceLog.AI_SERVICE_FAILED, function="step_back_function", user_id=user_id)
                log_state(StepBackLog.STEP_BACK_FAILED, function="step_back_function", user_id=user_id)
                log_state(StepBackLog.EXITING_STEP_BACK, function="step_back_function", user_id=user_id)
                log_state(ServiceLog.EXITING_AI_SERVICE, function="step_back_function", user_id=user_id)
                return APIResponse(
                    success=False,
                    data=None,
                    error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value,
                    error_message="No more tokens left to process this request",
                )
            raise AIServiceException(
                error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
                message="AI step-back output recovery process failed",
            ) from e

        if not recovered or not getattr(recovered, "step_back_question", None):
            log_state(RepairLog.AI_REPAIR_FAILED, function="step_back_function", user_id=user_id)
            log_state(ServiceLog.AI_SERVICE_FAILED, function="step_back_function", user_id=user_id)
            log_state(StepBackLog.STEP_BACK_FAILED, function="step_back_function", user_id=user_id)
            log_state(StepBackLog.EXITING_STEP_BACK, function="step_back_function", user_id=user_id)
            log_state(ServiceLog.EXITING_AI_SERVICE, function="step_back_function", user_id=user_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.RAW_REPAIR_FAILURE.value,
                error_message="Structured step-back parsing failed and manual repair returned no result.",
            )

        log_state(RepairLog.AI_REPAIR_SUCCESS, function="step_back_function", user_id=user_id)
        step_back_query = recovered.step_back_question

    try:        
        log_state(StepBackLog.STEP_BACK_RETRIEVAL_STARTED, function="step_back_function", user_id=user_id)
        retrieval_tasks: list[Awaitable[list[LangChainDocument]]] = [
            safe_retrieve(retriever, step_back_query),
            safe_retrieve(retriever, question),
        ]
        parallel_results: list[list[LangChainDocument]] = await asyncio.gather(*retrieval_tasks)
    except Exception as exc:
        log_state(ExceptionLog.NO_RELATED_VECTOR_DATABASE_FOUND, function="step_back_function", user_id=user_id, exc=str(exc))
        log_state(ServiceLog.AI_SERVICE_FAILED, function="step_back_function", user_id=user_id)
        log_state(StepBackLog.STEP_BACK_RETRIEVAL_FAILED, function="step_back_function", user_id=user_id)
        log_state(StepBackLog.EXITING_STEP_BACK, function="step_back_function", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="step_back_function", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.INTERNAL_SYSTEM_ERROR.value,
            error_message=f"Step-back dual retrieval failed: {str(exc)}",
        )

    # 4. RRF Merging & Deduplication
    fused_documents = _reciprocal_rank_fusion(user_id=user_id, results_per_query=parallel_results)
    if fused_documents is None:
        log_state(ServiceLog.AI_SERVICE_FAILED, function="step_back_function", user_id=user_id)
        log_state(StepBackLog.STEP_BACK_RETRIEVAL_FAILED, function="step_back_function", user_id=user_id)
        log_state(StepBackLog.EXITING_STEP_BACK, function="step_back_function", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="step_back_function", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.RRF_FAILURE.value,
            error_message="RRF unexpectedly failed during step-back document fusion",
        )

    final_documents = fused_documents[:top_n_final]

    # 5. Empty check and return
    if not final_documents:
        log_state(ExceptionLog.NO_RELATED_DOCUMENT_FOUND, function="step_back_function", user_id=user_id)
        log_state(ServiceLog.AI_SERVICE_FAILED, function="step_back_function", user_id=user_id)
        log_state(StepBackLog.STEP_BACK_RETRIEVAL_FAILED, function="step_back_function", user_id=user_id)
        log_state(StepBackLog.EXITING_STEP_BACK, function="step_back_function", user_id=user_id)
        log_state(ServiceLog.EXITING_AI_SERVICE, function="step_back_function", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=USER_ERROR_CODES.NO_RELATED_DOCUMENT_FOUND.value,
            error_message="No relevant document chunks retrieved via step-back prompting path.",
        )

    log_state(StepBackLog.STEP_BACK_RETRIEVAL_SUCCESS, function="step_back_function", user_id=user_id)
    log_state(StepBackLog.STEP_BACK_SUCCESS, function="step_back_function", user_id=user_id)
    log_state(ServiceLog.AI_SERVICE_COMPLETED, function="step_back_function", user_id=user_id)
    log_state(StepBackLog.EXITING_STEP_BACK, function="step_back_function", user_id=user_id)
    log_state(ServiceLog.EXITING_AI_SERVICE, function="step_back_function", user_id=user_id)

    return APIResponse(success=True, data=final_documents, error_code=None, error_message=None)