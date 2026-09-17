import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from Ai.retry_logic import check_provider_quota
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES
from utils.logging.helper_log import log_state
from utils.logging.logEvents import ProviderLog, ServiceLog
from utils.schemas import APIResponse, LogState
from Ai.main import model


class QuestionClassificationSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    responce: Literal["CACHEABLE", "NON_CACHEABLE"] = Field(
        ...,
        description=(
            "CACHEABLE if the question can be answered from stable, "
            "non-time-sensitive documentation or reference material. "
            "NON_CACHEABLE if the answer may depend on current state, "
            "real-time data, exact dates or times, dynamic user state, "
            "numerical calculations, tabular/spreadsheet data, or other "
            "information that may change between requests."
        ),
    )


system_classification_template = """You are a strict enterprise cache-admission classifier for an advanced RAG system.

Your ONLY task is to determine whether the user's question is safe to participate in semantic answer caching.

================ CACHEABILITY RULES ================

CACHEABLE:
- Static documentation and reference material.
- Stable architectural concepts.
- General programming concepts and coding principles.
- Explanations of how a documented feature or system works.
- Historical or otherwise time-independent reference information.
- Questions whose correct answer should remain stable unless the underlying documentation changes.

NON_CACHEABLE:
- Real-time or current information.
- Live database state, records, counters, or metrics.
- Current system/application/user state.
- Questions involving "now", "today", "latest", "currently", "recent", etc.
- Exact dates, times, schedules, or time-dependent values.
- Numerical calculations or questions requiring computation from provided data.
- Tables, spreadsheets, Excel, CSV, or other tabular-data analysis.
- Questions whose answer depends on dynamic values or changing external state.
- Personalized answers that depend on the current user's state or context.
- Any question where reusing a previous answer could produce an incorrect result because relevant information may have changed.

================ DECISION PRINCIPLE ================

When uncertain, choose NON_CACHEABLE.

The goal is correctness, not maximum cache utilization.

Do not answer the question.
Do not explain your decision.
Do not output anything outside the requested schema.

{format_instructions}
"""


async def get_classification(question: str, user_id: int) -> APIResponse:
    log_state(ServiceLog.AI_SERVICE_STARTED, function="get_classification", user_id=user_id)

    if not question or not question.strip():
        log_state(ServiceLog.AI_SERVICE_FAILED, function="get_classification", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
            error_message="Question string cannot be empty for classification."
        )

    parser = PydanticOutputParser(pydantic_object=QuestionClassificationSchema)

    full_prompt = ChatPromptTemplate.from_messages([
        ("system", system_classification_template),
        ("human", "Classify this question:\n\n<question>\n{question}\n</question>")
    ]).partial(format_instructions=parser.get_format_instructions())

    extracted_parsed = None

    try:
        log_state(ProviderLog.AI_PROVIDER_REQUEST, function="get_classification", user_id=user_id)
        log_state(ProviderLog.AI_PROVIDER_IN_PROCESSING, function="get_classification", user_id=user_id)

        raw_response = await (full_prompt | model).ainvoke({"question": question.strip()})
        cleaned_content = raw_response.content.strip()
        
        if cleaned_content.startswith("```"):
            cleaned_content = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned_content)
            cleaned_content = re.sub(r"\n?```$", "", cleaned_content).strip()

        extracted_parsed = parser.parse(cleaned_content)

    except Exception as e:
        if check_provider_quota(e):
            log_state(ProviderLog.AI_PROVIDER_FAILURE, level=LogState.EXCEPTION if "LogState" in globals() else "ERROR", function="get_classification", user_id=user_id, exc=e)
            log_state(ServiceLog.AI_SERVICE_FAILED, function="get_classification", user_id=user_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value,
                error_message="Quota reached during classification."
            )
        else:
            log_state(ProviderLog.AI_PROVIDER_FAILURE, function="get_classification", user_id=user_id, exc=e)
            log_state(ServiceLog.AI_SERVICE_FAILED, function="get_classification", user_id=user_id)
            
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
                error_message=str(e)
            )

    log_state(ProviderLog.AI_PROVIDER_SUCCESS, function="get_classification", user_id=user_id)

    if extracted_parsed:
        log_state(ServiceLog.AI_SERVICE_COMPLETED, function="get_classification", user_id=user_id)
        return APIResponse(
            success=True,
            data=extracted_parsed,
            error_code=None,
            error_message=None
        )

    # Fallback if parsing returned empty
    log_state(ServiceLog.AI_SERVICE_FAILED, function="get_classification", user_id=user_id)
    return APIResponse(
        success=False,
        data=None,
        error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
        error_message="Classification parsing yielded no result."
    )