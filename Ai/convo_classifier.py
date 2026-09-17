import re
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from Ai.retry_logic import check_provider_quota
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES
from utils.logging.helper_log import log_state
from utils.logging.logEvents import ProviderLog, ServiceLog
from utils.schemas import APIResponse, LogState
from Ai.main import model


AvailableIntents = Literal["next_question", "continue_chat"]


class ConversationIntentSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    intent: AvailableIntents = Field(
        ..., 
        description="Classify whether the user wants to ask a completely new question or continue chatting about the current topic."
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_intent(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean in ["next_question", "continue_chat"]:
                return v_clean
        return "next_question"


system_intent_template = """
You are a precise conversation flow router for an AI assistant specialized in UAE perfumes.
Classify the incoming user query into EXACTLY ONE of two categories:
- "continue_chat": The user is following up, asking for clarification, or continuing the discussion on the exact same perfume/topic currently active.
- "next_question": The user is shifting to a completely new perfume, brand, or an unrelated query, meaning past context is no longer needed.

Do not output anything outside the requested schema.

{format_instructions}
"""


async def classify_conversation_intent(text: str, user_id: int) -> APIResponse:
    """
    Parser-based conversation flow classifier powered by few-shot examples 
    to accurately determine if we need past context history.
    """
    log_state(ServiceLog.AI_SERVICE_STARTED, function="classify_conversation_intent", user_id=user_id)

    if not text or not text.strip():
        log_state(ServiceLog.AI_SERVICE_FAILED, function="classify_conversation_intent", user_id=user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
            error_message="Query string cannot be empty for intent classification."
        )

    parser = PydanticOutputParser(pydantic_object=ConversationIntentSchema)

    # Define concrete few-shot examples for technical documentation RAG
    examples = [
        {
            "input": "User input:\nHow do I configure the connection timeout for this setting?",
            "output": "{\n    \"intent\": \"continue_chat\",\n    \"confidence_score\": 0.98\n}"
        },
        {
            "input": "User input:\nCan you show me an example of how to implement that middleware?",
            "output": "{\n    \"intent\": \"continue_chat\",\n    \"confidence_score\": 0.99\n}"
        },
        {
            "input": "User input:\nWhat is the pricing model for the enterprise tier?",
            "output": "{\n    \"intent\": \"next_question\",\n    \"confidence_score\": 0.97\n}"
        },
        {
            "input": "User input:\nHow do I set up Docker Compose for local development?",
            "output": "{\n    \"intent\": \"next_question\",\n    \"confidence_score\": 0.96\n}"
        }
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_intent_template),
        few_shot_prompt,
        ("human", "User input:\n{query}")
    ]).partial(format_instructions=parser.get_format_instructions())

    extracted_parsed = None

    try:
        log_state(ProviderLog.AI_PROVIDER_REQUEST, function="classify_conversation_intent", user_id=user_id)
        log_state(ProviderLog.AI_PROVIDER_IN_PROCESSING, function="classify_conversation_intent", user_id=user_id)

        raw_response = await (prompt | model).ainvoke({"query": text.strip()})
        cleaned_content = raw_response.content.strip()

        if cleaned_content.startswith("```"):
            cleaned_content = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned_content)
            cleaned_content = re.sub(r"\n?```$", "", cleaned_content).strip()

        extracted_parsed = parser.parse(cleaned_content)

    except Exception as e:
        if check_provider_quota(e):
            log_state(ProviderLog.AI_PROVIDER_FAILURE, level=LogState.EXCEPTION if "LogState" in globals() else "ERROR", function="classify_conversation_intent", user_id=user_id, exc=e)
            log_state(ServiceLog.AI_SERVICE_FAILED, function="classify_conversation_intent", user_id=user_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.MY_QUOTA_REACHED.value,
                error_message="Quota reached during intent classification."
            )
        else:
            log_state(ProviderLog.AI_PROVIDER_FAILURE, function="classify_conversation_intent", user_id=user_id, exc=e)
            log_state(ServiceLog.AI_SERVICE_FAILED, function="classify_conversation_intent", user_id=user_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
                error_message=str(e)
            )

    log_state(ProviderLog.AI_PROVIDER_SUCCESS, function="classify_conversation_intent", user_id=user_id)

    if extracted_parsed:
        log_state(ServiceLog.AI_SERVICE_COMPLETED, function="classify_conversation_intent", user_id=user_id)
        return APIResponse(
            success=True,
            data=extracted_parsed,
            error_code=None,
            error_message=None
        )

    log_state(ServiceLog.AI_SERVICE_FAILED, function="classify_conversation_intent", user_id=user_id)
    return APIResponse(
        success=False,
        data=None,
        error_code=SYSTEM_ERROR_CODES.AI_SERVICE_FAILURE.value,
        error_message="Intent parsing yielded no result."
    )