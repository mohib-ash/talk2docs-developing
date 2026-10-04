import re
from typing import Any, Literal, List, Optional
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
You are a precise conversation flow router for an AI assistant.
Classify the incoming user query into EXACTLY ONE of two categories based on the provided conversation history (if any):
- "continue_chat": The latest question depends on, refers to, follows up on, clarifies, or extends information from the previous conversation (e.g., using implicit context, pronouns, or asking follow-up mechanics).
- "next_question": The latest question can be understood independently and does not meaningfully depend on the preceding conversation (even if it's within the same broader technical domain).

Do not output anything outside the requested schema.

{format_instructions}
"""

# Few-shot examples
examples = [
        {
            "input": "Conversation History:\nNo prior conversation history available.\n\nUser input:\nHow do I configure the connection timeout for this setting?",
            "output": "{\n    \"intent\": \"next_question\",\n    \"confidence_score\": 0.95\n}"
        },
        {
            "input": "Conversation History:\nRecent Short-Term Cache History:\n- Q: How does Redis persistence work?\n  A: Redis uses RDB snapshots and AOF logs to persist data to disk.\n\nUser input:\nAnd what happens if the process crashes before the snapshot?",
            "output": "{\n    \"intent\": \"continue_chat\",\n    \"confidence_score\": 0.99\n}"
        },
        {
            "input": "Conversation History:\nRecent Short-Term Cache History:\n- Q: Explain Redis persistence.\n  A: Redis provides RDB and AOF persistence mechanisms.\n\nUser input:\nOkay, what about Celery?",
            "output": "{\n    \"intent\": \"next_question\",\n    \"confidence_score\": 0.94\n}"
        }
    ]
parser = PydanticOutputParser(pydantic_object=ConversationIntentSchema)

async def classify_conversation_intent(
    text: str, 
    user_id: int, 
    last_LTM: Optional[list[tuple[str, str]]] = None, 
    last_cache: Optional[list[tuple[str, str]]] = None
) -> APIResponse:
    """
    Parser-based conversation flow classifier that dynamically incorporates 
    Short-Term Cache and Long-Term Memory (LTM) histories to evaluate intent.
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


    # Build context blocks dynamically based on what history is available
    context_blocks = []

    if last_cache:
        cache_str = "\n".join([f"- Q: {q}\n  A: {a}" for q, a in last_cache if q and a])
        if cache_str:
            context_blocks.append(f"Recent Short-Term Cache History:\n{cache_str}")
        

    if last_LTM:
        ltm_str = "\n".join([f"- Q: {q}\n  A: {a}" for q, a in last_LTM if q and a])
        if ltm_str:
            context_blocks.append(f"Long-Term Memory History:\n{ltm_str}")

    history_section: str = "\n\n".join(context_blocks) if context_blocks else "No prior conversation history available."   
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
        ("human", "Conversation History:\n{history}\n\nUser input:\n{query}")
    ]).partial(format_instructions=parser.get_format_instructions())

    extracted_parsed = None

    try:
        log_state(ProviderLog.AI_PROVIDER_REQUEST, function="classify_conversation_intent", user_id=user_id)
        log_state(ProviderLog.AI_PROVIDER_IN_PROCESSING, function="classify_conversation_intent", user_id=user_id)

        raw_response = await (prompt | model).ainvoke({
            "history": history_section,
            "query": text.strip()
        })
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