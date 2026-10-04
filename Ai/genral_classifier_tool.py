from typing import Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from utils.APIResponce_error_code_enum import USER_ERROR_CODES, SYSTEM_ERROR_CODES
from Ai.retry_logic import check_provider_quota
from Ai.raw_and_parsed_clean import extract_raw_data, extract_parsed_data
from core.Exceptions.exceptions import AIServiceException
from utils.schemas import APIResponse
from utils.logging.logEvents import ProviderLog, RepairLog, ServiceLog
from utils.logging.logger import log_exception, log_info, log_warning
from utils.logging.helper_log import log_state, LogState

class GeneralQueryUnderstanding(BaseModel):
    intent_header: str = Field(..., description="The core topic, technical domain, or category of what the user is asking.")
    summary: str = Field(..., description="A sharp, single-line summary explaining exactly what the user wants to achieve.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the analysis.")

async def get_general_query_intent(model, text: str, user_id: int, context: str = "") -> APIResponse:
    log_state(ServiceLog.AI_SERVICE_STARTED, function="get_general_query_intent", user_id=user_id)
        
    if not text or not text.strip():
        log_state(ServiceLog.AI_SERVICE_FAILED, function="get_general_query_intent", user_id=user_id)
        return "Classification error: Input text cannot be empty."

    parser = PydanticOutputParser(pydantic_object=GeneralQueryUnderstanding)
    examples = [
        {
            "input": "How do I setup a Redis message broker with Celery workers inside WSL2 Ubuntu without breaking port forwarding?",
            "output": '{"intent_header": "Backend Infrastructure & DevOps", "summary": "User wants step-by-step instructions for configuring Celery with a Redis broker inside WSL2.", "confidence": 0.98}'
        },
        {
            "input": "Can we refactor this C++ template class so the dynamic memory pointer doesn't throw a segmentation fault on destruction?",
            "output": '{"intent_header": "C++ Systems Programming", "summary": "User needs help debugging a segmentation fault caused by custom dynamic memory handling in a template class.", "confidence": 0.99}'
        },
        {
            "input": "What's the difference between BM25 sparse keyword search and dense vector embeddings in a hybrid RAG pipeline?",
            "output": '{"intent_header": "Machine Learning & RAG Architecture", "summary": "User is asking for a technical comparison between BM25 sparse search and vector embeddings.", "confidence": 0.97}'
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
    
    system_prompt = """You are an intent disambiguation and query analysis tool for an AI gateway.
        When the main model is uncertain about an ambiguous or complex user query, analyze it alongside the conversation context to extract:
        1. intent_header: A brief categorization of the topic or domain.
        2. summary: A precise 1-liner summary of what the user is trying to accomplish.
        3. confidence: A float score between 0.0 and 1.0.

        {format_instructions}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "Conversation Context:\n{context}\n\nAnalyze the ambiguous query:\n\n{query}\n")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    try:
        log_state(ProviderLog.AI_PROVIDER_REQUEST, function="get_general_query_intent", user_id=user_id)
        chain = prompt | model
        response = await chain.ainvoke({
            "query": text, 
            "context": context if context and context.strip() else "No prior context provided."
        })
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        if check_provider_quota(e):
            log_state(ServiceLog.AI_MY_QUOTA_REACHED, level=LogState.EXCEPTION, exc=e, function="get_general_query_intent", user_id=user_id)
            return "Ai Quota reached.."
        else:
            log_state(ProviderLog.AI_PROVIDER_FAILURE, level=LogState.EXCEPTION, exc=e, function="get_general_query_intent", user_id=user_id)
            return "AI processing failed due to unknown system error."

    log_state(ProviderLog.AI_PROVIDER_SUCCESS, function="get_general_query_intent", user_id=user_id)
        
    try:
        extracted_parsed = await parser.aparse(raw)
        if extracted_parsed is not None:
            log_state(ServiceLog.AI_SERVICE_COMPLETED, function="get_general_query_intent", user_id=user_id)
            return APIResponse(success=True, data=extracted_parsed, error_code=None, error_message=None)
    except Exception:
        pass
   
    # Fallback / Raw repair step using your existing manual recovery helper
    if not raw:
        log_state(ServiceLog.AI_SERVICE_FAILED, function="get_general_query_intent", user_id=user_id)
        return "Classifier's structured output parsing failed"
    
    try:    
        recovered: GeneralQueryUnderstanding | None = await extract_raw_data(raw, parser, model, text, GeneralQueryUnderstanding)
    except Exception as e:
        log_state(RepairLog.AI_REPAIR_PREMATURELY_ENDED, level=LogState.EXCEPTION, exc=e, function="get_general_query_intent", user_id=user_id)
        return "Classifier's faulty output recovery failed"
        
    if recovered is None:
        return "Classifier's parsing and manual recovery both failed"
    
    log_state(RepairLog.AI_REPAIR_SUCCESS, function="get_general_query_intent", user_id=user_id)
    return APIResponse(success=True, data=recovered, error_code=None, error_message=None)