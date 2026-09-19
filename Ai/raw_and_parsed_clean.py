from typing import Any, TypeVar, Type
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError
from pydantic import BaseModel
from Ai.retry_logic import safe_parse



T = TypeVar("T", bound=BaseModel)
def extract_parsed_data(parsed: Any, schema: Type[T]) -> T | None:     
    if parsed is not None:
        if isinstance(parsed, schema):
            return parsed
        if isinstance(parsed, dict):
            try:
                return schema.model_validate(parsed)
            except ValidationError:    

                return None 
    return None 



async def extract_raw_data(raw: Any, parser: PydanticOutputParser, model: Any, text: str, schema: type[T]) -> T | None:
    if hasattr(raw, "tool_calls") and raw.tool_calls:
        
        for call in raw.tool_calls: 
            
            function_data = getattr(call, "function", None)
            
            tool_args = (
                getattr(call, "args", None)
                or (
                    function_data.get("arguments")
                    if isinstance(function_data, dict)
                    else None
                )
                or (
                    call.get("args", {})
                    if isinstance(call, dict)
                    else None
                )
            )

            if isinstance(tool_args, dict) and tool_args:
                try:
                    return schema.model_validate(tool_args)

                except ValidationError:
                    continue

            if isinstance(tool_args, str):
                parsed: T | None = await safe_parse(tool_args, parser, model, text)
                if parsed:
                    return parsed
                else: continue
    if hasattr(raw, "content") and raw.content:
        return await safe_parse(raw.content.strip(), parser, model, text)
        
    if isinstance(raw, str) and raw.strip():
        return await safe_parse(raw, parser, model, text)
    
    return None