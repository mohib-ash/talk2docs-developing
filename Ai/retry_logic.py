from langchain_core.exceptions import OutputParserException 
from typing import List





def is_provider_limit_error(exception: Exception) -> bool:

    err_msg = str(exception).lower()
    limit_keywords = [
        "insufficient_quota", 
        "rate_limit_exceeded", 
        "token_limit_exceeded", 
        "quota exceeded", 
        "429", 
        "out of tokens"
    ]
    return any(keyword in err_msg for keyword in limit_keywords)


def check_provider_quota(exception: Exception) -> bool:
    """
    Returns True if AI provider quota/rate limit issue happened.
    """
    return is_provider_limit_error(exception)




def clean_json_string(raw_str):
    start_idx = raw_str.find('{')
    end_idx = raw_str.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        return raw_str # Fallback
    return raw_str[start_idx:end_idx + 1]






async def safe_parse(raw_output, parser, llm, query, max_retries=3) -> object | None:
    output = raw_output 

    for attempt in range(max_retries):
        current_str = output.content if hasattr(output, "content") else output    
        current_str = str(current_str).strip()


        # Remove markdown junk
        if "```" in current_str:
            parts: List[str] = current_str.split("```")
            current_str = max(parts, key=len).strip()
            
            
            if current_str.lower().startswith("json"):
                current_str = current_str[4:].strip() 
                
            current_str = clean_json_string(current_str)



        try:
            return parser.parse(current_str) 
        except OutputParserException as e: 
            if attempt == max_retries - 1:
                return None

            repair_prompt = f"""
            You MUST return valid JSON only.
            NO text.
            NO explanation.
            NO markdown blocks.

            Target Format Instructions:
            {parser.get_format_instructions()}

            Context/Task Being Evaluated:
            {query}

            Broken Target String:
            {current_str}

            Parser Parsing Error Encountered:
            {e}

            Fix the JSON formatting error above and return a fully compliant object matching the target format instructions exactly.
            """
            try:
                output = await llm.ainvoke(repair_prompt)

            except Exception as e:
                if check_provider_quota(e):
                    return None
    return None

