from Ai.convo_classifier import classify_conversation_intent
from utils.schemas import APIResponse


async def check_new_question_and_brach(question: str, user_id: int) -> APIResponse:
    convo_intent = "next_question" #coz if convo_intent == "continue_chat": is outa if covo.success and as such if covo fails convo_intent will throw error thus we have defult val
    convo: APIResponse = await classify_conversation_intent(text=question, user_id=user_id)
    if convo.success and convo.data:
        convo_intent = convo.data.intent.strip().lower()
    
    #early out!
    if convo_intent == "next_question":
        return APIResponse(
            data="next_question",
            success=True,
            error_code=None,
            error_message=None
        )
    
    
    if convo_intent == "continue_chat":
        return APIResponse(
            data="continue_chat",
            success=True,
            error_code=None,
            error_message=None
        ) #temp return so system doesnt shortcircut but make ai call here!
        
    #TODO here will be Ai responce later to make!
        #here we will recive APIRresponce form chat based ai which we will either send to global exception handler or send upstream with success=True
        #also gotta find a way or decide if things talked in covo should be cached or nah? if so how!
    else:
        pass    
