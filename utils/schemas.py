import enum

from pydantic import BaseModel, ConfigDict, Field, EmailStr, StringConstraints
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, Literal, List, Union

class UserRegisterSchema(BaseModel):
    email: EmailStr
    password: str

class UserLoginSchema(BaseModel): 
    email: EmailStr
    password: str



class CommentCreateSchema(BaseModel):
    text: str = Field(..., max_length=1000, description="The message content of the comment")


class UserResponseSchema(BaseModel):
    email: EmailStr
    user_id: int 
    created_at: datetime
    model_config = {"from_attributes": True}




class TokenSchema(BaseModel): 
    access_token: str
    token_type: str

class TokenDataSchema(BaseModel): 
    user_id: int = None
    jid: str = None


class ChangePasswordInputSchema(BaseModel):
    old_pass: str
    new_pass: str



#used in ai_route
class QuotaStatus(enum.Enum):
    ALLOWED = "ALLOWED" 
    EXHAUSTED = "EXHAUSTED" 
    COLLISION = "COLLISION" 

#AI
#Rephrase-route
class RephraseRequest_route(BaseModel):
    text: Annotated[str, StringConstraints(max_length=3000, strip_whitespace=True)]
    tone: Literal[
        "rephrase",
        "professional",
        "casual",
        "executive",
        "simplified",
        "legal"
    ]

class RephraseOutput_route(BaseModel):
    text: str
    confidence: float
    stylistic_explanation: Annotated[str, StringConstraints(max_length=500, strip_whitespace=True)]
    is_meaning_preserved: bool
    
    model_config = {"from_attributes": True} 





#summary-route:
class SummaryRequest_route(BaseModel):
    text: Annotated[str, StringConstraints(max_length=3000, strip_whitespace=True)]

class SummaryOut_route(BaseModel):
    text: str
    topic: str
    confidence_score: float
    stylistic_explanation: Annotated[str, StringConstraints(max_length=300, strip_whitespace=True)]
    is_meaning_preserved: bool
    model_config = {"from_attributes": True}



#sentiment-route:
class SentimentAnalysisRequest_route(BaseModel):
    text: Annotated[str, StringConstraints(max_length=3000, strip_whitespace=True)]

class SentimentAnalysisOut_route(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed", "casual"]
    confidence_score: float
    explanation: str 
    model_config = {"from_attributes": True}



#title_gem
class Title_genRequest_Route(BaseModel):
    text: Annotated[str, StringConstraints(max_length=3000, strip_whitespace=True)]

class Title_genOut_Route(BaseModel):
    main_title: Union[str, None] 
    variations: List[str] 
    minor_summary: Annotated[str, StringConstraints(max_length=500, strip_whitespace=True)]




#API
#overall server responce
class APIResponse(BaseModel):
    success: bool
    data: Any | None = None #basically the pydentic will be inside it!
    error_code: str | None = None
    error_message: str | None = None




#gateways:
#ai gatways:
class AIGatewayContext(BaseModel):
    user_id: int
    request_id: str




#logging:
#logging schema
from utils.logging.logEvents import BaseLogEvent
class LogContext(BaseModel):
    event: BaseLogEvent 
    
    # Correlation
    request_id: str | None = None
    user_id: int | None = None

    # Location
    route: str | None = None
    function: str | None = None

    # AI
    provider: str = Field(default="groq")
    model: str = Field(
        default="Llama-3.3-70B-Versatile"
    )

    # Performance
    latency_ms: int | None = None

    # Recovery
    repair_used: bool | None = Field(default=False)

    # Errors
    exception: str | None = None
    exception_type: str | None = None 

#helper classes:
class AIRequestState(str, enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    





#upldaing file: (for validation)
class UploadTaskPayload(BaseModel):
    request_id: str
    user_id: int

    original_filename: str = Field(description=("this is the name of file user sent"))
    stored_filename: str = Field(description=("this is the name of file we will uuid for our use"))
    file_dir: str = Field(description=("the folder name in which file will be stored! so later in same folder we can store markdown!"))

    file_hash: str
    file_extension: str
    file_content_type: str
    file_path: str
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )



class passed_vlidation_reponce(BaseModel):
    file_payload: UploadTaskPayload
    file_bytes: bytes 


#a middle man class b/w above one and bellow one!
class DataToFrontEndAfterUploadingRoute(BaseModel):
    request_id: str
    user_id: int


#After celery:
class All_worker_starter_responce(BaseModel):
    task_id: str
    status: str = "queued"  
    doc_upload_api_responce: DataToFrontEndAfterUploadingRoute




#uploaded doc status:
class DocumentStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"       # saved to disk
    PARSED = "PARSED"           # docling finished
    CHUNKED = "CHUNKED"
    EMBEDDED = "EMBEDDED"
    INDEXED = "INDEXED"
    READY = "READY"
    FAILED = "FAILED"
    

#metadata of saved doc esstially input for worker2
class SavedDocumentPayload(BaseModel):
    doc_id: int
    request_id: str
    user_id: int
    original_filename: str
    stored_filename: str
    file_extension: str
    file_size: int
    file_dir: str 
    file_path: str
    mime_type: str
    collection_name: str
    status: DocumentStatus

    model_config = ConfigDict(from_attributes=True)


#input for worker3 and return of worker2
class ParsedDocumentPayload(BaseModel):
    doc_id: int
    request_id: str 
    markdown_path: str
    
    
class UploadTask2_fail_cases(str, enum.Enum):
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBIDING = "EMBIDING"
    



#quertion route:
class QuestionRequest(BaseModel):
    question: str
    doc_name: list[str] | None = None 



class MultiIndexStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class LogState(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    EXCEPTION = "exception"
    ERROR = "error"

class BM25Status(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING" 
    READY = "READY"
    FAILED = "FAILED"
    STALE = "STALE"

class CacheVDBStatus(enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"