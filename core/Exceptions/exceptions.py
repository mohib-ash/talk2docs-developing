from utils.logging.logEvents import ExceptionLog


class AppException(Exception):
    """Base exception for application-level errors."""
    log_event: ExceptionLog = ExceptionLog.APP_EXCEPTION

    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(message)



class UploadingException(AppException):
    """
    If uploading fails
    """
    log_event: ExceptionLog = ExceptionLog.FILE_UPLADONG_EXCEPTION
    
class DocumentNotFoundException(AppException):
    """
    In 2nd task when tokanization process has been satrted and doc is not there
    """
    log_event: ExceptionLog = ExceptionLog.MISSING_DOCUMENT_EXCEPTION
    
    


class SavingValidatedFileException(AppException):
    """
    if validated file saving to dir failed
    """
    log_event: ExceptionLog = ExceptionLog.VALIDATED_FILE_SAVING_EXCEPTION
    


class InvalidTask2PayloadException(AppException):
    """
    if saving validated file task didnt give task of tokenzation proper input obj->dict 
    """
    log_event: ExceptionLog = ExceptionLog.INVALID_TASK2_PAYLOAD_EXCEPTION
    

class InvalidTask1PayloadException(AppException):
    """
    if task_1 didnt get correct input obj->dict
    """
    log_event: ExceptionLog = ExceptionLog.INVALID_TASK1_PAYLOAD_EXCEPTION
    

    
class TokenizationWorkerStarterException(AppException):
    """
    if worker 2 the parser,chunker,embedder startr fails
    """
    log_event: ExceptionLog = ExceptionLog.SAVING_VALIDATED_FILE_EXCEPTION

class ParsingSavedFileException(AppException):
    """
    if parsing fails, converting document into docling obj
    """
    log_event: ExceptionLog = ExceptionLog.PARSING_SAVED_FILE_EXCEPTION


class ChunkingParsedFileException(AppException):
    """
    if after pasing chunking fails
    """
    log_event: ExceptionLog = ExceptionLog.CHUNKING_PARSE_DOC_EXCEPTION
    

class EmbeddingChunkedFileException(AppException):
    """
    if Embedding fails and nothing is in vector db
    """
    log_event: ExceptionLog = ExceptionLog.EMBEDDING_CHUNKS_EXCEPTION
    

#later on this when we expand
class AIServiceException(AppException):
    """
    AI related failures.
    """
    
    log_event: ExceptionLog = ExceptionLog.AI_SERVICE_EXCEPTION




class PostServiceException(AppException):
    """
    Posting realted faliures
    """
    
    log_event: ExceptionLog = ExceptionLog.POSTING_SERVICE_EXCEPTION



class ChangePasswordException(AppException):
    """
    if somehow password wasnt changed
    """
    
    log_event: ExceptionLog = ExceptionLog.PASSWORD_CHANGE_EXCEPTION


class LoginServiceException(AppException):
    """
    Login and jwt creation faluires
    """
    
    log_event: ExceptionLog = ExceptionLog.LOGIN_EXCEPTION



class LogoutServiceException(AppException):
    """
    Logout issue, if someone logsout when they're logged out
    """
    
    log_event: ExceptionLog = ExceptionLog.LOGOUT_EXCEPTION



class LogoutAllDeviServiceException(AppException):
    """
    Logout form all devices issue
    """
    
    log_event: ExceptionLog = ExceptionLog.LOGOUT_ALL_DEVICES_EXCEPTION



class AllServiceContException(AppException):
    """
    when user wanna know how many active sessins -> perhspas in app settings this feature will be good
    """
    
    log_event: ExceptionLog = ExceptionLog.ACTIVE_SESSION_COUNT_EXCEPTION



class RvokeCurrentSessionException(AppException):
    """
    use when user is hacked and he appils, revoke his 1 device which is hacked's session, rest device no, keep em
    """
    
    log_event: ExceptionLog = ExceptionLog.REVOKATION_CURRENT_SESSION_EXCEPTION




class LikeServiceException(AppException):
    """
    Liking falure
    """
    
    log_event: ExceptionLog = ExceptionLog.LIKE_EXCEPTION



class UserCreationServiceException(AppException):
    log_event: ExceptionLog = ExceptionLog.USER_CREATION_EXCEPTION
    

class NoVectorDatabaseException(AppException):
    """
    user doesnt have a vector database yet
    """
    log_event: ExceptionLog = ExceptionLog.MISSING_VECTOR_DATABASE_EXCEPTION