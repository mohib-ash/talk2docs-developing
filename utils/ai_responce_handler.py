from utils.schemas import APIResponse
from core.Exceptions.exceptions import AppException
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES


def is_system_failure(result: APIResponse) -> bool:
    return (
        not result.success
        and isinstance(
            result.error_code, 
            SYSTEM_ERROR_CODES
        ) 
    )


def handle_service_response(
    result: APIResponse,
    exception_cls: type[AppException] = AppException
):
    if result.success:
        return result.data
    
    raise exception_cls(
        error_code=result.error_code,
        message=result.error_message
    )
