from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse



from core.Exceptions.exceptions import AppException
from core.Exceptions.error_registery import ERROR_STATUS_MAP
from utils.schemas import APIResponse


from utils.schemas import LogContext
from utils.logging.logger import (
    log_exception
)

from utils.logging.logEvents import ExceptionLog
from utils.logging.helper_log import log_state, LogState



async def global_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    
    current_log_event: ExceptionLog = getattr(exc, "log_event", ExceptionLog.APP_EXCEPTION) 
    log_state(current_log_event, level=LogState.EXCEPTION, function="global_exception_handler", route=str(request.url.path), exc=exc)
    
    response = APIResponse(
        success=False,
        data=None,
        error_code=exc.error_code, 
        error_message=exc.message
    )

    return JSONResponse(
        status_code=ERROR_STATUS_MAP.get( 
            exc.error_code, 
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        content=jsonable_encoder(response) 
    )


async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log_state(ExceptionLog.UNHANDLED_EXCEPTION, level=LogState.EXCEPTION, function="unexpected_exception_handler", route=str(request.url.path), exc=exc)

    response = APIResponse(
        success=False,
        data=None,
        error_code="SYSTEM_ERROR", 
        error_message="Internal server error"
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(response)
    )