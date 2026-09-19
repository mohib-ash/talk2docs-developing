from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession 
from db_tables.tables import User
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES, USER_ERROR_CODES
from utils.hashing import verify_hashed_password
from Oauth2 import create_access_token
from utils.logging.helper_log import LogState, log_state
from utils.logging.logEvents import AuthLogs
from utils.schemas import APIResponse, TokenDataSchema, TokenSchema
import uuid
import json
from Oauth2 import ACCESS_TOKEN_EXPIRE_MINUTES
from redis.asyncio import Redis


async def cache_jwt_data(redis: Redis, jid: str, user_id: int) -> None:
    log_state(AuthLogs.CACHING_JWT_SESSION, function="cache_jwt_data", user_id=user_id)
    data = {"user_id": user_id}
    
    await redis.set(f"session:{jid}", json.dumps(data), ex=ACCESS_TOKEN_EXPIRE_MINUTES*60) 
    await redis.sadd(f"user_sessions:{user_id}", jid) 
    
    log_state(AuthLogs.SUCCESS, function="cache_jwt_data", user_id=user_id)



async def login_user_service(user_credentials: OAuth2PasswordRequestForm, db: AsyncSession, redis: Redis) -> APIResponse:
    log_state(AuthLogs.AUTH_SERVICE_STARTED, function="login_user_service")
    log_state(AuthLogs.AUTHENTICATING_USER, function="login_user_service")
    

    try:
        log_state(AuthLogs.EXECUTING_DATABASE_QUERY, function="login_user_service")
        result = await db.execute(
            select(User).where(User.email == user_credentials.username)
        )
        fetched_user = result.scalar_one_or_none()

        if not fetched_user:
            log_state(AuthLogs.OPERATION_FAILED, function="login_user_service", level=LogState.WARNING)
            # log_state(AuthLogs.EXITING_AUTH_SERVICE, function="login_user_service")

            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.RESOURCE_NOT_FOUND.value,
                error_message="User not found."
            )

        log_state(AuthLogs.VALIDATING_REQUEST, function="login_user_service", user_id=fetched_user.user_id)

        if not verify_hashed_password(user_credentials.password, fetched_user.password):
            log_state(AuthLogs.OPERATION_FAILED, function="login_user_service", user_id=fetched_user.user_id, level=LogState.WARNING)
            # log_state(AuthLogs.EXITING_AUTH_SERVICE, function="login_user_service")
            
            
            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.UNAUTHORIZED_ACCESS.value,
                error_message="Invalid email or password."
            )
        
        log_state(AuthLogs.CONNECTING_TO_REDIS, function="login_user_service")
        log_state(AuthLogs.CREATING_SESSION, function="login_user_service")
        jid = str(uuid.uuid4())
        await cache_jwt_data(redis, jid, fetched_user.user_id)
        
        access_token = create_access_token(
            data={
                "user_id": fetched_user.user_id,
                "jid": jid
                }
        )
        
        log_state(AuthLogs.SESSION_CREATED, function="login_user_service")
        log_state(AuthLogs.SUCCESS, function="login_user_service", user_id=fetched_user.user_id)
        
        data: TokenSchema = TokenSchema(access_token=access_token, token_type="bearer") #manual validation
        return APIResponse(
            success=True,
            data=data,
            error_code=None,
            error_message=None
        )

    except Exception as e:
        log_state(AuthLogs.OPERATION_FAILED, function="login_user_service", level=LogState.EXCEPTION, exc=e, user_id=fetched_user.user_id)

        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.UNKNOWN_ERROR.value,
            error_message="Unexpected server error."
        )

    finally:
        log_state(AuthLogs.EXITING_AUTH_SERVICE, function="login_user_service", user_id=fetched_user.user_id)

async def logout_user_service(user_payload: TokenDataSchema, redis: Redis) -> APIResponse:
    log_state(AuthLogs.AUTH_SERVICE_STARTED, function="logout_user_service")
    try:
        log_state(AuthLogs.CONNECTING_TO_REDIS, function="logout_user_service")
        log_state(AuthLogs.CHECKING_SESSION, function="logout_user_service")
        log_state(AuthLogs.DELETING_SESSION, function="logout_user_service")

        deleted = await redis.delete(f"session:{user_payload.jid}")
        if deleted == 0:
            log_state(AuthLogs.SESSION_NOT_FOUND, function="logout_user_service", level=LogState.WARNING)
            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.RESOURCE_NOT_FOUND.value,
                error_message="Session not found or already logged out."
            )
        await redis.srem(f"user_sessions:{user_payload.user_id}", user_payload.jid)
        remaining = await redis.scard(f"user_sessions:{user_payload.user_id}")
        
        if remaining == 0: 
            await redis.delete(f"user_sessions:{user_payload.user_id}")

        log_state(AuthLogs.SESSION_DELETED, function="logout_user_service")
        log_state(AuthLogs.LOGGING_OUT_USER, function="logout_user_service", user_id=user_payload.user_id)
        log_state(AuthLogs.SUCCESS, function="logout_user_service")

        return APIResponse(
            success=True,
            data=200,
            error_code=None,
            error_message=None
        )

    except Exception as e:
        log_state(AuthLogs.OPERATION_FAILED, function="logout_user_service", level=LogState.EXCEPTION, exc=e)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.UNKNOWN_ERROR.value,
            error_message="Unexpected server error."
        )

    finally:
        log_state(AuthLogs.EXITING_AUTH_SERVICE, function="logout_user_service")


async def logout_all_devices_service(user_payload: TokenDataSchema, redis: Redis) -> APIResponse:
    log_state(AuthLogs.AUTH_SERVICE_STARTED, function="logout_all_devices_service", user_id=user_payload.user_id)
    try:
        log_state(AuthLogs.CONNECTING_TO_REDIS, function="logout_all_devices_service", user_id=user_payload.user_id)
        log_state(AuthLogs.CHECKING_ALL_SESSIONS, function="logout_all_devices_service", user_id=user_payload.user_id)
        
        jids: set = await redis.smembers(f"user_sessions:{user_payload.user_id}") #gets us all the active sessions of same user in the set
        if not jids:
            return APIResponse(
                success=False,
                data=None,
                error_code=SYSTEM_ERROR_CODES.NO_AVALIBLE_SESSIONS.value,
                error_message="No active sessions available."
            )
        log_state(AuthLogs.DELETING_ALL_SESSIONS, function="logout_all_devices_service", user_id=user_payload.user_id)
        for jid in jids:
            await redis.delete(f"session:{jid}") #del each jid in set
        await redis.delete(f"user_sessions:{user_payload.user_id}") #del the set itself
        
        log_state(AuthLogs.ALL_SESSIONS_DELETED, function="logout_all_devices_service", user_id=user_payload.user_id)
        log_state(AuthLogs.LOGGING_OUT_USER, function="logout_all_devices_service", user_id=user_payload.user_id)
        log_state(AuthLogs.SUCCESS, function="logout_all_devices_service", user_id=user_payload.user_id)

        return APIResponse(
            success=True,
            data=None,
            error_code=None,
            error_message=None
        )
        
    except Exception as e:
        log_state(AuthLogs.OPERATION_FAILED, function="logout_all_devices_service", level=LogState.EXCEPTION, exc=e, user_id=user_payload.user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.UNKNOWN_ERROR.value,
            error_message="Unexpected server error."
        )

    finally:
        log_state(AuthLogs.EXITING_AUTH_SERVICE, function="logout_all_devices_service", user_id=user_payload.user_id)


async def active_sessions_count_service(user_payload: TokenDataSchema, redis: Redis) -> APIResponse:
    log_state(AuthLogs.AUTH_SERVICE_STARTED, function="active_sessions_count_service", user_id=user_payload.user_id)
    try:
        log_state(AuthLogs.CONNECTING_TO_REDIS, function="active_sessions_count_service", user_id=user_payload.user_id)
        log_state(AuthLogs.CHECKING_ALL_SESSIONS, function="active_sessions_count_service", user_id=user_payload.user_id)

        
        log_state(AuthLogs.COUNTING_ACTIVE_SESSIONS, function="active_sessions_count_service", user_id=user_payload.user_id)
        session_count: int = await redis.scard(f"user_sessions:{user_payload.user_id}") 
        jids: list = list(await redis.smembers(f"user_sessions:{user_payload.user_id}"))
        
        if not jids and session_count == 0:
            return APIResponse(
                data=0,
                success=True,
                error_code=None,
                error_message=None
            )
    
        data = {"session_count": session_count, "active_session_list": jids}
        log_state(AuthLogs.SUCCESS, function="active_sessions_count_service", user_id=user_payload.user_id)
        return APIResponse(
            success=True,
            data=data,
            error_code=None,
            error_message=None
        )
        
    except Exception as e:
        log_state(AuthLogs.OPERATION_FAILED, function="active_sessions_count_service", level=LogState.EXCEPTION, exc=e, user_id=user_payload.user_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.UNKNOWN_ERROR.value,
            error_message="Unexpected server error."
        )

    finally:
        log_state(AuthLogs.EXITING_AUTH_SERVICE, function="active_sessions_count_service", user_id=user_payload.user_id)

async def revoke_curr_session_service(user_payload: TokenDataSchema, redis: Redis) -> None:
    log_state(AuthLogs.AUTH_SERVICE_STARTED, function="revoke_curr_session_service", user_id=user_payload.user_id)
    await logout_user_service(user_payload=user_payload, redis=redis)
    log_state(AuthLogs.EXITING_AUTH_SERVICE, function="revoke_curr_session_service", user_id=user_payload.user_id)
