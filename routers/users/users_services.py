from typing import Optional
from sqlalchemy import select
from db_tables.tables import User
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES, USER_ERROR_CODES
from utils.hashing import hash_password, verify_hashed_password
from utils.logging.helper_log import LogState, log_state
from utils.logging.logEvents import UserLogs
from utils.schemas import  ChangePasswordInputSchema, TokenDataSchema, UserRegisterSchema
from utils.schemas import APIResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError


async def create_user_service(user: UserRegisterSchema, db: AsyncSession) -> APIResponse:
    log_state(UserLogs.USER_SERVICE_STARTED, function="create_user_service")
    log_state(UserLogs.CREATING_USER, function="create_user_service")

    try:
        log_state(UserLogs.VALIDATING_REQUEST, function="create_user_service")
        hashed_pass = hash_password(user.password)
        new_user = User(
            email=user.email,
            password=hashed_pass
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        log_state(UserLogs.SUCCESS, function="create_user_service", user_id=new_user.user_id)
        return APIResponse(
            success=True,
            data=new_user,
            error_code=None,
            error_message=None
        )

    except SQLAlchemyError as e:
        await db.rollback()
        log_state(
            UserLogs.OPERATION_FAILED,
            function="create_user_service",
            level=LogState.ERROR,
            exc=e
        )

        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.DATABASE_ERROR.value,
            error_message="Database operation failed."
        )

    except Exception as e:
        await db.rollback()
        log_state(
            UserLogs.OPERATION_FAILED,
            function="create_user_service",
            level=LogState.EXCEPTION,
            exc=e
        )

        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.UNKNOWN_ERROR.value,
            error_message="Unexpected server error."
        )

    finally:
        log_state(UserLogs.EXITING_USER_SERVICE, function="create_user_service")

async def get_Nusers_service(user_payload, db: AsyncSession, limit: int = 10, offset: int = 0, search: Optional[str] = None) -> APIResponse:

    log_state(UserLogs.USER_SERVICE_STARTED, function="get_Nusers_service", user_id=user_payload.user_id)
    log_state(UserLogs.FETCHING_USERS, function="get_Nusers_service", user_id=user_payload.user_id)

    try:
        log_state(UserLogs.VALIDATING_REQUEST, function="get_Nusers_service", user_id=user_payload.user_id)

        if limit <= 0:
            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.UNSUPPORTED_INPUT.value,
                error_message="Limit must be greater than zero."
            )

        if offset < 0:
            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.UNSUPPORTED_INPUT.value,
                error_message="Offset cannot be negative."
            )

        log_state(UserLogs.EXECUTING_DATABASE_QUERY, function="get_Nusers_service", user_id=user_payload.user_id)
        stmt = select(User)

        if search:
            stmt = stmt.where(
                User.email.contains(search)
            )

        stmt = stmt.offset(offset).limit(limit)
        result = await db.execute(stmt)
        users = result.scalars().all()

        log_state(UserLogs.SUCCESS, function="get_Nusers_service", user_id=user_payload.user_id)
        return APIResponse(
            success=True,
            data=users,
            error_code=None,
            error_message=None
        )

    except SQLAlchemyError as e:
        log_state(UserLogs.OPERATION_FAILED, function="get_Nusers_service", user_id=user_payload.user_id, level=LogState.ERROR, exc=e)

        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.DATABASE_ERROR.value,
            error_message="Database query failed."
        )

    except Exception as e:
        log_state(UserLogs.OPERATION_FAILED, function="get_Nusers_service", user_id=user_payload.user_id, level=LogState.EXCEPTION, exc=e)

        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.UNKNOWN_ERROR.value,
            error_message="Unexpected server error."
        )

    finally:
        log_state(UserLogs.EXITING_USER_SERVICE, function="get_Nusers_service", user_id=user_payload.user_id)
        
async def get_user_by_id_service(user_payload, id: int, db: AsyncSession) -> APIResponse:
    log_state(UserLogs.USER_SERVICE_STARTED, function="get_user_by_id_service", user_id=user_payload.user_id)
    log_state(UserLogs.FETCHING_USER, function="get_user_by_id_service", user_id=user_payload.user_id)

    try:
        log_state(UserLogs.EXECUTING_DATABASE_QUERY, function="get_user_by_id_service", user_id=user_payload.user_id)
        result = await db.execute(
            select(User).where(User.user_id == id)
        )
        fetched_user = result.scalar_one_or_none()

        if not fetched_user:
            log_state(UserLogs.OPERATION_FAILED, function="get_user_by_id_service", user_id=user_payload.user_id, level=LogState.WARNING)

            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.RESOURCE_NOT_FOUND.value,
                error_message="User not found."
            )

        log_state(UserLogs.SUCCESS, function="get_user_by_id_service", user_id=user_payload.user_id)
        return APIResponse(
            success=True,
            data=fetched_user,
            error_code=None,
            error_message=None
        )

    except SQLAlchemyError as e:
        log_state(UserLogs.OPERATION_FAILED, function="get_user_by_id_service", user_id=user_payload.user_id, level=LogState.ERROR, exc=e)

        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.DATABASE_ERROR.value,
            error_message="Database query failed."
        )

    except Exception as e:
        log_state(UserLogs.OPERATION_FAILED, function="get_user_by_id_service", user_id=user_payload.user_id, level=LogState.EXCEPTION, exc=e)

        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.UNKNOWN_ERROR.value,
            error_message="Unexpected server error."
        )

    finally:
        log_state(UserLogs.EXITING_USER_SERVICE, function="get_user_by_id_service", user_id=user_payload.user_id)


async def change_passowrd_service(user_payload: TokenDataSchema, db: AsyncSession, input_payload: ChangePasswordInputSchema) -> APIResponse:
    log_state(UserLogs.USER_SERVICE_STARTED, function="change_passowrd_service", user_id=user_payload.user_id)
    
    given_old_pass = input_payload.old_pass
    given_new_pass = input_payload.new_pass    
    user_id = user_payload.user_id

    try:
        log_state(UserLogs.FETCHING_USER, function="change_passowrd_service", user_id=user_payload.user_id)
        stmt = select(User).where(User.user_id == user_id)
        
        log_state(UserLogs.EXECUTING_DATABASE_QUERY, function="change_passowrd_service", user_id=user_payload.user_id)
        result = await db.execute(stmt)
        
        user = result.scalar_one_or_none()  #user is the row which we fetched, we can update its any col!
        if user is None:
            log_state(UserLogs.OPERATION_FAILED, function="change_passowrd_service", user_id=user_payload.user_id, level=LogState.INFO)
            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.RESOURCE_NOT_FOUND.value,
                error_message="User not found."
            )
        
        if not verify_hashed_password(given_old_pass, user.password):
            log_state(UserLogs.AUTHENTICATION_FAILED, function="change_passowrd_service", user_id=user_payload.user_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.INVALID_CREDENTIALS.value,
                error_message="Invalid old password."
            )
        
        if verify_hashed_password(given_new_pass, user.password):
            log_state(UserLogs.OPERATION_FAILED, function="change_passowrd_service", user_id=user_payload.user_id, level=LogState.INFO)
            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.INPUT_ALREADY_EXISTS.value,
                error_message="New Password can't be same as old password."
            )
        
        user.password = hash_password(given_new_pass)
        log_state(UserLogs.UPDATING_USER, function="change_passowrd_service", user_id=user_payload.user_id, level=LogState.INFO)
        await db.commit()
        await db.refresh(user)
        log_state(UserLogs.SUCCESS, function="change_passowrd_service", user_id=user_payload.user_id, level=LogState.INFO)
        
        return APIResponse(
            success=True,
            data=None,
            error_code=None,
            error_message=None
        )
        
    except Exception as e:
        log_state(UserLogs.OPERATION_FAILED, function="change_passowrd_service", user_id=user_payload.user_id, level=LogState.EXCEPTION, exc=e)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.UNKNOWN_ERROR.value,
            error_message="Unexpected server error."
        )
    finally:
        log_state(UserLogs.EXITING_USER_SERVICE, function="change_passowrd_service", user_id=user_payload.user_id)