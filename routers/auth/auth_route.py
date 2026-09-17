from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession 
from Oauth2 import get_user_jwt_payload
from core.rate_limiters.limiter_file import limiter
from core.rate_limiters.limiter_utils import RateLimits
from core.Exceptions.exceptions import AllServiceContException, LoginServiceException, LogoutAllDeviServiceException, LogoutServiceException, RvokeCurrentSessionException
from db import get_db
from db_tables.tables import User
from routers.auth.auth_service import active_sessions_count_service, login_user_service, logout_all_devices_service, logout_user_service, revoke_curr_session_service
from core.redis import get_redis
from utils.ai_responce_handler import handle_service_response
from utils.schemas import TokenDataSchema, TokenSchema
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from utils.schemas import APIResponse
from redis.asyncio import Redis
from fastapi import status


router = APIRouter(tags=["Authentication"])


@router.post('/login', response_model=TokenSchema)
@limiter.limit(RateLimits.Auth.LOGIN) 
async def login(request: Request, response: Response, user_credentials: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db), redis: Redis = Depends(get_redis)):
    result: APIResponse = await login_user_service(user_credentials=user_credentials, db=db, redis=redis)
    return handle_service_response(result, LoginServiceException)

@router.post("/logout")
@limiter.limit(RateLimits.Auth.LOGOUT)
async def logout(request: Request, response: Response, user_payload: TokenDataSchema = Depends(get_user_jwt_payload), redis: Redis = Depends(get_redis)):
    result: APIResponse = await logout_user_service(user_payload=user_payload, redis=redis)
    return handle_service_response(result, LogoutServiceException)

@router.post('/logout_all')
@limiter.limit(RateLimits.Auth.LOGOUT_ALL)
async def logOut_all(request: Request, response: Response, user_payload: TokenDataSchema = Depends(get_user_jwt_payload), redis: Redis = Depends(get_redis)):
    result: APIResponse = await logout_all_devices_service(user_payload=user_payload, redis=redis)
    return handle_service_response(result, LogoutAllDeviServiceException)



@router.get("/sessions")
@limiter.limit(RateLimits.Session.COUNT_ACTIVE)
async def get_active_sessions(request: Request, response: Response, user_payload: TokenDataSchema = Depends(get_user_jwt_payload), redis: Redis = Depends(get_redis)):
    result: APIResponse = await active_sessions_count_service(user_payload=user_payload, redis=redis)
    return handle_service_response(result, AllServiceContException)




@router.delete("/revoke_curr_session", status_code=status.HTTP_200_OK)
@limiter.limit(RateLimits.Session.REVOKE_CURRENT)
async def revoke_current_session(request: Request, response: Response, user_payload: TokenDataSchema = Depends(get_user_jwt_payload), redis: Redis = Depends(get_redis)):
    result: APIResponse = await revoke_curr_session_service(user_payload=user_payload, redis=redis)
    return handle_service_response(result, RvokeCurrentSessionException)






@router.get('/admin/get_all_active_sessions_by_id/{id}')
async def get_all_active_sessions(id: int, redis: Redis = Depends(get_redis)):
    sessions = await redis.scard(f"user_sessions:{id}")
    return {
        "user_id": id,
        "number_of_active_sessions": sessions
    }


@router.post('/admin/revoke-all-sessions-by-id/{id}', status_code=status.HTTP_200_OK)
async def revoke_all_sessions(id: int, redis: Redis = Depends(get_redis)):
    session_ids = await redis.smembers(f"user_sessions:{id}")

    if session_ids:
        for jid in session_ids:
            await redis.delete(f"session:{jid}")
        await redis.delete(f"user_sessions:{id}") 
    
    else:
        return {
    "message": "No active sessions found."
}



@router.put("/admin/ban-user/{id}", status_code=status.HTTP_200_OK)
async def ban_user(id: int, db: AsyncSession = Depends(get_db), redis: Redis = Depends(get_redis)):
    user = await db.get(User, id)
    if not user:
        return {
            "message": "User not found"
        }
    
    if user.is_banned:
        return {
            "message": "User already banned"
        }
    user.is_banned = True
    await db.commit()
    
    return {
        "message": "User banned successfully"
    }