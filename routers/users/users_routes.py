from fastapi import Request, status, Depends, APIRouter, Response
from Oauth2 import get_user_jwt_payload
from core.Exceptions.exceptions import ChangePasswordException, UserCreationServiceException
from db_tables.tables import User
from db import get_db
from typing import List, Optional
from routers.users.users_services import change_passowrd_service, create_user_service, get_Nusers_service, get_user_by_id_service
from utils.ai_responce_handler import handle_service_response
from utils.schemas import ChangePasswordInputSchema, TokenDataSchema, UserResponseSchema, UserRegisterSchema
from sqlalchemy.ext.asyncio import AsyncSession
from utils.schemas import APIResponse
from core.rate_limiters.limiter_file import limiter
from core.rate_limiters.limiter_utils import RateLimits




router = APIRouter(
    prefix="/users",
    tags=["users"]
)


@router.post("/sign-up", status_code=status.HTTP_201_CREATED, response_model=UserResponseSchema)
@limiter.limit(RateLimits.Auth.REGISTER)
async def create_user(user: UserRegisterSchema, request: Request, response: Response, db: AsyncSession = Depends(get_db)) -> UserResponseSchema:
    result: APIResponse = await create_user_service(user=user, db=db)
    return handle_service_response(result, UserCreationServiceException)


@router.get("/admin/get_all_users", response_model=List[UserResponseSchema]) 
@limiter.limit(RateLimits.User.READ)
async def get_all_users(request: Request, response: Response, user_payload: TokenDataSchema = Depends(get_user_jwt_payload), db: AsyncSession = Depends(get_db), limit: int = 10, offset: int = 0, search: Optional[str] = None) -> List[UserResponseSchema]: 
    result: APIResponse = await get_Nusers_service(user_payload=user_payload, db=db, limit=limit, offset=offset, search=search)
    return handle_service_response(result, UserCreationServiceException)



@router.get("/admin/get_user_by_id/{id}", response_model=UserResponseSchema)
@limiter.limit(RateLimits.User.READ)
async def get_user_by_id(request: Request, response: Response, id: int, user_payload: TokenDataSchema = Depends(get_user_jwt_payload), db: AsyncSession = Depends(get_db)):  
    result: APIResponse = await get_user_by_id_service(user_payload=user_payload, db=db, id=id)
    return handle_service_response(result, UserCreationServiceException)



@router.post("/change-password", status_code=status.HTTP_200_OK)
@limiter.limit(RateLimits.Auth.CHANGE_PASSWORD)
async def change_passowrd(request: Request, response: Response, input_payload: ChangePasswordInputSchema, user_payload: TokenDataSchema = Depends(get_user_jwt_payload), db: AsyncSession = Depends(get_db)):
    result: APIResponse = await change_passowrd_service(user_payload=user_payload, db=db, input_payload=input_payload)
    return handle_service_response(result, ChangePasswordException)


