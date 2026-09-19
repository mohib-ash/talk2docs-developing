from fastapi import Depends, status, HTTPException
from jose import JWTError, jwt
from datetime import datetime, timezone, timedelta  
from core.redis import get_redis
from utils.schemas import TokenDataSchema
from fastapi.security import OAuth2PasswordBearer
from utils.config import settings
from redis.asyncio import Redis

SECRET_KEY = settings.hash_secret_key
ALGORITHM = settings.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='login')

def create_access_token(data: dict) -> str: 
    """Encodes operational dict data into signed JWT Access string."""
    to_encode_data = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode_data.update({"exp": expire})
    
    return jwt.encode(to_encode_data, SECRET_KEY, algorithm=ALGORITHM)


async def verify_access_token(token: str, credentials_exception, redis: Redis) -> TokenDataSchema:
    """Decodes string and extracts user identities, protecting endpoints from malicious signatures."""
    try:
        payload = jwt.decode(token, key=SECRET_KEY, algorithms=[ALGORITHM])
        user_id_from_payload = payload.get("user_id")
        jwt_id_from_payload = payload.get("jid")
        
        if user_id_from_payload is None:
            raise credentials_exception
        if jwt_id_from_payload is None:
            raise credentials_exception
        
        
        session_exists = await redis.exists(
            f"session:{jwt_id_from_payload}" 
        ) 
        
        if not session_exists:
            raise credentials_exception
        
    
        token_data: TokenDataSchema = TokenDataSchema(user_id=user_id_from_payload, jid=jwt_id_from_payload) 
    except JWTError:
        raise credentials_exception
    
    return token_data


async def get_user_jwt_payload(token: str = Depends(oauth2_scheme), redis: Redis = Depends(get_redis)) -> TokenDataSchema:
    """FastAPI Dependency enforcing explicit bearer authorization validation checks."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )
    return await verify_access_token(token=token, credentials_exception=credentials_exception, redis=redis)