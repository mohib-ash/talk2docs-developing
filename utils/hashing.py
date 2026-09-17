from passlib.context import CryptContext

# Instantiate CryptContext utilizing bcrypt algorithm for password security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Generates secure one-way bcrypt string from raw user password."""
    return pwd_context.hash(password)

def verify_hashed_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password input against its stored database hash verification string."""
    return pwd_context.verify(plain_password, hashed_password)