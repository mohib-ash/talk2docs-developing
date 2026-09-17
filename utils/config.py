from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Parses system environment or `.env` file variables into structured Python attributes."""
    database_hostname: str
    database_port: str
    database_password: str
    database_name: str
    database_username: str
    
    hash_secret_key: str  
    algorithm: str
    access_token_expire_minutes: int
    
    
    #Ai:
    model: str
    api_key: str
    
    
    #updaded file:
    upload_dir: Path


    #tokenizer and embedding_model:
    tokenizer: str
    tokenizer_max_tokens: int 
    
    embedding_model: str 
    
    defualt_collection_name: str
    
    
    #chroma_db:
    chroma_db_dir: str 
    
    
    
    #reranker
    cohere_api_key: str
    cohere_rerank_model: str
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()