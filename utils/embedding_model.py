from langchain_huggingface import HuggingFaceEmbeddings
from utils.config import settings
embedding_model = HuggingFaceEmbeddings(
    model_name=settings.embedding_model
)