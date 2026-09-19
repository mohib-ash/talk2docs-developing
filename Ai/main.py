#& G:\python\G_Env\Scripts\python.exe "g:/python/BackEnd FastAPI/Lecs/L17/Ai/main.py"
from langchain_groq import ChatGroq
from utils.config import settings
from langchain_groq import ChatGroq
from utils.config import settings

# Initialize Model
model = ChatGroq(
    api_key=settings.api_key,   
    model=settings.model
)
