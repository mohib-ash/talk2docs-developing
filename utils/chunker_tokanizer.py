from utils.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from docling.chunking import HybridChunker


max_tokens = settings.tokenizer_max_tokens
model_id = settings.tokenizer
tokenizer_ = AutoTokenizer.from_pretrained(model_id)


chunker = (
    HybridChunker(  
        tokenizer=tokenizer_,
        max_tokens=max_tokens,
        merge_peers=True,  
    )
)