from utils.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from docling.chunking import HybridChunker


max_tokens = settings.tokenizer_max_tokens
model_id = settings.tokenizer
tokenizer_ = AutoTokenizer.from_pretrained(model_id)


#tokenizer_ -> points to HF repo of which has (model and model's tokenizer) [POINTS TO REPO!] 
#now HybridChunker uses same repo to pull tokenizer form repo for same model! and chunks wrt to vocab_id of same model! (now go to ai_service.py) --eq(1)
chunker = (
    HybridChunker(  # internally chunks on the basises of similarity of next sentence with another so, onece a new convo starts thats where its gon chunk
        # assuming last convo finished roughlysepaking actual is more robust and has overlap internally so dont even wory!
        tokenizer=tokenizer_,
        max_tokens=max_tokens,  # useally enough to preserve context
        merge_peers=True,  # Merge small adjacent chunks
    )
)