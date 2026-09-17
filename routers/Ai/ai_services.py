from pathlib import Path
from uuid import uuid4
from fastapi import UploadFile
import filetype
import secrets
from core.Exceptions.exceptions import ChunkingParsedFileException, DocumentNotFoundException, EmbeddingChunkedFileException, ParsingSavedFileException, SavingValidatedFileException, UploadingException
from db_tables.tables import Document, BM25Resource
from sqlalchemy import select
from utils.APIResponce_error_code_enum import SYSTEM_ERROR_CODES, USER_ERROR_CODES
from utils.config import settings
from docling.document_converter import DocumentConverter
from utils.logging.helper_log import log_state
from utils.logging.logEvents import UploadFileLogs
from utils.schemas import DocumentStatus, SavedDocumentPayload, TokenDataSchema, UploadTaskPayload, APIResponse, passed_vlidation_reponce, BM25Status
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document as LangChainDocument #coz ive Document as a db table dont want no mixing
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker import BaseChunk
from utils.chunker_tokanizer import chunker
from utils.schemas import UploadTask2_fail_cases
import hashlib
import io
import zipfile
from utils.schemas import LogState
from starlette.concurrency import run_in_threadpool
import asyncio
import time


converter = DocumentConverter()
max_tokens = settings.tokenizer_max_tokens



from utils.embedding_model import embedding_model


UPLOAD_DIR = settings.upload_dir
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTENSIONS = {
    # PDF
    ".pdf",
    # Office / OpenDocument
    ".docx", ".pptx", ".xlsx",
    ".odt", ".ods", ".odp",
    # Plain text & Markup
    ".txt", ".text", ".md", ".qmd", ".rmd",
    ".html", ".xhtml", ".adoc", ".asciidoc", ".tex",
    # Structured & Mail
    ".csv", ".epub", ".eml", ".msg",
}

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/vnd.oasis.opendocument.presentation",
    "text/plain", "text/markdown", "text/html", "application/xhtml+xml",
    "text/csv", "application/epub+zip", "message/rfc822",
    "application/vnd.ms-outlook", "application/x-tex",
    "text/asciidoc", "text/x-asciidoc", "text/x-markdown",
    "application/octet-stream",
}

TEXT_EXTENSIONS = {
    ".txt", ".text", ".md", ".qmd", ".rmd",
    ".html", ".xhtml", ".adoc", ".asciidoc",
    ".tex", ".csv", ".eml",
}

ZIP_CONTAINER_SIGNATURES = {
    ".docx": "word/document.xml",
    ".pptx": "ppt/presentation.xml",
    ".xlsx": "xl/workbook.xml",
}

OPEN_DOCUMENT_MIMES = {
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".epub": "application/epub+zip",
}

OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

DEFAULT_COLLECTION_NAME = settings.defualt_collection_name




def _validate_text_file(file_bytes: bytes) -> bool:
    """Validates text encoding using UTF-8 (with NUL-byte guards) or BOM-aware UTF-16."""
    try:
        file_bytes.decode("utf-8")

        if b"\x00" in file_bytes[:4096]:
            return False
        return True
    except UnicodeDecodeError:
        pass

    try:
        
        file_bytes.decode("utf-16")
        return True
    except UnicodeDecodeError:
        return False
    





def _validate_zip_container(file_bytes: bytes, file_extension: str, user_id: int) -> bool:
    """Inspects zip structures and verifies internal XML structures or mimetype metadata."""
    log_state(UploadFileLogs.VALIDATING_MAGIC_BYTES, function="_validate_zip_container", user_id=user_id)
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:  
            namelist = z.namelist()  
            required_file = ZIP_CONTAINER_SIGNATURES.get(file_extension) 
            
            if required_file:  
                is_valid = required_file in namelist
                if not is_valid:
                    log_state(UploadFileLogs.VALIDATION_FAILED, function="_validate_zip_container", user_id=user_id)
                return is_valid


            expected_mime = OPEN_DOCUMENT_MIMES.get(file_extension)
            if expected_mime: 
                if "mimetype" in namelist:  
                    with z.open("mimetype") as f: 
                        internal_mime = f.read().decode("utf-8", errors="ignore").strip()  
                        is_valid = (internal_mime == expected_mime)  
                        if not is_valid:
                            log_state(UploadFileLogs.VALIDATION_FAILED, function="_validate_zip_container", user_id=user_id)
                        return is_valid
                log_state(UploadFileLogs.VALIDATION_FAILED, function="_validate_zip_container", user_id=user_id)
                return False

            log_state(UploadFileLogs.VALIDATION_FAILED, function="_validate_zip_container", user_id=user_id)
            return False
            
    except zipfile.BadZipFile as e:
        log_state(UploadFileLogs.VALIDATION_SERVICE_FAILURE, level=LogState.EXCEPTION, function="_validate_zip_container", exc=e, user_id=user_id)
        return False



def validate_file_content_type(file_bytes: bytes, file_extension: str, user_id: int) -> bool:
    """Authoritative validation dispatcher based on file category."""
    log_state(UploadFileLogs.VALIDATING_MAGIC_BYTES, function="validate_file_content_type", user_id=user_id)

    # 1. Text Formats
    if file_extension in TEXT_EXTENSIONS:
        if filetype.guess(file_bytes) is not None:  
            log_state(UploadFileLogs.VALIDATION_FAILED, function="validate_file_content_type", user_id=user_id)
            return False  
        return _validate_text_file(file_bytes)

    # 2. PDF
    if file_extension == ".pdf":
        is_valid = file_bytes.startswith(b"%PDF-")  
        if not is_valid:
            log_state(UploadFileLogs.VALIDATION_FAILED, function="validate_file_content_type", user_id=user_id)
        return is_valid

    # 3. Zip Containers (.docx, .pptx, .xlsx, .odt, .ods, .odp, .epub)
    if file_extension in ZIP_CONTAINER_SIGNATURES or file_extension in OPEN_DOCUMENT_MIMES: 
        return _validate_zip_container(file_bytes, file_extension, user_id=user_id)  

    # 4. Outlook MSG
    if file_extension == ".msg":
        is_valid = file_bytes.startswith(OLE2_MAGIC) 
        if not is_valid:
            log_state(UploadFileLogs.VALIDATION_FAILED, function="validate_file_content_type", user_id=user_id)
        return is_valid

    log_state(UploadFileLogs.VALIDATION_FAILED, function="validate_file_content_type", user_id=user_id)
    return False





async def file_validation_service(
    file: UploadFile,
    user_jwt_payload: TokenDataSchema,
    db: AsyncSession,
) -> APIResponse:
    request_id = str(uuid4())
    user_id = user_jwt_payload.user_id
    
    log_state(UploadFileLogs.VALIDATION_SERVICE_STARTED, function="file_validation_service", user_id=user_id, request_id=request_id)

    # STEP 1: Verify filename exists
    log_state(UploadFileLogs.CHECKING_FILENAME, function="file_validation_service", user_id=user_id, request_id=request_id)
    original_filename = file.filename
    if not original_filename:
        log_state(UploadFileLogs.VALIDATION_FAILED, function="file_validation_service", user_id=user_id, request_id=request_id)
        log_state(UploadFileLogs.EXITING_VALIDATION_SERVICE, function="file_validation_service", user_id=user_id, request_id=request_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=USER_ERROR_CODES.UNSUPPORTED_FILE.value,
            error_message="File has no name.",
        )

    file_extension = str(Path(original_filename).suffix.lower())
    file_content_type = file.content_type

    # STEP 2: Fast-Fail — Extension validation
    log_state(UploadFileLogs.CHECKING_EXTENSION, function="file_validation_service", user_id=user_id, request_id=request_id)
    if file_extension not in ALLOWED_EXTENSIONS:
        log_state(UploadFileLogs.VALIDATION_FAILED, function="file_validation_service", user_id=user_id, request_id=request_id)
        log_state(UploadFileLogs.EXITING_VALIDATION_SERVICE, function="file_validation_service", user_id=user_id, request_id=request_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=USER_ERROR_CODES.UNSUPPORTED_FILE.value,
            error_message=f"Unsupported file extension '{file_extension}'.",
        )

    # STEP 3: Fast-Fail — Client MIME Check
    log_state(UploadFileLogs.CHECKING_MIME_TYPE, function="file_validation_service", user_id=user_id, request_id=request_id)
    if file_content_type not in ALLOWED_MIME_TYPES:
        log_state(UploadFileLogs.VALIDATION_FAILED, function="file_validation_service", user_id=user_id, request_id=request_id)
        log_state(UploadFileLogs.EXITING_VALIDATION_SERVICE, function="file_validation_service", user_id=user_id, request_id=request_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=USER_ERROR_CODES.UNSUPPORTED_FILE.value,
            error_message="Invalid MIME type header.",
        )

    try:
        # STEP 4: Read File Bytes & Fast-Fail on Empty / Size Limit
        log_state(UploadFileLogs.READING_FILE_BYTES, function="file_validation_service", user_id=user_id, request_id=request_id)
        file_bytes: bytes = await file.read()

        if not file_bytes:
            log_state(UploadFileLogs.VALIDATION_FAILED, function="file_validation_service", user_id=user_id, request_id=request_id)
            log_state(UploadFileLogs.EXITING_VALIDATION_SERVICE, function="file_validation_service", user_id=user_id, request_id=request_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.INAPPROPRIATE_FILE.value,
                error_message="File is empty.",
            )

        if len(file_bytes) > MAX_FILE_SIZE:
            log_state(UploadFileLogs.VALIDATION_FAILED, function="file_validation_service", user_id=user_id, request_id=request_id)
            log_state(UploadFileLogs.EXITING_VALIDATION_SERVICE, function="file_validation_service", user_id=user_id, request_id=request_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.FILE_TOO_BIG.value,
                error_message="File size exceeds maximum limit of 25MB.",
            )

        
        log_state(UploadFileLogs.VALIDATING_MAGIC_BYTES, function="file_validation_service", user_id=user_id, request_id=request_id)
        start_t = time.time()
        is_valid_content = await asyncio.to_thread(
            validate_file_content_type, file_bytes, file_extension, user_id
            )
        print(f"Content validation took: {time.time() - start_t} seconds")

        if not is_valid_content:
            log_state(UploadFileLogs.VALIDATION_FAILED, function="file_validation_service", user_id=user_id, request_id=request_id)
            log_state(UploadFileLogs.EXITING_VALIDATION_SERVICE, function="file_validation_service", user_id=user_id, request_id=request_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.INAPPROPRIATE_FILE.value,
                error_message="File signature or internal structure does not match extension.",
            )

        # STEP 6: SHA-256 Duplicate Check
        log_state(UploadFileLogs.CHECKING_DUPLICATE_HASH, function="file_validation_service", user_id=user_id, request_id=request_id)
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        stmt = select(Document).where(Document.user_id == user_id, Document.file_hash == file_hash)
        result = await db.execute(stmt)
        
        if result.scalar_one_or_none() is not None:
            log_state(UploadFileLogs.VALIDATION_FAILED, function="file_validation_service", user_id=user_id, request_id=request_id)
            log_state(UploadFileLogs.EXITING_VALIDATION_SERVICE, function="file_validation_service", user_id=user_id, request_id=request_id)
            return APIResponse(
                success=False,
                data=None,
                error_code=USER_ERROR_CODES.DUPLICATE_FILE.value,
                error_message="You have already uploaded this document.",
            )


        # STEP 7: Construct Validated Payload
        log_state(UploadFileLogs.CONSTRUCTING_PAYLOAD, function="file_validation_service", user_id=user_id, request_id=request_id)
        stored_filename = f"{secrets.token_hex(16)}{file_extension}"
        file_path = UPLOAD_DIR / stored_filename

        file_payload = UploadTaskPayload(
            request_id=request_id,
            user_id=user_id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_dir=str(UPLOAD_DIR),
            file_hash=file_hash,
            file_extension=file_extension,
            file_content_type=file_content_type,
            file_path=str(file_path),
        )

    except Exception as e:
        log_state(UploadFileLogs.VALIDATION_SERVICE_FAILURE, level=LogState.EXCEPTION, function="file_validation_service", exc=e, user_id=user_id, request_id=request_id)
        log_state(UploadFileLogs.EXITING_VALIDATION_SERVICE, function="file_validation_service", user_id=user_id, request_id=request_id)
        return APIResponse(
            success=False,
            data=None,
            error_code=SYSTEM_ERROR_CODES.FILE_VALIDATION_FAILED.value,
            error_message="File validation failed during processing.",
        )

    response_data = passed_vlidation_reponce(
        file_payload=file_payload,
        file_bytes=file_bytes,
    )

    log_state(UploadFileLogs.VALIDATION_SERVICE_SUCCESS, function="file_validation_service", user_id=user_id, request_id=request_id)
    log_state(UploadFileLogs.EXITING_VALIDATION_SERVICE, function="file_validation_service", user_id=user_id, request_id=request_id)

    return APIResponse(
        success=True,
        data=response_data,
        error_code=None,
        error_message=None,
    )




async def save_validated_doc_task_service(payload: passed_vlidation_reponce, db: AsyncSession):
    doc_bytes: bytes = payload.file_bytes
    meta: UploadTaskPayload = payload.file_payload

    log_state(UploadFileLogs.SAVE_SERVICE_STARTED, function="save_validated_doc_task_service", user_id=meta.user_id, request_id=meta.request_id)

    try:
        # Save file
        log_state(UploadFileLogs.WRITING_FILE_TO_DISK, function="save_validated_doc_task_service", user_id=meta.user_id, request_id=meta.request_id)
        with open(meta.file_path, "wb") as f:
            f.write(doc_bytes)

        document = Document(
            request_id=meta.request_id,
            user_id=meta.user_id,
            original_filename=meta.original_filename,
            stored_filename=meta.stored_filename,
            file_dir=meta.file_dir,
            file_hash=meta.file_hash,
            file_extension=meta.file_extension,
            file_size=len(doc_bytes),
            file_path=meta.file_path,
            mime_type=meta.file_content_type,
            collection_name=DEFAULT_COLLECTION_NAME,
            status=DocumentStatus.UPLOADED,
        )

        db.add(document)

        log_state(UploadFileLogs.COMMITTING_DOC_TO_DB, function="save_validated_doc_task_service", user_id=meta.user_id, request_id=meta.request_id)
        await db.commit()
        await db.refresh(document) 
        
        log_state(UploadFileLogs.VALIDATING_SAVED_DOC_ORM, function="save_validated_doc_task_service", user_id=meta.user_id, request_id=meta.request_id)
        document_metadata = SavedDocumentPayload.model_validate(document)
        
        log_state(UploadFileLogs.SAVE_SERVICE_SUCCESS, function="save_validated_doc_task_service", user_id=meta.user_id, request_id=meta.request_id)
        return document_metadata.model_dump() 


    except Exception as e:
        log_state(UploadFileLogs.SAVE_SERVICE_FAILED_ROLLING_BACK, function="save_validated_doc_task_service", user_id=meta.user_id, request_id=meta.request_id, exc=e)
        await db.rollback()

        if Path(meta.file_path).exists():
            Path(meta.file_path).unlink()

        raise SavingValidatedFileException(
            error_code=SYSTEM_ERROR_CODES.SAVING_VALIDATED_FILE_EXCEPTION.value,
            message="Unexpected error saving validated file in dir"
        ) from e






def parse_stage(document: Document) -> DoclingDocument:
    result = converter.convert(document.file_path)
    
    document.status = DocumentStatus.PARSED
    doc: DoclingDocument = result.document
    
    return doc

def chunking_stage(document: Document, doc: DoclingDocument) -> list[BaseChunk]:
    chunks = list(chunker.chunk(dl_doc=doc))
    document.chunk_count = len(chunks)
    document.status = DocumentStatus.CHUNKED
    return chunks

def build_langchain_doc(document: Document, chunks: list[BaseChunk]) -> list[LangChainDocument]:
    docs = []
    for index, chunk in enumerate(chunks):
        raw_content = chunker.contextualize(chunk)
        docs.append(
            LangChainDocument(
                page_content=raw_content,
                metadata={
                    "chunk_id": f"{document.doc_id}_{index}",
                    "doc_type": "raw",
                    "doc_id": document.doc_id,
                    "request_id": document.request_id,
                    "file_name": document.original_filename,
                    "user_id": document.user_id,
                },
            )
        )
    return docs

async def embeding_stage(document: Document, chunks: list[BaseChunk], db: AsyncSession, user_id: int):
    docs = build_langchain_doc(document=document, chunks=chunks)
    user_id = document.user_id
    user_chroma_dir = (Path(settings.chroma_db_dir) / f"user_{user_id}")
    
    
    Chroma.from_documents( 
        documents=docs,
        embedding=embedding_model,
        collection_name=document.collection_name,
        persist_directory=str(user_chroma_dir),
    )
    document.status = DocumentStatus.READY


    #why do this? well its a mark! which means new doc has come by! and bm25 needs to be updated in the future!
    # Find the user's BM25 resource and mark it stale if it was previously READY
    bm25_stmt = select(BM25Resource).where(BM25Resource.user_id == user_id)
    bm25_result = await db.execute(bm25_stmt)
    bm25_res = bm25_result.scalar_one_or_none()

    if bm25_res and bm25_res.status == BM25Status.READY:
        bm25_res.status = BM25Status.STALE
        await db.commit()

async def failed_case(document: Document, db: AsyncSession, reason: Exception, stage: str):
    document.status = DocumentStatus.FAILED
    document.failure_reason = f"{stage} failed: {reason}"
    await db.commit()

async def parse_chunk_embed_saved_doc_task2_service(doc_meta_obj: SavedDocumentPayload, db: AsyncSession):
    log_state(UploadFileLogs.TASK_2_SERVICE_STARTED, function="parse_chunk_embed_saved_doc_task2_service", user_id=doc_meta_obj.user_id, request_id=doc_meta_obj.request_id)
    document = await db.get(Document, doc_meta_obj.doc_id)  

    if document is None:
        log_state(UploadFileLogs.DOCUMENT_NOT_FOUND_IN_SERVICE, function="parse_chunk_embed_saved_doc_task2_service", user_id=doc_meta_obj.user_id, request_id=doc_meta_obj.request_id)
        raise DocumentNotFoundException(
            error_code=SYSTEM_ERROR_CODES.DOCUMENT_NOT_FOUND.value,
            message="Document metadata could not be found.",
        )
        
    #1. PARSE STAGE
    log_state(UploadFileLogs.PARSING_STAGE_STARTED, function="parse_chunk_embed_saved_doc_task2_service", user_id=document.user_id, request_id=document.request_id)
    try:
        doc: DoclingDocument = parse_stage(document=document)
        await db.commit()
        log_state(UploadFileLogs.PARSING_STAGE_SUCCESS, function="parse_chunk_embed_saved_doc_task2_service", user_id=document.user_id, request_id=document.request_id)

    except Exception as exc:
        log_state(UploadFileLogs.PARSING_STAGE_FAILED, function="parse_chunk_embed_saved_doc_task2_service", user_id=document.user_id, request_id=document.request_id, exc=exc)
        await failed_case(document=document, db=db, reason=exc, stage=UploadTask2_fail_cases.PARSING.value)
        raise ParsingSavedFileException(
            error_code=SYSTEM_ERROR_CODES.FILE_PARSE_EXCEPTION.value,
            message="Failed to parse uploaded document.",
        ) from exc


    # 2. CHUNK STAGE
    log_state(UploadFileLogs.CHUNKING_STAGE_STARTED, function="parse_chunk_embed_saved_doc_task2_service", user_id=document.user_id, request_id=document.request_id)
    try:
        chunks = chunking_stage(document=document, doc=doc)
        await db.commit()
        log_state(UploadFileLogs.CHUNKING_STAGE_SUCCESS, function="parse_chunk_embed_saved_doc_task2_service", user_id=document.user_id, request_id=document.request_id)

    except Exception as exc:
        log_state(UploadFileLogs.CHUNKING_STAGE_FAILED, function="parse_chunk_embed_saved_doc_task2_service", user_id=document.user_id, request_id=document.request_id, exc=exc)
        await failed_case(document=document, db=db, reason=exc, stage=UploadTask2_fail_cases.CHUNKING.value)
        raise ChunkingParsedFileException(
            error_code=SYSTEM_ERROR_CODES.FILE_CHUNKING_EXCEPTION.value,
            message="Failed to chunk parsed document.",
        ) from exc


    # 3. EMBED
    log_state(UploadFileLogs.EMBEDDING_STAGE_STARTED, function="parse_chunk_embed_saved_doc_task2_service", user_id=document.user_id, request_id=document.request_id)
    try:
        await embeding_stage(document=document, chunks=chunks, user_id=document.user_id, db=db)
        await db.commit()
        # await db.refresh(document)# --eq(1) thats why we refresh! double bennift ;)
        log_state(UploadFileLogs.EMBEDDING_STAGE_SUCCESS, function="parse_chunk_embed_saved_doc_task2_service", user_id=document.user_id, request_id=document.request_id)

    except Exception as exc:
        log_state(UploadFileLogs.EMBEDDING_STAGE_FAILED, function="parse_chunk_embed_saved_doc_task2_service", user_id=document.user_id, request_id=document.request_id, exc=exc)
        await failed_case(document=document, db=db, reason=exc, stage=UploadTask2_fail_cases.EMBIDING.value)
        raise EmbeddingChunkedFileException(
            error_code=SYSTEM_ERROR_CODES.FILE_EMBEDING_EXCEPTION.value,
            message="Failed to generate embeddings and index document into ChromaDB.",
        ) from exc

    log_state(UploadFileLogs.TASK_2_SERVICE_SUCCESS, function="parse_chunk_embed_saved_doc_task2_service", user_id=document.user_id, request_id=document.request_id)