from datetime import datetime, timezone
from db import Base
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship
from sqlalchemy import Enum
from utils.schemas import BM25Status, CacheVDBStatus, DocumentStatus, MultiIndexStatus
from sqlalchemy import Boolean, text


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    is_banned = Column(Boolean, nullable=False, server_default=text("false"))

    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    
    bm25_resource = relationship("BM25Resource", back_populates="user", uselist=False, cascade="all, delete-orphan")
    cache_vdb_resource = relationship("CacheVDBResource", back_populates="user", uselist=False, cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, default=1, server_default="1", nullable=False)
    
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    request_id = Column(String(36), unique=True, nullable=False, index=True)

    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), unique=True, nullable=False)
    collection_name = Column(String(100), nullable=False)

    file_extension = Column(String(20), nullable=False)
    file_size = Column(Integer, nullable=False)
    
    file_path = Column(String(512), nullable=False)
    file_dir = Column(String(100), nullable=False)
    markdown_path = Column(String(512), nullable=True)

    chunk_count = Column(Integer, default=0, server_default="0", nullable=False)
    mime_type = Column(String(100), nullable=False)

    failure_reason = Column(Text, nullable=True)
    
    
    status = Column(Enum(DocumentStatus), default=DocumentStatus.UPLOADED, nullable=False, index=True)
    
    summary_vdb_status = Column(Enum(MultiIndexStatus), default=MultiIndexStatus.PENDING, nullable=False, index=True)
    explanation_vdb_status = Column(Enum(MultiIndexStatus), default=MultiIndexStatus.PENDING, nullable=False, index=True)

    embedding_model = Column(String(100), nullable=True)
    file_hash = Column(String(64), nullable=False) #so we know if a same file came again

    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    

    user = relationship("User", back_populates="documents")


class Message(Base):
    __tablename__ = "messages"

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    doc_id = Column(Integer, ForeignKey("documents.doc_id", ondelete="SET NULL"), nullable=True, index=True)

    prompt = Column(Text, nullable=False)


    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    user = relationship("User", back_populates="messages")
    ai_response = relationship("AiResponse", back_populates="message", uselist=False, cascade="all, delete-orphan")


class AiResponse(Base):
    __tablename__ = "ai_responses"

    response_id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False, unique=True)

    response_text = Column(Text, nullable=False)

    retrieved_chunks_count = Column(Integer, default=3, server_default="3", nullable=False)
    processing_time_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    message = relationship("Message", back_populates="ai_response")



class BM25Resource(Base):
    __tablename__ = "bm25_resources"

    bm25_id = Column(Integer, primary_key=True, autoincrement=True)

    user_id = Column(
        Integer,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    ) 


    status = Column(Enum(BM25Status), default=BM25Status.PENDING, nullable=False, index=True)
    version = Column(Integer, default=0, server_default="0", nullable=False)

    index_path = Column(String(512), nullable=True)
    failure_reason = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", back_populates="bm25_resource")



class CacheVDBResource(Base):
    __tablename__ = "cache_vdb_resources"

    cache_vdb_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, unique=True, index=True)

    status = Column(Enum(CacheVDBStatus), default=CacheVDBStatus.PENDING, nullable=False, index=True)
    version = Column(Integer, default=0, server_default="0", nullable=False)

    vdb_path = Column(String(512), nullable=True)
    failure_reason = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user = relationship("User", back_populates="cache_vdb_resource")