from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)  # 新增邮箱字段
    hashed_password = Column(String)
    role = Column(String, default="student")  # student, admin
    reset_code = Column(String, nullable=True)  # 存储重置验证码
    reset_code_expires = Column(DateTime, nullable=True)  # 验证码过期时间
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="active")  # active, disabled

    conversations = relationship("ChatHistory", back_populates="user")

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String)  # user, assistant
    content = Column(Text)
    image_url = Column(String, nullable=True)
    source_label = Column(String, nullable=True)  # 新增：来源标注（大模型/知识库）
    matched_docs = Column(Text, nullable=True)    # 新增：匹配的文档列表（存储为JSON字符串）
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    feedbacks = relationship("Feedback", back_populates="message")

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    message_id = Column(Integer, ForeignKey("chat_history.id"), index=True)
    feedback_type = Column(String)  # like / dislike
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")
    message = relationship("ChatHistory", back_populates="feedbacks")

class KnowledgeFile(Base):
    __tablename__ = "knowledge_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_path = Column(String)
    file_type = Column(String)
    file_size = Column(Integer, nullable=True)  # 文件大小(字节)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    progress = Column(Integer, default=0)       # 0-100
    error_message = Column(String, nullable=True) # 失败原因
    created_at = Column(DateTime, default=datetime.utcnow)
