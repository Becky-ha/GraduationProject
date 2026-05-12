from fastapi import FastAPI, Request, Depends, status, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import os
import json
import asyncio
import uuid
import base64
from PIL import Image
import io
import aiofiles
import numpy as np
import requests
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from dashscope import ImageSynthesis

# 数据库相关导入
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete, func
from models import Base, User as UserModel, ChatHistory as ChatHistoryModel, KnowledgeFile as KnowledgeFileModel, Feedback as FeedbackModel, QuestionClusterModel

from auth import get_password_hash, verify_password, create_access_token, get_current_user_id, get_current_user_id_optional

# LangChain导入
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 加载环境变量
load_dotenv()

# JWT配置
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Token过期时间（分钟）

# 获取DashScope API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    print("错误: 未在 .env 文件中找到 DASHSCOPE_API_KEY。请确保 .env 文件存在并包含正确的 API Key。")
    # 为了防止程序崩溃，可以设置一个占位符或提醒用户
    # os.environ["DASHSCOPE_API_KEY"] = "YOUR_API_KEY_HERE"
else:
    # 设置API密钥
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
    print(f"使用DashScope API Key: {DASHSCOPE_API_KEY[:8]}...（已隐藏部分）")

# 创建图片存储目录
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
print(f"图片上传目录: {UPLOAD_DIR}")

# 创建知识库存储目录
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
print(f"知识库目录: {KNOWLEDGE_DIR}")

# 创建向量库持久化目录
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
print(f"向量库持久化目录: {CHROMA_PERSIST_DIR}")

from file_parser import parse_file, load_all_knowledge

class LocalFallbackEmbeddings(Embeddings):
    """在 DashScope 嵌入不可用时的本地降级嵌入，用于保证向量库可创建。"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * 384
            for i, ch in enumerate(text):
                vec[i % 384] += (ord(ch) % 97) / 97.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


try:
    embeddings = None
    try:
        from langchain_community.embeddings.dashscope import DashScopeEmbeddings
        embeddings = DashScopeEmbeddings(model="tongyi-embedding-vision-flash-2026-03-06")
        test_vector = embeddings.embed_query("embedding health check")
        print(f"DashScope嵌入模型初始化成功，向量维度: {len(test_vector)}")
        print("使用新模型: tongyi-embedding-vision-flash-2026-03-06")
    except Exception as e:
        print(f"DashScope嵌入模型不可用，改用本地降级嵌入: {str(e)}")
        embeddings = LocalFallbackEmbeddings()
        print("本地降级嵌入初始化成功")
except Exception as e:
    print(f"嵌入模型初始化失败: {str(e)}")
    embeddings = LocalFallbackEmbeddings()
    print("强制启用本地降级嵌入")

# 数据库配置
DATABASE_URL = "sqlite+aiosqlite:///./ai_assistant.db"
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# 获取数据库会话的依赖
async def get_db():
    async with async_session() as session:
        yield session

from contextlib import asynccontextmanager

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                continue

manager = ConnectionManager()

# OCR 并发限制器（最多同时处理 2 个文件）
ocr_semaphore = asyncio.Semaphore(2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动时执行 ---
    # 在应用启动时创建数据库表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("数据库表已同步")
    
    # 初始化知识库
    print(">>> 正在初始化知识库...")
    global vectorstore, retriever
    vectorstore = initialize_rag()
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(">>> 知识库初始化完成")
    
    yield
    # --- 关闭时执行 ---
    print(">>> 正在关闭服务...")

# 初始化FastAPI应用
app = FastAPI(title="QAI Bot", lifespan=lifespan)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    image_url: Optional[str] = None  # 添加图片URL字段

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[Message]] = []

class MultiModalRequest(BaseModel):
    message: str
    chat_history: Optional[List[Message]] = []
    image_data: Optional[str] = None  # Base64编码的图片数据

class UserRegister(BaseModel):
    username: str
    password: str
    email: str
    role: Optional[str] = "student" # 新增角色字段，默认为 student

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserStatusUpdate(BaseModel):
    user_id: int
    status: str # 'active' or 'disabled'

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    email: str
    code: str
    new_password: str

class FeedbackRequest(BaseModel):
    message_id: int
    feedback_type: str  # 'like' or 'dislike'

class FeedbackStatResponse(BaseModel):
    like_count: int
    dislike_count: int
    user_feedback: Optional[str] = None

class FeedbackOverviewItem(BaseModel):
    message_id: int
    question: str
    answer: str
    like_count: int
    dislike_count: int

class HotQuestionItem(BaseModel):
    represent_question: str
    count: int
    examples: List[str]

class KnowledgeFileResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: Optional[int] = 0
    status: str
    progress: int
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {
        "from_attributes": True
    }

class KeywordCoverage(BaseModel):
    keyword: str
    is_covered: bool
    source_count: int = 0
    hot_count: int = 0
    suggested_action: str
    priority: str = "medium" # low, medium, high, urgent

class KnowledgeAnalysisResponse(BaseModel):
    total_files: int
    total_chunks: int
    coverage_rate: float
    keyword_details: List[KeywordCoverage]

class StreamingCallbackHandler:
    """用于处理流式回调的处理器"""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()
        
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """当新的token生成时将它放入队列"""
        await self.queue.put(token)
    
    async def on_llm_end(self, response: Any, **kwargs) -> None:
        """当LLM结束时设置事件标志"""
        self.done.set()
    
    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """处理错误"""
        await self.queue.put(f"Error: {str(error)}")
        self.done.set()

# 内存存储对话历史
conversation_store = {}

# WebSocket 端点：用于实时推送知识库解析进度
@app.websocket("/ws/knowledge-status")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 保持连接，等待心跳或消息
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 异步后台任务：解析知识库文件并推送到 WebSocket
async def process_file_background(file_id: int):
    async with async_session() as db:
        try:
            # 1. 获取文件信息
            result = await db.execute(select(KnowledgeFileModel).where(KnowledgeFileModel.id == file_id))
            db_file = result.scalars().first()
            if not db_file:
                return

            # 2. 更新状态为正在处理
            db_file.status = "processing"
            db_file.progress = 5
            await db.commit()
            await manager.broadcast({"file_id": file_id, "status": "processing", "progress": 5, "filename": db_file.filename})

            # 3. 开始解析（使用并发限制）
            async with ocr_semaphore:
                # 更新进度到 20%
                db_file.progress = 20
                await db.commit()
                await manager.broadcast({"file_id": file_id, "status": "processing", "progress": 20})

                # 调用 parse_file (由于它可能是 CPU 密集型的，我们可以在 executor 中运行)
                loop = asyncio.get_event_loop()
                documents = await loop.run_in_executor(None, parse_file, db_file.file_path)
                
                if not documents:
                    raise Exception("解析失败：未提取到有效内容")

                # 更新进度到 60%
                db_file.progress = 60
                await db.commit()
                await manager.broadcast({"file_id": file_id, "status": "processing", "progress": 60})

                # 分割文档并添加到向量库
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                all_splits = text_splitter.split_documents(documents)
                
                # 准备元数据
                safe_splits = []
                for doc in all_splits:
                    metadata = dict(getattr(doc, "metadata", {}) or {})
                    cleaned_metadata = {str(k): str(v) for k, v in metadata.items() if v is not None}
                    safe_splits.append(Document(page_content=doc.page_content, metadata=cleaned_metadata))

                # 添加到向量库
                if vectorstore:
                    await loop.run_in_executor(None, vectorstore.add_documents, safe_splits)
                
                # 更新进度到 100%
                db_file.status = "completed"
                db_file.progress = 100
                await db.commit()
                await manager.broadcast({"file_id": file_id, "status": "completed", "progress": 100})
                print(f"文件 {db_file.filename} 后台解析完成并存入向量库")

        except Exception as e:
            print(f"后台解析文件 {file_id} 失败: {str(e)}")
            # 这里需要刷新 session 重新获取对象以确保状态能更新
            try:
                result = await db.execute(select(KnowledgeFileModel).where(KnowledgeFileModel.id == file_id))
                db_file = result.scalars().first()
                if db_file:
                    db_file.status = "failed"
                    db_file.error_message = str(e)
                    await db.commit()
                    await manager.broadcast({"file_id": file_id, "status": "failed", "error": str(e)})
            except:
                pass

# 清理旧向量库（当嵌入模型变更时使用）
def clear_old_vector_store():
    """清理旧的向量库数据"""
    try:
        import shutil
        if os.path.exists(CHROMA_PERSIST_DIR):
            print(f"🗑️ 正在清理旧向量库: {CHROMA_PERSIST_DIR}")
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print("✅ 旧向量库清理完成")
        else:
            print("📁 未找到旧向量库，无需清理")
    except Exception as e:
        print(f"❌ 清理向量库失败: {str(e)}")

# 初始化RAG组件
def initialize_rag(force_rebuild=False):
    # 如果嵌入模型初始化失败，则不初始化RAG
    if embeddings is None:
        print("由于嵌入模型初始化失败，RAG组件无法初始化")
        return None
        
    try:
        # 检查是否已存在持久化数据
        if not force_rebuild and os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
            print(f"检测到已存在的向量库，正在从本地加载: {CHROMA_PERSIST_DIR}")
            try:
                vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=embeddings,
                    collection_name="qai_bot_knowledge"
                )
                print(f"成功加载本地向量库，包含约 {vectorstore._collection.count()} 个文本块")
                
                # 检查嵌入模型是否匹配
                test_vector = embeddings.embed_query("test")
                print(f"当前嵌入模型向量维度: {len(test_vector)}")
                
                return vectorstore
            except Exception as load_err:
                print(f"加载本地向量库失败，将重新构建: {str(load_err)}")
                print("⚠️ 可能是由于嵌入模型变更导致向量不兼容")
                # 强制重建向量库
                force_rebuild = True

        print("未检测到本地向量库或强制重构，开始解析文档并构建向量库...")
        # 尝试找到文件路径
        possible_file_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json"),
            "./dataset.json",
        ]
        
        dataset_json_path = ""
        for path in possible_file_paths:
            if os.path.exists(path):
                dataset_json_path = path
                break
        
        # 使用 file_parser 加载所有知识库文件
        documents = load_all_knowledge(KNOWLEDGE_DIR, dataset_json_path)

        if not documents or len(documents) == 0:
            print("警告: 知识库内容为空，无法初始化 RAG")
            return None
            
        print(f"成功加载文档，共有 {len(documents)} 个文档段落")
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_splits = text_splitter.split_documents(documents)
        
        if not all_splits or len(all_splits) == 0:
            print("警告: 文档分割后内容为空")
            return None
            
        print(f"文档分割完成，共有 {len(all_splits)} 个文本块")
        
        # 创建向量存储
        try:
            safe_splits = []
            for idx, doc in enumerate(all_splits):
                metadata = dict(getattr(doc, "metadata", {}) or {})
                cleaned_metadata = {str(k): str(v) for k, v in metadata.items() if v is not None}
                safe_splits.append(Document(page_content=doc.page_content, metadata=cleaned_metadata))

            print(f"开始创建向量存储并持久化到: {CHROMA_PERSIST_DIR}")
            vectorstore = Chroma.from_documents(
                documents=safe_splits, 
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name="qai_bot_knowledge"
            )
            print("成功创建并持久化向量存储")
            return vectorstore
            
        except Exception as e:
            print(f"创建向量存储失败: {str(e)}")
            return None
            
    except Exception as e:
        print(f"初始化RAG组件失败: {str(e)}")
        return None

# 获取当前登录用户对象并检查状态
async def get_active_user(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if user.status == "disabled":
        raise HTTPException(status_code=403, detail="Your account has been disabled")
    return user

# 管理员鉴权依赖（增强版，同时检查状态）
async def get_current_admin_v2(
    user: UserModel = Depends(get_active_user)
):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Only administrators can perform this action")
    return user

# 用户登录
@app.post("/api/login")
async def login(
    user: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """用户登录"""
    # 查找用户
    result = await db.execute(select(UserModel).where(UserModel.username == user.username))
    db_user = result.scalars().first()
    
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # 验证密码
    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # 检查用户状态
    if db_user.status == "disabled":
        raise HTTPException(status_code=403, detail="Account is disabled")
    
    # 生成访问令牌
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(db_user.id)}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": db_user.id,
            "username": db_user.username,
            "email": db_user.email,
            "role": db_user.role,
            "status": db_user.status
        }
    }

# 用户注册
@app.post("/api/register")
async def register(
    user: UserRegister,
    db: AsyncSession = Depends(get_db)
):
    # 检查用户名是否已存在
    result = await db.execute(select(UserModel).where(UserModel.username == user.username))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Username already exists")
    try:
        # 检查是否已存在持久化数据
        if not force_rebuild and os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
            print(f"检测到已存在的向量库，正在从本地加载: {CHROMA_PERSIST_DIR}")
            try:
                vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=embeddings,
                    collection_name="qai_bot_knowledge"
                )
                print(f"成功加载本地向量库，包含约 {vectorstore._collection.count()} 个文本块")
                return vectorstore
            except Exception as load_err:
                print(f"加载本地向量库失败，将重新构建: {str(load_err)}")

        print("未检测到本地向量库或强制重构，开始解析文档并构建向量库...")
        # 尝试找到文件路径
        possible_file_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json"),
            "./dataset.json",
        ]
        
        dataset_json_path = None
        for path in possible_file_paths:
            if os.path.exists(path):
                dataset_json_path = path
                break
        
        # 使用 file_parser 加载所有知识库文件
        documents = load_all_knowledge(KNOWLEDGE_DIR, dataset_json_path or "")

        if not documents or len(documents) == 0:
            print("警告: 知识库内容为空，无法初始化 RAG")
            return None
            
        print(f"成功加载文档，共有 {len(documents)} 个文档段落")
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_splits = text_splitter.split_documents(documents)
        
        if not all_splits or len(all_splits) == 0:
            print("警告: 文档分割后内容为空")
            return None
            
        print(f"文档分割完成，共有 {len(all_splits)} 个文本块")
        
        # 创建向量存储
        try:
            safe_splits = []
            for idx, doc in enumerate(all_splits):
                metadata = dict(getattr(doc, "metadata", {}) or {})
                cleaned_metadata = {str(k): str(v) for k, v in metadata.items() if v is not None}
                safe_splits.append(Document(page_content=doc.page_content, metadata=cleaned_metadata))
                if idx < 3:
                    print(f"向量化样本[{idx+1}]: 元数据键={list(cleaned_metadata.keys())}, 内容长度={len(doc.page_content)}")

            print(f"开始创建向量存储并持久化到: {CHROMA_PERSIST_DIR}")
            vectorstore = Chroma.from_documents(
                documents=safe_splits, 
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name="qai_bot_knowledge"
            )
            print("成功创建并持久化向量存储")
            return vectorstore
            
        except Exception as e:
            print(f"创建向量存储失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"初始化RAG组件失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 格式化文档
def format_docs(docs):
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)


RAG_TEMPLATE = """你是一位专业的智能辅助问答助手，名为QAI Bot。

重要指令：
1. 如果提供的“参考信息”与用户的问题【完全无关】（例如问题是关于做菜、娱乐等，而参考信息是关于学术或技术的），请【忽略】参考信息，直接利用你的通用知识回答。
2. 只有当参考信息确实能回答用户问题时，才结合参考信息进行回答。

参考信息：
{context}

聊天历史：
{history_context}

当前问题: {question}

请遵循以下回答规则：
1. 使用Markdown格式化你的回答，使其更易于阅读，例如使用标题、列表、粗体、表格等
2. 专业术语使用**加粗**或*斜体*并简要解释
3. 结构化信息使用清晰的标题层级和列表
4. 重要建议使用加粗标记
5. 保持助手的耐心和专业，提供友好的回答
6. **引导追问**：在回答结束并在末尾划定分隔线后，请基于当前对话内容提供3个相关的追问问题。追问问题请以“你可能还想了解：”作为标题，并使用 1. 2. 3. 列表形式列出。

始终保持专业、准确和有帮助的态度。"""

DEFAULT_TEMPLATE = """你是一位专业的智能辅助问答助手，名为"QAI Bot"。你的主要职责是提供准确、有用的信息咨询服务。

用户问题: {question}

请遵循以下回答规则：
1. 使用Markdown格式化你的回答，使其更易于阅读，例如使用标题、列表、粗体、表格等
2. 确保回答具备准确性和教学可读性，像老师一样循序渐进地讲解
3. 如有专业术语，可以使用斜体或加粗标记，并简单解释其含义
4. 提供清晰的结构，使用标题（#、##）分隔不同部分
5. **非常重要**：在回答的最后，请基于当前对话内容，提供3个用户可能感兴趣的相关追问问题。追问问题请以“你可能还想了解：”作为标题，并使用1. 2. 3. 列表形式列出。

回答时，请注意：
- 保持专业、准确、耐心且友好
- 结构化地呈现信息
- 对未知问题，坦诚承认自己的局限性

回答：
"""

SYSTEM_PROMPT = """你是一位专业的智能辅助问答助手，名为QAI Bot，提供准确、科学的信息。请注意：
1. 使用Markdown组织回答，包括标题、列表、表格等
2. 专业术语使用**加粗**或*斜体*并简要解释
3. 结构化信息使用清晰的标题层级和列表
4. 重要建议使用加粗标记
5. 保持助手的耐心和专业，提供友好的回答

始终保持专业、准确和有帮助的态度。"""

general_prompt = ChatPromptTemplate.from_template(DEFAULT_TEMPLATE)
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# 初始化智能问答组件
vectorstore = initialize_rag()
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    retriever = None

# 智能回答函数
async def smart_answer(question, model, chat_history=None):
    """
    根据问题和聊天历史生成智能回答
    
    Args:
        question: 用户的当前问题
        model: 要使用的语言模型
        chat_history: 可选的聊天记录列表
        
    Returns:
        tuple[str, dict]: 回答文本与来源信息
    """
    try:
        response_meta = {
            "source": "llm",
            "source_label": "仅来自于大模型"
        }
        # 准备消息列表，始终以系统提示开始
        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]
        
        # 添加聊天历史到消息列表
        if chat_history:
            # 限制历史消息数量，保留最近的10条
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            for msg in recent_history:
                # 处理不同类型的消息对象
                if isinstance(msg, dict):
                    # 如果是字典类型（来自JSON）
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                elif hasattr(msg, "role") and hasattr(msg, "content"):
                    # 如果是Message类对象
                    role = msg.role
                    content = msg.content
                else:
                    # 跳过无法处理的消息
                    print(f"警告: 无法处理的消息类型: {type(msg)}")
                    continue
                
                # 根据角色添加适当的消息
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        
        # 如果没有成功初始化检索器，则只使用模型直接回答
        source_docs = []
        if vectorstore:
            try:
                # 使用 similarity_search_with_score，它返回的是原始 L2 距离
                # 注意：距离越小越相似。对于 DashScope v3，通常距离 < 0.8 都是非常相关的
                search_results = vectorstore.similarity_search_with_score(question, k=5)  # 增加检索数量
                
                # 改进的过滤策略：
                source_docs = []
                for doc, score in search_results:
                    print(f"🔍 检索分值 (L2距离): {score:.4f} | 内容预览: {doc.page_content[:30]}...")
                    # 严格阈值：只有距离 < 0.9 的才视为相关
                    if score < 0.9: 
                        source_docs.append(doc)
                
                # 如果没有找到足够相关的文档，尝试多样性检索
                if len(source_docs) < 2 and len(search_results) >= 2:
                    print("🔄 相关文档不足，尝试多样性检索...")
                    # 取距离相对较小且内容不同的文档
                    for doc, score in search_results[2:5]:  # 检查第3-5个结果
                        if score < 1.2 and len(source_docs) < 3:
                            # 简单的内容去重检查
                            content_similar = False
                            for existing_doc in source_docs:
                                if doc.page_content[:50] in existing_doc.page_content or existing_doc.page_content[:50] in doc.page_content:
                                    content_similar = True
                                    break
                            if not content_similar:
                                source_docs.append(doc)
                                print(f"🎯 添加多样性文档: {doc.page_content[:30]}...")
                
                if not source_docs:
                    print("⚠️ 未找到相关文档，转为大模型回答")
                else:
                    print(f"✅ 找到 {len(source_docs)} 个相关文档")
                
            except Exception as e:
                print(f"检索器调用失败: {str(e)}")
                import traceback
                traceback.print_exc()

        if source_docs:
            response_meta = {
                "source": "knowledge_base",
                "source_label": "知识库"
            }
            context = format_docs(source_docs)
            print(f"找到相关文档，使用知识库回答，文档数量: {len(source_docs)}")

            history_context = ""
            if chat_history and len(chat_history) > 0:
                history_context = "\n\n聊天历史:\n"
                recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
                for msg in recent_history:
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                    elif hasattr(msg, "role") and hasattr(msg, "content"):
                        role = msg.role
                        content = msg.content
                    else:
                        continue
                    role_name = "用户" if role == "user" else "AI助手"
                    history_context += f"{role_name}: {content}\n"

            try:
                print(f"准备调用 RAG 链，模型类型: {type(model)}")
                rag_chain = rag_prompt | model | StrOutputParser()
                print("RAG 链创建成功")
                
                answer = await rag_chain.ainvoke({
                    "context": context,
                    "question": question,
                    "history_context": history_context
                })
                print(f"RAG 链调用成功，回答长度: {len(answer) if answer else 0}")
                
                # 提取引用的文档信息
                matched_docs = []
                seen_filenames = set()
                for doc in source_docs:
                    raw_filename = os.path.basename(doc.metadata.get("source", "未知文档"))
                    # 去除 UUID 前缀 (例如: 85b70fd7-806c-46bd-ace0-ca513485b0da_文件名.pdf)
                    # 假设 UUID 长度加下划线至少为 37 位
                    if len(raw_filename) > 37 and "_" in raw_filename[:40]:
                        display_filename = raw_filename.split("_", 1)[1]
                    else:
                        display_filename = raw_filename
                        
                    if display_filename not in seen_filenames:
                        matched_docs.append({
                            "filename": display_filename,
                            "content_preview": doc.page_content[:100] + "..."
                        })
                        seen_filenames.add(display_filename)
                
                response_meta["matched_docs"] = matched_docs
                return answer, response_meta
            except Exception as e:
                print(f"RAG链调用失败，回退到大模型: {str(e)}")

        messages.append(HumanMessage(content=question))
        print(f"未命中知识库，使用大模型回答，历史消息数量: {len(messages)-1}")
        try:
            print(f"准备调用模型，消息数量: {len(messages)}")
            print(f"模型对象: {type(model)}")
            
            # ChatOpenAI 可以直接使用 LangChain 消息格式
            response = model.invoke(messages)
            print(f"模型响应类型: {type(response)}")
            print(f"模型响应内容: {response}")
            
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
                
            response_meta = {
                "source": "llm",
                "source_label": "仅来自于大模型"
            }
            return answer, response_meta
        except Exception as e:
            print(f"模型调用失败: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return _generate_fallback_response(question, chat_history), {
                "source": "fallback",
                "source_label": "仅来自于大模型"
            }
    except Exception as e:
        print(f"智能回答生成出错: {str(e)}")
        return _generate_fallback_response(question, chat_history), {
            "source": "fallback",
            "source_label": "仅来自于大模型"
        }

# 后备回答生成函数
def _generate_fallback_response(question, chat_history=None):
    """当API调用失败时生成的后备回答"""
    # 检查问题是否询问之前的对话内容
    memory_keywords = ["之前", "刚才", "上面", "之前问", "刚问", "之前聊", "刚才说", "刚刚问"]
    history_question = any(keyword in question for keyword in memory_keywords)
    
    if history_question and chat_history and len(chat_history) > 0:
        # 尝试从历史记录中提取最近的1-2条用户消息
        recent_user_msgs = []
        for msg in reversed(chat_history):
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                role = msg.role
                content = msg.content
            else:
                continue
            role_name = "用户" if role == "user" else "AI助手"
            recent_user_msgs.append(f"{role_name}: {content}")
        
        if recent_user_msgs:
            history_summary = "、".join(recent_user_msgs[:2][::-1])
            return f"""根据我的记忆，您之前问了关于"{history_summary}"的问题。

很抱歉，我目前遇到了一些技术问题，无法提供完整的回答。请稍后再试，或者重新表述您的问题，我会尽力帮助您。"""
    
    # 通用后备回答
    return """很抱歉，我目前遇到了一些技术问题，无法处理您的请求。这可能是由于以下原因：

1. 服务器负载过高
2. API调用限制
3. 网络连接问题

请稍后再试，或者重新表述您的问题，我会尽力帮助您。如果问题持续存在，请联系技术支持。

感谢您的理解。"""

# 获取对话ID的依赖
def get_conversation_id(request: Request) -> str:
    """从请求中获取会话ID, 如果没有则创建新的"""
    # 先尝试从查询参数获取会话ID
    conversation_id = request.query_params.get("conversation_id")
    
    # 如果查询参数中没有，再尝试从头部获取
    if not conversation_id:
        conversation_id = request.headers.get("X-Conversation-ID")
    
    # 如果没有找到会话ID，生成一个新的
    if not conversation_id:
        conversation_id = f"sess_{int(datetime.now().timestamp())}_{os.urandom(4).hex()}"
    
    # 如果是新的会话ID，初始化
    if conversation_id not in conversation_store:
        print(f"创建新会话: {conversation_id}")
        conversation_store[conversation_id] = {
            "messages": []
        }
    
    return conversation_id

@app.post("/api/chat/stream")
@app.get("/api/chat/stream")
async def chat_stream(
    request: ChatRequest = None, 
    conversation_id: str = Depends(get_conversation_id),
    token: Optional[str] = None,
    request_raw: Request = None
):
    """流式聊天API端点 - 支持POST和GET请求"""
    try:
        # 严格获取用户ID
        user_id = None
        auth_token = token
        
        if request_raw and "Authorization" in request_raw.headers:
            auth_header = request_raw.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                auth_token = auth_header.split(" ")[1]
        
        if auth_token:
            user_id = await get_current_user_id_optional(auth_token)

        if not user_id:
            print(f"❌ 拒绝流式请求: 会话 {conversation_id} 未提供有效 Token")
            return JSONResponse(status_code=401, content={"error": "Not authenticated"})

        effective_user_id = user_id

        # 如果是GET请求，尝试从查询参数获取消息，用于EventSource
        message = ""
        chat_history = []
        
        if request_raw and request_raw.method == "GET":
            # 从会话ID尝试获取最后一条用户消息
            if conversation_id in conversation_store and conversation_store[conversation_id]["messages"]:
                messages = conversation_store[conversation_id]["messages"]
                chat_history = messages.copy()  # 保存整个历史
                user_messages = [msg for msg in messages if msg["role"] == "user"]
                if user_messages:
                    message = user_messages[-1]["content"]
                    print(f"GET请求使用会话 {conversation_id} 的最后一条用户消息: '{message}'")
                    print(f"聊天历史包含 {len(chat_history)} 条消息")
                else:
                    print(f"警告: 会话 {conversation_id} 没有用户消息")
            else:
                print(f"警告: 未找到会话 {conversation_id} 或会话为空")
        elif request:
            message = request.message
            
            # 处理聊天历史，可能是Pydantic模型或已经是字典列表
            if request.chat_history:
                # 转换Message对象为字典
                chat_history = []
                for msg in request.chat_history:
                    if isinstance(msg, dict):
                        chat_history.append(msg)
                    elif hasattr(msg, "role") and hasattr(msg, "content"):
                        chat_history.append({
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp if hasattr(msg, "timestamp") else None
                        })
            
            print(f"POST请求接收到的消息: '{message}'")
            print(f"消息历史长度: {len(chat_history)}")
        else:
            print("警告: 未能获取到消息内容，使用空字符串")
        
        # 打印接收到的请求信息
        print(f"处理流式请求 - 消息: '{message}', 会话ID: {conversation_id}")
        print(f"请求方法: {request_raw.method if request_raw else 'POST'}")
        
        # 如果是POST请求，我们将消息存入会话存储中并直接返回
        # 这样随后的GET请求（EventSource）就可以从存储中读取消息并开始流式传输
        if request_raw and request_raw.method == "POST":
            if conversation_id not in conversation_store:
                conversation_store[conversation_id] = {"messages": []}

            # 只有当消息不为空且不是重复添加时才记录
            if message:
                # 检查是否已经是最后一条用户消息
                messages = conversation_store[conversation_id]["messages"]
                if not messages or messages[-1]["role"] != "user" or messages[-1]["content"] != message:
                    conversation_store[conversation_id]["messages"].append({
                        "role": "user",
                        "content": message,
                        "timestamp": datetime.now().isoformat()
                    })
                    print(f"POST请求: 已将用户消息保存到会话 {conversation_id}")

            return {"status": "message_received", "conversation_id": conversation_id}

        if conversation_id not in conversation_store:
            conversation_store[conversation_id] = {"messages": []}

        # 创建一个异步生成器来流式传输聊天响应
        async def event_generator():
            # 在生成器内部声明非局部变量，确保可以访问外部作用域的变量
            nonlocal message, chat_history, conversation_id
            
            try:
                if not message:
                    # 如果没有消息，发送简单提示
                    yield {
                        "event": "message",
                        "data": "请"
                    }
                    await asyncio.sleep(0.05)
                    yield {
                        "event": "message",
                        "data": "输"
                    }
                    await asyncio.sleep(0.05)
                    yield {
                        "event": "message",
                        "data": "入"
                    }
                    await asyncio.sleep(0.05)
                    yield {
                        "event": "message",
                        "data": "您"
                    }
                    await asyncio.sleep(0.05)
                    yield {
                        "event": "message",
                        "data": "的"
                    }
                    await asyncio.sleep(0.05)
                    yield {
                        "event": "message",
                        "data": "问"
                    }
                    await asyncio.sleep(0.05)
                    yield {
                        "event": "message",
                        "data": "题"
                    }
                    
                    # 发送完成事件
                    yield {
                        "event": "done",
                        "data": json.dumps({"message": "Stream completed", "conversation_id": conversation_id})
                    }
                    return
                
                print(f"开始处理流式响应...")
                try:
                    print("正在初始化 ChatOpenAI 模型...")
                    print(f"API Key 状态: {'已设置' if DASHSCOPE_API_KEY else '未设置'}")
                    model = ChatOpenAI(
                        api_key=DASHSCOPE_API_KEY,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        model_name="qwen3.6-flash"
                    )
                    print("模型初始化成功")
                    
                    # 调用智能回答
                    response_result = await smart_answer(message, model, chat_history)
                    if isinstance(response_result, tuple):
                        full_response, response_meta = response_result
                    else:
                        full_response = response_result
                        
                    # 逐字发送
                    for char in full_response:
                        yield {
                            "event": "message",
                            "data": char
                        }
                        await asyncio.sleep(0.01)
                        
                    # --- 关键：先持久化到数据库，再发送结束元数据 ---
                    async with async_session() as db_session:
                        user_history = ChatHistoryModel(
                            conversation_id=conversation_id,
                            user_id=effective_user_id,
                            role="user",
                            content=message,
                            timestamp=datetime.utcnow()
                        )
                        assistant_history = ChatHistoryModel(
                            conversation_id=conversation_id,
                            user_id=effective_user_id,
                            role="assistant",
                            content=full_response,
                            source_label=response_meta.get("source_label"),
                            matched_docs=json.dumps(response_meta.get("matched_docs")) if response_meta.get("matched_docs") else None,
                            timestamp=datetime.utcnow() + timedelta(milliseconds=10)
                        )
                        db_session.add(user_history)
                        db_session.add(assistant_history)
                        await db_session.commit()
                        print(f"会话 {conversation_id} 已成功持久化到数据库")
                        
                except Exception as model_error:
                    print(f"模型初始化或调用失败: {str(model_error)}")
                    print(f"错误类型: {type(model_error).__name__}")
                    import traceback
                    traceback.print_exc()
                    
                    # 发送错误信息
                    error_message = "模型服务暂时不可用，请稍后重试。"
                    for char in error_message:
                        yield {
                            "event": "message",
                            "data": char
                        }
                        await asyncio.sleep(0.01)
                    
                    yield {
                        "event": "done",
                        "data": json.dumps({
                            "message": "Error occurred", 
                            "conversation_id": conversation_id,
                            "error": str(model_error)
                        })
                    }
                    return

                    # 更新内存记录助手消息
                    if conversation_id not in conversation_store:
                        conversation_store[conversation_id] = {"messages": []}
                        
                    conversation_store[conversation_id]["messages"].append({
                        "role": "user",
                        "content": message,
                        "timestamp": datetime.now().isoformat()
                    })
                    conversation_store[conversation_id]["messages"].append({
                        "role": "assistant",
                        "content": full_response,
                        "source_label": response_meta.get("source_label"),
                        "matched_docs": response_meta.get("matched_docs"),
                        "timestamp": datetime.now().isoformat()
                    })

                    # 发送结束元数据
                    yield {
                        "event": "done",
                        "data": json.dumps({
                            "conversation_id": conversation_id,
                            "source_label": response_meta.get("source_label"),
                            "matched_docs": response_meta.get("matched_docs", [])
                        })
                    }

                except Exception as e:
                    error_msg = str(e)
                    # 即使模型报错，也尝试保存一条错误记录，防止会话“消失”
                    async with async_session() as db_session:
                        fail_history = ChatHistoryModel(
                            conversation_id=conversation_id,
                            user_id=effective_user_id,
                            role="assistant",
                            content=f"抱歉，系统暂时无法响应: {error_msg}",
                            timestamp=datetime.utcnow()
                        )
                        db_session.add(fail_history)
                        await db_session.commit()
                    yield {"event": "error", "data": json.dumps({"error": error_msg})}

            except Exception as e:
                error_msg = f"流式响应错误: {str(e)}"
                print(error_msg)
                print(f"错误详情: {type(e).__name__}, {e.__traceback__.tb_lineno}")
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)})
                }
        
        # 使用EventSourceResponse正确构造SSE响应
        print("创建EventSourceResponse...")
        response = EventSourceResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive", 
                "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
                "Access-Control-Allow-Origin": "*",  # 允许跨域
            }
        )
        print("返回EventSourceResponse")
        return response
        
    except Exception as e:
        error_msg = f"流式聊天API端点错误: {str(e)}"
        print(error_msg)
        print(f"错误详情: {type(e).__name__}, {e.__traceback__.tb_lineno}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@app.get("/api/history")
async def get_all_conversations(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取用户会话列表（含标题、最后消息、更新时间）"""
    result = await db.execute(
        select(ChatHistoryModel)
        .where(ChatHistoryModel.user_id == user_id)
        .order_by(ChatHistoryModel.timestamp.asc(), ChatHistoryModel.id.asc())
    )
    history_rows = result.scalars().all()

    conversation_map = {}
    for row in history_rows:
        conv_id = row.conversation_id
        if conv_id not in conversation_map:
            conversation_map[conv_id] = {
                "conversation_id": conv_id,
                "title": None,
                "last_message": "",
                "updated_at": None,
                "messages": []
            }

        matched_docs = None
        if row.matched_docs:
            try:
                matched_docs = json.loads(row.matched_docs)
            except Exception:
                matched_docs = None

        # 获取该消息的反馈状态
        fb_result = await db.execute(
            select(FeedbackModel.feedback_type).where(
                FeedbackModel.message_id == row.id,
                FeedbackModel.user_id == user_id
            )
        )
        user_feedback = fb_result.scalar()

        msg_obj = {
            "id": row.id,
            "role": row.role,
            "content": row.content,
            "image_url": row.image_url,
            "source_label": row.source_label,
            "matched_docs": matched_docs,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "feedback": user_feedback
        }
        conversation_map[conv_id]["messages"].append(msg_obj)

        # 始终使用最新的用户消息作为标题
        if row.role == "user":
            conversation_map[conv_id]["title"] = row.content[:50]
        elif conversation_map[conv_id]["title"] is None:
            # 如果目前还没有标题（例如第一条消息是助手的欢迎语），则暂用助手内容
            conversation_map[conv_id]["title"] = row.content[:50]

        if row.content:
            conversation_map[conv_id]["last_message"] = row.content[:80]

        if row.timestamp:
            conversation_map[conv_id]["updated_at"] = row.timestamp

    for conv_id, conv_data in conversation_store.items():
        messages = conv_data.get("messages", [])
        if not messages:
            continue
        if conv_id not in conversation_map:
            conversation_map[conv_id] = {
                "conversation_id": conv_id,
                "title": None,
                "last_message": "",
                "updated_at": None,
                "messages": []
            }
        summary = conversation_map[conv_id]
        user_messages = [m for m in messages if m.get("role") == "user" and m.get("content")]
        last_messages = [m for m in messages if m.get("content")]
        # 始终使用最新的用户消息作为标题
        if user_messages:
            summary["title"] = user_messages[-1]["content"][:50]
        elif not summary["title"] and last_messages:
            # 如果没有用户消息且没有预设标题，则使用最后一条消息
            summary["title"] = last_messages[-1]["content"][:50]

        if last_messages:
            summary["last_message"] = last_messages[-1]["content"][:80]
            ts = last_messages[-1].get("timestamp")
            if ts:
                try:
                    summary["updated_at"] = datetime.fromisoformat(ts)
                except Exception:
                    pass
        for msg in messages:
            summary["messages"].append({
                "id": None,
                "role": msg.get("role"),
                "content": msg.get("content", ""),
                "image_url": msg.get("image_url"),
                "source_label": msg.get("source_label"),
                "matched_docs": msg.get("matched_docs"),
                "timestamp": msg.get("timestamp"),
                "feedback": msg.get("feedback")
            })

    conversations = list(conversation_map.values())
    for item in conversations:
        if not item["title"]:
            item["title"] = "新会话"
        if item["updated_at"]:
            updated_at = item["updated_at"]
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            item["updated_at"] = updated_at.isoformat()

    conversations.sort(
        key=lambda x: x["updated_at"] or "",
        reverse=True
    )

    print(f"历史会话列表返回 {len(conversations)} 条记录")
    return {
        "conversation_ids": [item["conversation_id"] for item in conversations],
        "conversations": conversations
    }

@app.get("/api/history/{conversation_id}")
async def get_history(
    conversation_id: str, 
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """从数据库获取特定会话的历史记录"""
    result = await db.execute(
        select(ChatHistoryModel)
        .where(ChatHistoryModel.conversation_id == conversation_id)
        .where(ChatHistoryModel.user_id == user_id)
        .order_by(ChatHistoryModel.timestamp.asc(), ChatHistoryModel.id.asc())
    )
    history = result.scalars().all()

    if history:
        conversation_store[conversation_id] = {
            "messages": []
        }

    formatted_history = []
    for msg in history:
        # 获取该消息的反馈状态
        fb_result = await db.execute(
            select(FeedbackModel.feedback_type).where(
                FeedbackModel.message_id == msg.id,
                FeedbackModel.user_id == user_id
            )
        )
        user_feedback = fb_result.scalar()

        formatted_history.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": (
                msg.timestamp.replace(tzinfo=timezone.utc).isoformat()
                if msg.timestamp and msg.timestamp.tzinfo is None
                else (msg.timestamp.isoformat() if msg.timestamp else None)
            ),
            "image_url": msg.image_url,
            "source_label": msg.source_label,
            "matched_docs": json.loads(msg.matched_docs) if msg.matched_docs else None,
            "feedback": user_feedback
        })
        conversation_store[conversation_id]["messages"].append(formatted_history[-1])
    
    return {
        "history": formatted_history,
        "conversation_id": conversation_id
    }

@app.delete("/api/history/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """删除特定会话的所有历史记录"""
    try:
        # 执行删除操作
        await db.execute(
            delete(ChatHistoryModel)
            .where(ChatHistoryModel.conversation_id == conversation_id)
            .where(ChatHistoryModel.user_id == user_id)
        )
        
        # 同时从内存中删除
        if conversation_id in conversation_store:
            del conversation_store[conversation_id]
            print(f"会话 {conversation_id} 已从内存中删除")
            
        await db.commit()
        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


# --- 知识库管理模块开始 ---

@app.get("/api/admin/knowledge-files", response_model=List[KnowledgeFileResponse])
async def get_knowledge_files(
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    """管理员：获取所有知识库文件"""
    result = await db.execute(select(KnowledgeFileModel).order_by(KnowledgeFileModel.created_at.desc()))
    return result.scalars().all()

@app.post("/api/admin/upload-knowledge")
async def upload_knowledge(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    """管理员：上传知识库文件并启动后台解析"""
    uploaded_files = []
    
    for file in files:
        # 1. 保存物理文件
        file_ext = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(KNOWLEDGE_DIR, unique_filename)
        
        try:
            content = await file.read()
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # 2. 写入数据库记录
            new_file = KnowledgeFileModel(
                filename=file.filename,
                file_path=file_path,
                file_type=file_ext,
                file_size=len(content),
                status="pending",
                progress=0
            )
            db.add(new_file)
            await db.commit()
            await db.refresh(new_file)
            
            # 3. 启动后台任务解析
            background_tasks.add_task(process_file_background, new_file.id)
            uploaded_files.append({"filename": file.filename, "file_id": new_file.id})
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            print(f"上传文件 {file.filename} 失败: {str(e)}")
            continue
            
    return {"message": f"Successfully uploaded {len(uploaded_files)} files", "files": uploaded_files}

@app.delete("/api/admin/knowledge-files/{file_id}")
async def delete_knowledge_file(
    file_id: int,
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    """管理员：删除知识库文件及其向量库条目"""
    # 1. 获取文件记录
    result = await db.execute(select(KnowledgeFileModel).where(KnowledgeFileModel.id == file_id))
    db_file = result.scalars().first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        # 2. 从向量库中删除相关条目
        if vectorstore:
            try:
                # 使用 metadata 中的 source 进行过滤删除
                # 在 initialize_rag 中，source 存储的是物理路径
                vectorstore.delete(where={"source": db_file.file_path})
                print(f">>> 已从向量库中删除文件相关条目: {db_file.filename}")
            except Exception as ve:
                print(f">>> 从向量库删除条目失败: {str(ve)}")

        # 3. 删除物理文件
        if os.path.exists(db_file.file_path):
            os.remove(db_file.file_path)
            
        # 4. 删除数据库记录
        await db.delete(db_file)
        await db.commit()
        
        return {"message": "File deleted successfully"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

# --- 知识库管理模块结束 ---


# --- 语义聚类逻辑开始 ---

import re

def preprocess_text(text: str) -> str:
    """文本预处理：去除标点、统一大小写、去除空格"""
    # 去除中英文标点
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
    # 统一大小写并去除空格
    return text.lower().strip()

async def update_question_clusters():
    """离线任务：更新问题聚类结果 (改进的语义聚类 + LLM 标题提炼)"""
    print("\n>>> [语义聚类] 开始执行增强版语义聚类任务...")
    
    # 检查嵌入模型
    if embeddings is None:
        print(">>> [语义聚类] 嵌入模型不可用，跳过任务")
        return

    async with async_session() as db:
        try:
            # 1. 获取所有用户问题 (长度大于2)
            result = await db.execute(
                select(ChatHistoryModel.content)
                .where(ChatHistoryModel.role == "user", func.length(ChatHistoryModel.content) > 2)
            )
            raw_questions = [row[0] for row in result.all()]
            
            if not raw_questions:
                print(">>> [语义聚类] 无有效问题数据，跳过任务")
                return

            # 2. 向量化所有问题
            print(f">>> [语义聚类] 正在对 {len(raw_questions)} 条问题进行向量化...")
            loop = asyncio.get_event_loop()
            question_vectors = await loop.run_in_executor(None, embeddings.embed_documents, raw_questions)
            
            # 3. 改进的聚类算法 (固定中心法防止漂移)
            SIMILARITY_THRESHOLD = 0.92  # 提高阈值，确保更精准的归类
            clusters = [] # List of dict: {"center_vector": np.array, "questions": [str], "count": int}

            for q_text, q_vec in zip(raw_questions, question_vectors):
                q_vec = np.array(q_vec)
                matched_idx = -1
                max_sim = -1
                
                # 寻找最相似的已有类簇中心
                for idx, cluster in enumerate(clusters):
                    center_vec = cluster["center_vector"]
                    # 计算余弦相似度
                    dot_product = np.dot(q_vec, center_vec)
                    norm_a = np.linalg.norm(q_vec)
                    norm_b = np.linalg.norm(center_vec)
                    similarity = dot_product / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0
                    
                    if similarity > SIMILARITY_THRESHOLD and similarity > max_sim:
                        max_sim = similarity
                        matched_idx = idx
                
                if matched_idx != -1:
                    # 归入已有类簇，但不更新中心向量，防止类簇漂移（核心成员决定类簇）
                    clusters[matched_idx]["questions"].append(q_text)
                    clusters[matched_idx]["count"] += 1
                else:
                    # 创建新类簇，将第一个加入的成员作为永久中心
                    clusters.append({
                        "center_vector": q_vec,
                        "questions": [q_text],
                        "count": 1
                    })

            # 4. LLM 标题提炼与结果整合
            print(f">>> [语义聚类] 聚类完成，识别出 {len(clusters)} 个初步类簇，开始提炼核心意图...")
            
            # 按规模排序，优先处理规模大的类簇
            clusters.sort(key=lambda x: x["count"], reverse=True)
            top_clusters = clusters[:20]  # 取前 20 个
            
            cluster_results = []
            model = ChatOpenAI(
                api_key=DASHSCOPE_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model_name="qwen3.6-flash"
            )
            
            for idx, c in enumerate(top_clusters):
                # 提取唯一示例
                unique_examples = list(set(c["questions"]))
                sample_for_llm = unique_examples[:8]  # 取前 8 条给 LLM 参考
                
                # 利用 LLM 总结核心意图
                prompt = f"""以下是用户的一组相似提问示例，请你总结出它们共同的核心意图作为标题。
要求：
1. 语言极其简练，通常在 5-10 个字之间。
2. 能够准确覆盖示例的主旨。
3. 直接返回标题，不要有任何解释。

提问示例：
{chr(10).join(['- ' + q for q in sample_for_llm])}

核心意图标题："""
                
                try:
                    response = model.invoke(prompt)
                    represent = response.content.strip()
                    # 去除引号或句号
                    represent = represent.replace('"', '').replace('"', '').replace('"', '').replace('。', '')
                    print(f"   - 类簇 {idx+1}: '{represent}' (成员数: {c['count']})")
                except Exception as e:
                    print(f"   - 类簇 {idx+1}: LLM 提炼失败，使用首条提问。错误: {str(e)}")
                    represent = c["questions"][0][:30]

                cluster_results.append({
                    "represent_question": represent,
                    "count": c["count"],
                    "examples": unique_examples[:10]  # 保留 10 个示例供展示
                })

            # 5. 清理旧数据并写入新数据
            await db.execute(delete(QuestionClusterModel))
            for res in cluster_results:
                new_cluster_row = QuestionClusterModel(
                    represent_question=res["represent_question"],
                    count=res["count"],
                    examples=json.dumps(res["examples"], ensure_ascii=False)
                )
                db.add(new_cluster_row)
            
            await db.commit()
            print(f">>> [语义聚类] 增强版任务完成，已更新 {len(cluster_results)} 个意图分类。")
            
        except Exception as e:
            await db.rollback()
            print(f">>> [语义聚类] 任务异常失败: {str(e)}")
            import traceback
            traceback.print_exc()

# --- 语义聚类逻辑结束 ---


# --- 反馈模块开始 ---
@app.post("/api/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """提交点赞/点踩反馈"""
    print(f"\n>>> [反馈请求] 用户ID: {user_id}, 消息ID: {request.message_id}, 类型: {request.feedback_type}")
    
    if request.feedback_type not in ["like", "dislike"]:
        print(">>> [反馈结果] 失败: 无效的反馈类型")
        raise HTTPException(status_code=400, detail="Invalid feedback type")
        
    # 检查消息是否存在
    result = await db.execute(select(ChatHistoryModel).where(ChatHistoryModel.id == request.message_id))
    message = result.scalars().first()
    if not message:
        print(f">>> [反馈结果] 失败: 消息 {request.message_id} 不存在")
        raise HTTPException(status_code=404, detail="Message not found")

    # 检查是否已有反馈
    result = await db.execute(
        select(FeedbackModel).where(
            FeedbackModel.user_id == user_id,
            FeedbackModel.message_id == request.message_id
        )
    )
    existing_feedback = result.scalars().first()

    if existing_feedback:
        # 如果已有相同反馈，则取消（Toggle）
        if existing_feedback.feedback_type == request.feedback_type:
            print(f">>> [反馈结果] 成功: 用户取消了之前的 {request.feedback_type}")
            await db.delete(existing_feedback)
            await db.commit()
            return {"message": "Feedback removed", "feedback_type": None}
        else:
            # 更新反馈类型
            old_type = existing_feedback.feedback_type
            existing_feedback.feedback_type = request.feedback_type
            existing_feedback.created_at = datetime.utcnow()
            print(f">>> [反馈结果] 成功: 将反馈从 {old_type} 修改为 {request.feedback_type}")
    else:
        # 创建新反馈
        new_feedback = FeedbackModel(
            user_id=user_id,
            message_id=request.message_id,
            feedback_type=request.feedback_type
        )
        db.add(new_feedback)
        print(f">>> [反馈结果] 成功: 新增 {request.feedback_type} 反馈")

    await db.commit()
    return {"message": "Feedback submitted successfully", "feedback_type": request.feedback_type}

@app.get("/api/feedback/stat", response_model=FeedbackStatResponse)
async def get_feedback_stat(
    message_id: int,
    user_id: Optional[int] = Depends(get_current_user_id_optional),
    db: AsyncSession = Depends(get_db)
):
    """获取单条消息的反馈统计"""
    # 统计点赞数
    result = await db.execute(
        select(func.count()).where(
            FeedbackModel.message_id == message_id,
            FeedbackModel.feedback_type == "like"
        )
    )
    like_count = result.scalar() or 0

    # 统计点踩数
    result = await db.execute(
        select(func.count()).where(
            FeedbackModel.message_id == message_id,
            FeedbackModel.feedback_type == "dislike"
        )
    )
    dislike_count = result.scalar() or 0

    # 获取当前用户的反馈状态
    user_feedback = None
    if user_id:
        result = await db.execute(
            select(FeedbackModel.feedback_type).where(
                FeedbackModel.user_id == user_id,
                FeedbackModel.message_id == message_id
            )
        )
        user_feedback = result.scalar()

    return {
        "like_count": like_count,
        "dislike_count": dislike_count,
        "user_feedback": user_feedback
    }

@app.get("/api/admin/feedback/overview", response_model=List[FeedbackOverviewItem])
async def get_feedback_overview(
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    """管理员：获取所有回答的反馈概览"""
    # 子查询：统计每条消息的点赞数
    likes_sub = select(
        FeedbackModel.message_id,
        func.count(FeedbackModel.id).label("likes")
    ).where(FeedbackModel.feedback_type == "like").group_by(FeedbackModel.message_id).subquery()

    # 子查询：统计每条消息的点踩数
    dislikes_sub = select(
        FeedbackModel.message_id,
        func.count(FeedbackModel.id).label("dislikes")
    ).where(FeedbackModel.feedback_type == "dislike").group_by(FeedbackModel.message_id).subquery()

    # 查询助手发出的所有消息及其反馈统计
    query = select(
        ChatHistoryModel.id,
        ChatHistoryModel.content.label("answer"),
        func.coalesce(likes_sub.c.likes, 0).label("like_count"),
        func.coalesce(dislikes_sub.c.dislikes, 0).label("dislike_count")
    ).outerjoin(
        likes_sub, ChatHistoryModel.id == likes_sub.c.message_id
    ).outerjoin(
        dislikes_sub, ChatHistoryModel.id == dislikes_sub.c.message_id
    ).where(
        ChatHistoryModel.role == "assistant"
    ).order_by(ChatHistoryModel.timestamp.desc())

    result = await db.execute(query)
    items = []
    for row in result:
        # 寻找对应的用户提问
        q_result = await db.execute(
            select(ChatHistoryModel.content, ChatHistoryModel.conversation_id).where(
                ChatHistoryModel.id < row.id,
                ChatHistoryModel.role == "user"
            ).order_by(ChatHistoryModel.id.desc()).limit(1)
        )
        q_row = q_result.first()
        question = q_row[0] if q_row else "未知提问"
        
        items.append({
            "message_id": row.id,
            "question": question,
            "answer": row.answer,
            "like_count": row.like_count,
            "dislike_count": row.dislike_count
        })
    
    return items

@app.get("/api/admin/hot-questions", response_model=List[HotQuestionItem])
async def get_hot_questions(
    background_tasks: BackgroundTasks,
    force_update: bool = False,
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    """管理员：高频问题统计（从聚类表读取）"""
    # 如果强制更新，直接启动后台任务
    if force_update:
        background_tasks.add_task(update_question_clusters)
        return JSONResponse(status_code=202, content={"message": "Clustering task started"})

    result = await db.execute(
        select(QuestionClusterModel)
        .order_by(QuestionClusterModel.count.desc())
        .limit(20)
    )
    clusters = result.scalars().all()
    
    # 如果表中没数据，启动后台任务异步计算，不阻塞当前请求
    if not clusters:
        background_tasks.add_task(update_question_clusters)
        return []
    
    return [
        {
            "represent_question": c.represent_question,
            "count": c.count,
            "examples": json.loads(c.examples) if c.examples else []
        }
        for c in clusters
    ]

@app.get("/api/admin/knowledge-analysis", response_model=KnowledgeAnalysisResponse)
async def get_knowledge_analysis(
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    """管理员：知识库覆盖率分析 (与高频问题联动)"""
    # 1. 获取核心关键词列表
    core_keywords = [
        "人工智能", "机器学习", "深度学习", "神经网络", "大语言模型", 
        "RAG", "向量数据库", "Transformer", "自然语言处理", "计算机视觉",
        "监督学习", "无监督学习", "强化学习", "模型训练", "推理"
    ]
    
    # 2. 获取统计数据
    result_files = await db.execute(select(func.count(KnowledgeFileModel.id)))
    total_files = result_files.scalar() or 0
    
    # 获取高频问题聚类数据用于联动分析
    result_hot = await db.execute(select(QuestionClusterModel))
    hot_clusters = result_hot.scalars().all()
    
    # 从向量库获取总块数
    total_chunks = 0
    if vectorstore:
        try:
            total_chunks = vectorstore._collection.count()
        except:
            pass
            
    keyword_details = []
    covered_count = 0
    
    # 3. 关键词覆盖分析 (优先使用向量库进行语义搜索检查)
    for kw in core_keywords:
        is_covered = False
        source_count = 0
        hot_count = 0
        
        # 统计高频问题中提及该关键词的次数 (语义匹配)
        for cluster in hot_clusters:
            if kw.lower() in cluster.represent_question.lower():
                hot_count += cluster.count
        
        if vectorstore:
            try:
                # 增强检索：使用组合词提高匹配成功率（模拟真实对话提问）
                search_queries = [kw, f"什么是{kw}", f"{kw}的概念和原理"]
                best_valid_results = []
                
                for query in search_queries:
                    search_results = vectorstore.similarity_search_with_score(query, k=3)
                    # 将判定阈值从 1.0 放宽至 1.25，与对话检索逻辑保持一致
                    valid_results = [res for res, score in search_results if score < 1.25]
                    if len(valid_results) > len(best_valid_results):
                        best_valid_results = valid_results
                
                source_count = len(best_valid_results)
                is_covered = source_count > 0
            except:
                pass
        
        # 联动优先级逻辑：
        # 1. 未覆盖且高频问题多 -> Urgent (紧急)
        # 2. 未覆盖但无高频问题 -> High (建议补充)
        # 3. 已覆盖但高频问题多 -> Medium (优化质量)
        # 4. 已覆盖且无高频问题 -> Low (良好)
        
        if not is_covered:
            if hot_count > 5:
                priority = "urgent"
                action = " 极度紧缺！高频提问且无知识覆盖"
            else:
                priority = "high"
                action = "建议补充相关基础文档"
        else:
            covered_count += 1
            if hot_count > 5:
                priority = "medium"
                action = "覆盖中等，建议针对高频提问优化内容"
            else:
                priority = "low"
                action = "覆盖良好"
            
        keyword_details.append(KeywordCoverage(
            keyword=kw,
            is_covered=is_covered,
            source_count=source_count,
            hot_count=hot_count,
            suggested_action=action,
            priority=priority
        ))
        
    # 4. 排序逻辑：未覆盖的优先显示，其次按高频提问次数降序，最后按优先级
    keyword_details.sort(key=lambda x: (x.is_covered, -x.hot_count))
    
    coverage_rate = (covered_count / len(core_keywords)) * 100 if core_keywords else 0
    
    return KnowledgeAnalysisResponse(
        total_files=total_files,
        total_chunks=total_chunks,
        coverage_rate=round(coverage_rate, 2),
        keyword_details=keyword_details
    )


# --- 用户管理模块开始 ---

@app.get("/api/admin/users")
async def get_all_users(
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    """管理员：获取所有用户列表"""
    result = await db.execute(select(UserModel).order_by(UserModel.created_at.desc()))
    users = result.scalars().all()
    
    return [
        {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "status": user.status,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
        for user in users
    ]

@app.put("/api/admin/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    request: dict,
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    """管理员：更新用户状态（激活/禁用）"""
    new_status = request.get("status")
    if new_status not in ["active", "disabled"]:
        raise HTTPException(status_code=400, detail="Invalid status")
        
    result = await db.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    user.status = new_status
    await db.commit()
    return {"message": f"User status updated to {new_status}"}

# --- 用户管理模块结束 ---


# --- 用户个人信息维护模块开始 ---

@app.get("/api/user/me")
async def get_my_info(
    user: UserModel = Depends(get_active_user)
):
    """获取当前登录用户信息"""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "status": user.status,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }

@app.put("/api/user/update")
async def update_my_info(
    request: dict,
    user: UserModel = Depends(get_active_user),
    db: AsyncSession = Depends(get_db)
):
    """修改当前用户信息"""
    username = request.get("username")
    email = request.get("email")
    
    if username:
        # 检查用户名是否冲突
        result = await db.execute(select(UserModel).where(UserModel.username == username, UserModel.id != user.id))
        if result.scalars().first():
            raise HTTPException(status_code=400, detail="Username already taken")
        user.username = username
        
    if email:
        user.email = email
        
    await db.commit()
    return {"message": "User info updated successfully"}

@app.post("/api/user/change-password")
async def change_my_password(
    request: dict,
    user: UserModel = Depends(get_active_user),
    db: AsyncSession = Depends(get_db)
):
    """修改当前用户密码"""
    old_password = request.get("old_password")
    new_password = request.get("new_password")
    
    if not old_password or not new_password:
        raise HTTPException(status_code=400, detail="Missing password fields")
        
    if not verify_password(old_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect old password")
        
    user.hashed_password = get_password_hash(new_password)
    await db.commit()
    return {"message": "Password changed successfully"}

# --- 用户个人信息维护模块结束 ---


# --- 反馈模块结束 ---


if __name__ == "__main__":
    import uvicorn
    # 强制使用 127.0.0.1 和 8088 端口，避开所有可能的冲突
    print(">>> 正在启动 AI 课程助教后端 <<<")
    print(">>> 请确保前端请求地址为: http://127.0.0.1:8088 <<<")
    uvicorn.run("main:app", host="127.0.0.1", port=8088, reload=True)