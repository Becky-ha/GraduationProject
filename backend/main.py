from fastapi import FastAPI, Request, Depends, status, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
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
from langchain_community.chat_models import ChatTongyi
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
        embeddings = DashScopeEmbeddings(model="text-embedding-v1")
        test_vector = embeddings.embed_query("embedding health check")
        print(f"DashScope嵌入模型初始化成功，向量维度: {len(test_vector)}")
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
    allow_origins=["*"],
    allow_credentials=False,
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
                return vectorstore
            except Exception as load_err:
                print(f"加载本地向量库失败，将重新构建: {str(load_err)}")

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
                # 使用 vectorstore 进行带分值的相似度搜索
                # search_type="similarity_score_threshold" 在 Chroma 中可能不直接支持 invoke
                # 故使用向量库原生的相似度搜索方法
                # k=3，设置一个相似度阈值 (0-1)，分值越接近 1 越相似
                # 注意：Chroma 的分值通常是距离值（L2 距离），越小越相似，通常 < 1.0 是较好的匹配
                search_results = vectorstore.similarity_search_with_relevance_scores(question, k=3)
                
                # 过滤掉低于相关度阈值的文档
                # 提高阈值到 0.65 左右通常能过滤掉大部分不相关的“噪音”
                threshold = 0.65
                source_docs = [doc for doc, score in search_results if score >= threshold]
                
                if search_results:
                    print(f"原始检索结果数量: {len(search_results)}, 最高得分: {search_results[0][1]:.4f}")
                    if source_docs:
                        print(f"过滤后保留文档数量: {len(source_docs)}")
                    else:
                        print("分值过低，已过滤所有检索结果，转为模型直接回答")
                
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

            rag_chain = rag_prompt | model | StrOutputParser()
            try:
                answer = await rag_chain.ainvoke({
                    "context": context,
                    "question": question,
                    "history_context": history_context
                })
                
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
            answer = model.invoke(messages).content
            response_meta = {
                "source": "llm",
                "source_label": "仅来自于大模型"
            }
            return answer, response_meta
        except Exception as e:
            print(f"模型调用失败: {str(e)}")
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
        conversation_id = str(datetime.now().timestamp())
    
    # 如果是新的会话ID，初始化
    if conversation_id not in conversation_store:
        print(f"创建新会话: {conversation_id}")
        conversation_store[conversation_id] = {
            "messages": []
        }
    
    return conversation_id

@app.get("/")
async def root():
    """健康检查端点"""
    current_time = datetime.now().isoformat()
    return {
        "status": "ok", 
        "message": "AI课程助教系统正在运行",
        "version": "1.0.0",
        "rag_status": "enabled" if retriever else "disabled",
        "timestamp": current_time
    }

# 用户注册
@app.post("/api/register")
async def register(user: UserRegister, db: AsyncSession = Depends(get_db)):
    print(f"收到注册请求: {user.username}, 邮箱: {user.email}, 角色: {user.role}")
    # 检查用户名是否已存在
    result = await db.execute(select(UserModel).where(UserModel.username == user.username))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # 检查邮箱是否已存在
    result = await db.execute(select(UserModel).where(UserModel.email == user.email))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # 创建新用户
    hashed_password = get_password_hash(user.password)
    # 限制角色只能是 admin 或 student
    final_role = user.role if user.role in ["admin", "student"] else "student"
    new_user = UserModel(
        username=user.username, 
        email=user.email, 
        hashed_password=hashed_password, 
        role=final_role,
        status="active" # 明确设置初始状态
    )
    db.add(new_user)
    await db.commit()
    return {"message": "User registered successfully"}

# 用户登录
@app.post("/api/login")
async def login(user: UserLogin, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserModel).where(UserModel.username == user.username))
    db_user = result.scalars().first()
    
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    # 检查用户是否被封禁
    if db_user.status == "disabled":
        raise HTTPException(status_code=403, detail="Your account has been disabled")
    
    access_token = create_access_token(data={"sub": str(db_user.id)})
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "username": db_user.username,
        "role": db_user.role
    }

# 忘记密码 - 发送验证码
@app.post("/api/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    try:
        print(f"尝试找回密码，邮箱: {request.email}")
        result = await db.execute(select(UserModel).where(UserModel.email == request.email))
        user = result.scalars().first()
        
        if not user:
            print(f"找回密码失败: 未找到邮箱为 {request.email} 的用户")
            # 出于安全考虑，对外返回成功，但控制台记录真实情况
            return {"message": "If the email exists, a reset code has been sent."}
        
        # 生成 6 位随机验证码
        import random
        reset_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        
        # 设置过期时间（10分钟后）
        user.reset_code = reset_code
        user.reset_code_expires = datetime.utcnow() + timedelta(minutes=10)
        
        await db.commit()
        
        # 模拟发送邮件 - 在控制台打印验证码
        print(f"\n{'='*30}")
        print(f"找回密码验证码: {reset_code}")
        print(f"{'='*30}\n")
        
        return {"message": "Reset code sent to your email (simulated in console)."}
    except Exception as e:
        print(f"找回密码接口内部错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"数据库查询失败，请检查是否删除了旧的 .db 文件: {str(e)}")

# 重置密码
@app.post("/api/reset-password")
async def reset_password(request: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserModel).where(UserModel.email == request.email))
    user = result.scalars().first()
    
    if not user or user.reset_code != request.code:
        raise HTTPException(status_code=400, detail="Invalid email or code")
    
    if user.reset_code_expires < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Reset code has expired")
    
    # 更新密码
    user.hashed_password = get_password_hash(request.new_password)
    user.reset_code = None  # 清除验证码
    user.reset_code_expires = None
    
    await db.commit()
    return {"message": "Password has been reset successfully"}

# --- 用户管理模块开始 ---

# 用户管理接口：获取当前登录用户信息
@app.get("/api/user/me")
async def get_my_info(user: UserModel = Depends(get_active_user)):
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "status": user.status,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }

# 管理员接口：获取所有用户列表
@app.get("/api/admin/users")
async def get_all_users(
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(UserModel).order_by(UserModel.id.desc()))
    users = result.scalars().all()
    return [{
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "role": u.role,
        "status": u.status,
        "created_at": u.created_at.isoformat() if u.created_at else None
    } for u in users]

# 管理员接口：查看某个用户详情
@app.get("/api/admin/users/{user_id}")
async def get_user_detail(
    user_id: int,
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "status": user.status,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }

# 管理员接口：封禁/解封用户
@app.post("/api/admin/ban_user")
async def update_user_status(
    request: UserStatusUpdate,
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    if request.status not in ["active", "disabled"]:
        raise HTTPException(status_code=400, detail="Invalid status. Must be 'active' or 'disabled'")
    
    result = await db.execute(select(UserModel).where(UserModel.id == request.user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="You cannot ban yourself")
    
    user.status = request.status
    await db.commit()
    return {"message": f"User status updated to {request.status}"}

# 用户管理模块结束 ---

# --- 用户个人信息维护模块开始 ---

# 修改个人基本信息
@app.put("/api/user/update")
async def update_user_info(
    request: UserUpdate,
    user: UserModel = Depends(get_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    用户修改个人基本信息 (支持修改 username 和 email)
    """
    try:
        updated = False
        
        # 1. 处理用户名修改
        if request.username is not None and request.username != user.username:
            # 检查用户名是否已被占用
            result = await db.execute(select(UserModel).where(UserModel.username == request.username))
            if result.scalars().first():
                raise HTTPException(status_code=400, detail="Username already exists")
            user.username = request.username
            updated = True
            
        # 2. 处理邮箱修改
        if request.email is not None and request.email != user.email:
            # 检查邮箱是否已被占用
            result = await db.execute(select(UserModel).where(UserModel.email == request.email))
            if result.scalars().first():
                raise HTTPException(status_code=400, detail="Email already registered by another user")
            user.email = request.email
            updated = True
            
        if not updated:
            return {"message": "No changes provided"}
            
        await db.commit()
        await db.refresh(user)
        
        return {
            "message": "User information updated successfully",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "status": user.status
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        print(f"更新用户信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update user information: {str(e)}")

# 修改密码
@app.post("/api/user/change-password")
async def change_password(
    request: PasswordChange,
    user: UserModel = Depends(get_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    用户在登录状态下修改密码
    """
    try:
        # 1. 校验旧密码
        if not verify_password(request.old_password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect old password")
        
        # 2. 加密并保存新密码
        user.hashed_password = get_password_hash(request.new_password)
        
        await db.commit()
        return {"message": "Password updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        print(f"修改密码失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update password: {str(e)}")

# --- 用户个人信息维护模块结束 ---

@app.post("/api/chat")
async def chat(
    request: ChatRequest, 
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    conversation_id: str = Depends(get_conversation_id)
):
    """非流式聊天API端点"""
    try:
        # 打印接收到的请求信息
        print(f"非流式请求 - 消息: '{request.message}', 会话ID: {conversation_id}")
        
        try:
            model = ChatTongyi(model_name="qwen-turbo")
        except Exception as e:
            print(f"模型初始化失败: {str(e)}")
            # 返回错误响应
            return {
                "response": _generate_fallback_response(request.message, request.chat_history),
                "conversation_id": conversation_id,
                "error": f"模型初始化失败: {str(e)}"
            }
        
        # 处理聊天历史，可能是Pydantic模型或已经是字典列表
        chat_history = []
        if request.chat_history:
            # 转换Message对象为字典
            for msg in request.chat_history:
                if isinstance(msg, dict):
                    chat_history.append(msg)
                elif hasattr(msg, "role") and hasattr(msg, "content"):
                    chat_history.append({
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp if hasattr(msg, "timestamp") else None
                    })
        
        # 如果是新的会话，获取聊天历史
        current_chat_history = []
        if conversation_id in conversation_store:
            current_chat_history = conversation_store[conversation_id]["messages"]
        
        # 合并当前会话历史和请求中的历史 - 优先使用请求中的历史，如果没有则使用服务器存储的历史
        if not chat_history and current_chat_history:
            chat_history = current_chat_history
        
        # 打印历史长度
        print(f"使用的历史消息数量: {len(chat_history)}")
        
        # 使用智能回答函数处理请求
        response_result = await smart_answer(request.message, model, chat_history)
        if isinstance(response_result, tuple):
            response_content, response_meta = response_result
        else:
            response_content = response_result
            response_meta = {
                "source": "llm",
                "source_label": "仅来自于大模型"
            }
        print(f"生成的回答: '{response_content[:50]}...'(长度:{len(response_content)})")
        
        # 保存到内存会话，供前端实时会话与历史回放使用
        if conversation_id not in conversation_store:
            conversation_store[conversation_id] = {"messages": []}
        conversation_store[conversation_id]["messages"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat()
        })
        conversation_store[conversation_id]["messages"].append({
            "role": "assistant",
            "content": response_content,
            "source_label": response_meta.get("source_label"),
            "matched_docs": response_meta.get("matched_docs"),
            "timestamp": datetime.utcnow().isoformat()
        })
        print(f"会话 {conversation_id} 已写入内存历史，当前消息数: {len(conversation_store[conversation_id]['messages'])}")
        
        # 保存用户消息到数据库
        user_msg = ChatHistoryModel(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=request.message,
            timestamp=datetime.utcnow()
        )
        db.add(user_msg)
        
        # 保存助手消息到数据库
        assistant_msg = ChatHistoryModel(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=response_content,
            source_label=response_meta.get("source_label"),
            matched_docs=json.dumps(response_meta.get("matched_docs")) if response_meta.get("matched_docs") else None,
            timestamp=datetime.utcnow() + timedelta(milliseconds=10)
        )
        db.add(assistant_msg)
        await db.commit()
        
        return {
            "response": response_content,
            "conversation_id": conversation_id,
            "response_meta": response_meta
        }
        
    except Exception as e:
        error_msg = f"聊天API端点错误: {str(e)}"
        print(error_msg)
        print(f"错误详情: {type(e).__name__}, {e.__traceback__.tb_lineno}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

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
        # 尝试从请求头或查询参数获取用户ID
        user_id = None
        if request_raw and "Authorization" in request_raw.headers:
            auth_header = request_raw.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                token_from_header = auth_header.split(" ")[1]
                user_id = await get_current_user_id_optional(token_from_header)
        elif token:
            user_id = await get_current_user_id_optional(token)

        effective_user_id = user_id

        if not effective_user_id:
            print("警告: 流式请求未提供有效的身份验证信息")
            # 在实际生产环境中，这里应该抛出 401 错误
            # raise HTTPException(status_code=401, detail="Not authenticated")
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
                    model = ChatTongyi(model_name="qwen-turbo")
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

                    # 更新内存记录助手消息
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
                    yield {"event": "error", "data": json.dumps({"error": str(e)})}

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
        await db.commit()
        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


# --- 语义聚类逻辑开始 ---

import re

def preprocess_text(text: str) -> str:
    """文本预处理：去除标点、统一大小写、去除空格"""
    # 去除中英文标点
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
    # 统一大小写并去除空格
    return text.lower().strip()

async def update_question_clusters():
    """离线任务：更新问题聚类结果"""
    print("\n>>> [语义聚类] 开始执行增量聚类任务...")
    async with async_session() as db:
        try:
            # 1. 获取所有用户问题
            result = await db.execute(
                select(ChatHistoryModel.content)
                .where(ChatHistoryModel.role == "user", func.length(ChatHistoryModel.content) > 2)
            )
            questions = [row[0] for row in result.all()]
            if not questions:
                print(">>> [语义聚类] 无有效问题数据，跳过任务")
                return

            clusters = [] # 存储结构: {"represent": str, "count": int, "examples": set}
            processed_map = {} # processed_text -> cluster_index
            
            for q in questions:
                p_q = preprocess_text(q)
                if not p_q: continue
                
                # 精确匹配（预处理后）
                if p_q in processed_map:
                    idx = processed_map[p_q]
                    clusters[idx]["count"] += 1
                    clusters[idx]["examples"].add(q)
                else:
                    # 模糊匹配：如果预处理后的字符串包含或非常接近，则归类
                    matched = False
                    # 仅对已有的 Top 类别进行检查以保证性能
                    for cluster in clusters:
                        p_rep = preprocess_text(cluster["represent"])
                        if p_q in p_rep or p_rep in p_q:
                            cluster["count"] += 1
                            cluster["examples"].add(q)
                            processed_map[p_q] = clusters.index(cluster)
                            matched = True
                            break
                    
                    if not matched:
                        processed_map[p_q] = len(clusters)
                        clusters.append({
                            "represent": q,
                            "count": 1,
                            "examples": {q}
                        })

            # 排序并取 Top 30
            clusters.sort(key=lambda x: x["count"], reverse=True)
            top_clusters = clusters[:30]

            # 清理旧数据并写入新数据
            await db.execute(delete(QuestionClusterModel))
            for c in top_clusters:
                new_cluster = QuestionClusterModel(
                    represent_question=c["represent"],
                    count=c["count"],
                    examples=json.dumps(list(c["examples"])[:5], ensure_ascii=False)
                )
                db.add(new_cluster)
            
            await db.commit()
            print(f">>> [语义聚类] 任务完成，成功识别 {len(top_clusters)} 个核心意图类别")
        except Exception as e:
            print(f">>> [语义聚类] 聚类任务失败: {str(e)}")
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
    admin: UserModel = Depends(get_current_admin_v2),
    db: AsyncSession = Depends(get_db)
):
    """管理员：高频问题统计（从聚类表读取）"""
    result = await db.execute(
        select(QuestionClusterModel)
        .order_by(QuestionClusterModel.count.desc())
        .limit(20)
    )
    clusters = result.scalars().all()
    
    # 如果表中没数据，先执行一次同步聚类
    if not clusters:
        await update_question_clusters()
        result = await db.execute(
            select(QuestionClusterModel)
            .order_by(QuestionClusterModel.count.desc())
            .limit(20)
        )
        clusters = result.scalars().all()
    
    return [
        {
            "represent_question": c.represent_question,
            "count": c.count,
            "examples": json.loads(c.examples) if c.examples else []
        }
        for c in clusters
    ]

# --- 反馈模块结束 ---


if __name__ == "__main__":
    import uvicorn
    # 强制使用 127.0.0.1 和 8088 端口，避开所有可能的冲突
    print(">>> 正在启动 AI 课程助教后端 <<<")
    print(">>> 请确保前端请求地址为: http://127.0.0.1:8088 <<<")
    uvicorn.run("main:app", host="127.0.0.1", port=8088, reload=True)