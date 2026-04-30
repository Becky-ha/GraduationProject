import React, { useState, useEffect, useRef } from 'react'
import {
  sendMessage,
  Message,
  getChatHistory,
  callTextToImage,
  getConversationList,
  ConversationSummary,
  deleteConversation
} from '../services/chatService'
import ReactMarkdown from 'react-markdown'
import rehypeSanitize from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'
import './ChatInterface.css'

interface ChatInterfaceProps {
  onLogout?: () => void
}

interface ChatMessage extends Message {
  user_question?: string
  source_label?: string
  matched_docs?: Array<{ filename: string; content_preview: string }>
  isStreaming?: boolean
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ onLogout }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  const [conversationId, setConversationId] = useState<string | undefined>(
    undefined
  )
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [activeMenuId, setActiveMenuId] = useState<string | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [useStreamResponse] = useState<boolean>(true)
  const [messageFeedback, setMessageFeedback] = useState<Record<number, 'like' | 'dislike' | null>>({})
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // 自动滚动到底部
  useEffect(() => {
    // 使用setTimeout确保DOM已更新后再滚动
    setTimeout(() => {
      scrollToBottom()
    }, 0)
  }, [messages])

  const refreshConversationList = async () => {
    try {
      const list = await getConversationList()
      setConversations(list)
    } catch (error: any) {
      console.error('加载会话列表失败:', error)
      if (error.response?.status === 401 && onLogout) {
        onLogout()
      }
    }
  }

  // 第一次加载时检查本地存储的会话ID
  useEffect(() => {
    refreshConversationList()

    // 从本地存储中恢复会话ID
    const savedConversationId = localStorage.getItem(
      'aiCourseAssistantConversationId'
    )

    if (savedConversationId) {
      console.log(`从本地存储恢复会话ID: ${savedConversationId}`)
      void loadChatHistory(savedConversationId)
    } else {
      showWelcomeMessage(false)
    }
  }, [])


  // 统一显示欢迎消息的函数
  const showWelcomeMessage = (isReturnUser: boolean = false) => {
    const content = isReturnUser
      ? '欢迎回来！我是QAI Bot，您的智能辅助问答助手。请问今天有什么可以继续帮您的？'
      : '您好！我是QAI Bot，您的智能辅助问答助手。您可以向我询问任何问题，我会尽力为您解答。请问今天有什么可以帮您的？'
    
    const welcomeMessage: ChatMessage = {
      role: 'assistant',
      content: content,
      timestamp: new Date().toISOString()
    }

    setMessages([welcomeMessage])
  }

  // 保存会话ID到本地存储
  useEffect(() => {
    if (conversationId) {
      localStorage.setItem('aiCourseAssistantConversationId', conversationId)
      console.log(`保存会话ID到本地存储: ${conversationId}`)
    } else {
      localStorage.removeItem('aiCourseAssistantConversationId')
    }
  }, [conversationId])

  // 加载历史聊天记录
  const loadChatHistory = async (conversationId: string) => {
    try {
      setIsLoadingHistory(true)
      console.log(`加载会话历史: ${conversationId}`)
      const history = await getChatHistory(conversationId)

      if (history && history.history && history.history.length > 0) {
        // 转换历史消息为我们应用的格式
        const formattedMessages: ChatMessage[] = history.history.map(
          (msg: any) => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp,
            image_url: msg.image_url,
            response_meta: msg.response_meta
          })
        )

        setMessages(formattedMessages)
        console.log(`加载了 ${formattedMessages.length} 条历史消息`)
      } else {
        // 如果没有历史消息，显示欢迎消息
        showWelcomeMessage(true)
      }
    } catch (error: any) {
      console.error('加载历史消息失败:', error)
      
      // 如果是 401 错误，说明 Token 过期，直接退出登录
      if (error.response?.status === 401) {
        console.log('Token 过期，执行退出登录')
        if (onLogout) {
          onLogout()
        }
        return
      }
      
      // 这里的处理比较宽泛，无论是网络错误还是其他鉴权错误，都允许用户开始新对话
      console.log('加载历史失败，回退到欢迎消息并重置会话')
      localStorage.removeItem('aiCourseAssistantConversationId')
      setConversationId(undefined)
      showWelcomeMessage(false)
    } finally {
      setIsLoadingHistory(false)
    }
  }

  const handleSelectConversation = (selectedConversationId: string) => {
    if (selectedConversationId === conversationId || isLoading) return
    setConversationId(selectedConversationId)
    void loadChatHistory(selectedConversationId)
  }

  const handleCreateNewConversation = () => {
    if (isLoading) return
    setConversationId(undefined)
    localStorage.removeItem('aiCourseAssistantConversationId')
    setMessages([])
    setActiveMenuId(null)
    showWelcomeMessage(false)
  }

  const handleDeleteConversation = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!window.confirm('确定要删除这个会话吗？')) return;
    
    try {
      await deleteConversation(id);
      if (conversationId === id) {
        setConversationId(undefined);
        localStorage.removeItem('aiCourseAssistantConversationId');
        showWelcomeMessage(false);
      }
      refreshConversationList();
      setActiveMenuId(null);
    } catch (error) {
      console.error('删除会话失败:', error);
      alert('删除会话失败，请稍后重试');
    }
  };

  const toggleMenu = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setActiveMenuId(activeMenuId === id ? null : id);
  };

  useEffect(() => {
    const handleClickOutside = () => setActiveMenuId(null);
    window.addEventListener('click', handleClickOutside);
    return () => window.removeEventListener('click', handleClickOutside);
  }, []);

  const formatDateTime24 = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN', {
      hour12: false,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'end' // 确保滚动到底部
      })
    }
  }

  // 处理发送消息
  const handleSendMessage = async () => {
    // 如果正在加载或者消息为空，不处理
    if (isLoading || !input.trim()) return

    // 清除消息输入框并设置加载状态
    const messageContent = input.trim()
    setInput('')
    setIsLoading(true)

    try {
      console.log('发送新消息:', messageContent)

      // 准备聊天历史 - 仅包含文本消息，不包含系统消息和临时消息
      const chatHistory = messages
        .filter((msg) => msg.role !== 'system' && !msg.isTemporary)
        .map((msg) => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp,
          ...(msg.image_url ? { image_url: msg.image_url } : {})
        }))

      console.log(`准备聊天历史, 共 ${chatHistory.length} 条消息`)

      // 在消息列表中添加用户消息
      const userMessage: ChatMessage = {
        role: 'user',
        content: messageContent,
        timestamp: new Date().toISOString()
      }

      setMessages((prevMessages) => [...prevMessages, userMessage])

      // 添加一个临时的助手消息占位符
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        isTemporary: true,
        isStreaming: true,
        user_question: messageContent
      }

      setMessages((prevMessages) => [...prevMessages, assistantMessage])

      // 检查是否是文生图请求
      const isImageGenRequest = isTextToImageRequest(messageContent)

      // 根据消息类型选择不同的API
      if (isImageGenRequest) {
        // 如果是文生图请求，调用文生图API
        console.log('检测到文生图请求，调用文生图API')

        // 从请求中提取提示词
        const prompt = extractPromptFromRequest(messageContent)
        console.log(`提取的图像提示词: "${prompt}"`)

        try {
          // 调用文生图API
          const imageResponse = await callTextToImage(prompt, {
            conversationId: conversationId
          })

          console.log('文生图响应:', imageResponse)

          // 设置会话ID
          if (
            imageResponse.conversation_id &&
            (!conversationId ||
              conversationId !== imageResponse.conversation_id)
          ) {
            setConversationId(imageResponse.conversation_id)
          }

          // 如果有生成的图片，更新助手消息
          if (imageResponse.image_urls && imageResponse.image_urls.length > 0) {
            const imageUrl = imageResponse.image_urls[0]

            // 更新助手消息
            setMessages((prevMessages) => {
              const assistantIndex = prevMessages.findIndex(
                (msg) => msg.isTemporary
              )
              if (assistantIndex !== -1) {
                const updatedMessages = [...prevMessages]
                updatedMessages[assistantIndex] = {
                  ...updatedMessages[assistantIndex],
                  content: `已根据您的描述生成图片: ${prompt}`,
                  image_url: imageUrl,
                  isTemporary: false,
                  user_question: messageContent
                }
                return updatedMessages
              }
              return prevMessages
            })
          } else {
            // 如果没有生成图片，显示错误信息
            setMessages((prevMessages) => {
              const assistantIndex = prevMessages.findIndex(
                (msg) => msg.isTemporary
              )
              if (assistantIndex !== -1) {
                const updatedMessages = [...prevMessages]
                updatedMessages[assistantIndex] = {
                  ...updatedMessages[assistantIndex],
                  content:
                    '很抱歉，图像生成失败。请尝试提供更详细的描述，或者稍后再试。',
                  isTemporary: false,
                  user_question: messageContent
                }
                return updatedMessages
              }
              return prevMessages
            })
          }
        } catch (error) {
          console.error('文生图请求失败:', error)
          // 更新助手消息显示错误
          setMessages((prevMessages) => {
            const assistantIndex = prevMessages.findIndex(
              (msg) => msg.isTemporary
            )
            if (assistantIndex !== -1) {
              const updatedMessages = [...prevMessages]
              updatedMessages[assistantIndex] = {
                ...updatedMessages[assistantIndex],
                content: '很抱歉，图像生成服务暂时不可用。请稍后再试。',
                isTemporary: false,
                user_question: messageContent
              }
              return updatedMessages
            }
            return prevMessages
          })
        }
      } else {
        // 不是文生图请求，使用普通文本API
        try {
          // 处理流式响应
          if (useStreamResponse) {
            const response = await sendMessage(
              messageContent,
              conversationId,
              // 确保发送的是最新的聊天历史，不包括刚添加的用户消息和临时助手消息
              messages
                .filter((msg) => !msg.isTemporary)
                .map((msg) => ({
                  role: msg.role,
                  content: msg.content,
                  timestamp: msg.timestamp
                })),
              useStreamResponse,
              // 添加token接收回调，用于实时更新UI
              (token: string) => {
                // 使用函数式更新确保我们总是基于最新状态更新消息
                setMessages((prevMessages) => {
                  const assistantIndex = prevMessages.findIndex(
                    (msg) => msg.isTemporary || msg.isStreaming
                  )

                  if (assistantIndex !== -1) {
                    const updatedMessages = [...prevMessages]
                    updatedMessages[assistantIndex] = {
                      ...updatedMessages[assistantIndex],
                      content: updatedMessages[assistantIndex].content + token,
                      role: 'assistant',
                      isTemporary: false,
                      isStreaming: true
                    }
                    return updatedMessages
                  }

                  return prevMessages
                })
              }
            )

            // 如果是新会话，保存会话ID
            if (
              response.conversationId &&
              (!conversationId || conversationId !== response.conversationId)
            ) {
              setConversationId(response.conversationId)
              console.log(`设置会话ID: ${response.conversationId}`)
              refreshConversationList()
            }

            // 更新助手消息的 meta 信息（包括知识库来源和文档）
            setMessages((prevMessages) => {
              const assistantIndex = prevMessages.findIndex(
                (msg) =>
                  !msg.isTemporary &&
                  msg.role === 'assistant' &&
                  msg.content === response.content
              )
              if (assistantIndex !== -1) {
                const updatedMessages = [...prevMessages]
                const res = response as any
                updatedMessages[assistantIndex] = {
                  ...updatedMessages[assistantIndex],
                  source_label:
                    res.source_label || res.response_meta?.source_label,
                  matched_docs:
                    res.matched_docs || res.response_meta?.matched_docs,
                  image_url: res.image_url || updatedMessages[assistantIndex].image_url,
                  isStreaming: false
                }
                return updatedMessages
              }
              return prevMessages
            })
          } else {
            // 非流式响应
            const response = await sendMessage(
              messageContent,
              conversationId,
              // 确保发送的是最新的聊天历史，不包括刚添加的用户消息和临时助手消息
              messages
                .filter((msg) => !msg.isTemporary)
                .map((msg) => ({
                  role: msg.role,
                  content: msg.content,
                  timestamp: msg.timestamp
                }))
            )

            // 如果是新会话，保存会话ID
            if (
              response.conversationId &&
              (!conversationId || conversationId !== response.conversationId)
            ) {
              setConversationId(response.conversationId)
              console.log(`设置会话ID: ${response.conversationId}`)
              refreshConversationList()
            }

            // 更新助手消息
            setMessages((prevMessages) => {
              const assistantIndex = prevMessages.findIndex(
                (msg) => msg.isTemporary
              )
              if (assistantIndex !== -1) {
                const updatedMessages = [...prevMessages]
                updatedMessages[assistantIndex] = {
                  ...updatedMessages[assistantIndex],
                  content: response.content,
                  ...(response.image_url ? { image_url: response.image_url } : {}),
                  isTemporary: false,
                  isStreaming: false,
                  user_question: messageContent,
                  source_label: response.response_meta?.source_label,
                  matched_docs: (response.response_meta as any)?.matched_docs
                }
                return updatedMessages
              }
              return prevMessages
            })
          }
        } catch (error) {
          console.error('发送消息失败:', error)
          handleMessageError(error)
        }
      }

      setIsLoading(false)
    } catch (error: any) {
      console.error('处理消息出错:', error)
      setIsLoading(false)
      handleMessageError(error)
    }
  }

  // 处理消息错误的辅助函数
  const handleMessageError = (error: any) => {
    // 如果是 401 错误，执行退出登录
    if (error.response?.status === 401) {
      console.log('Token 过期，执行退出登录')
      if (onLogout) {
        onLogout()
      }
      return
    }

    // 添加详细的错误消息
    const errorContent = `很抱歉，服务暂时出现问题: ${
      error.message || '未知错误'
    }

如果您在学习过程中遇到困难，请尝试查阅课程教材或稍后再向我咨询。`

    // 首先尝试更新现有的临时消息
    setMessages((prevMessages) => {
      const assistantIndex = prevMessages.findIndex((msg) => msg.isTemporary)
      if (assistantIndex !== -1) {
        const updatedMessages = [...prevMessages]
        updatedMessages[assistantIndex] = {
          role: 'system',
          content: errorContent,
          timestamp: new Date().toISOString(),
          isTemporary: false
        }
        return updatedMessages
      }
      // 如果没有找到临时消息，添加一个新的错误消息
      return [
        ...prevMessages,
        {
          role: 'system',
          content: errorContent,
          timestamp: new Date().toISOString()
        }
      ]
    })
  }


  const handleFeedback = (index: number, type: 'like' | 'dislike') => {
    setMessageFeedback((prev) => ({
      ...prev,
      [index]: prev[index] === type ? null : type
    }))
  }

  const handleCopyMessage = async (content: string, index: number) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedIndex(index)
      setTimeout(() => setCopiedIndex(null), 2000)
    } catch (error) {
      console.error('复制消息失败:', error)
    }
  }

  const handleRetryMessage = (message: ChatMessage) => {
    const userQuestion = message.user_question
    if (!userQuestion) return
    void handleResendQuestion(userQuestion)
  }

  const getSourceLabel = (message: ChatMessage): string => {
    if (!message.source_label) return ''
    return message.source_label
  }

  const handleResendQuestion = async (question: string) => {
    if (isLoading || !question.trim()) return
    setIsLoading(true)
    const messageContent = question.trim()
    try {
      const chatHistory = messages
        .filter((msg) => msg.role !== 'system' && !msg.isTemporary)
        .map((msg) => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp,
          ...(msg.image_url ? { image_url: msg.image_url } : {})
        }))

      const userMessage: ChatMessage = {
        role: 'user',
        content: messageContent,
        timestamp: new Date().toISOString()
      }
      setMessages((prevMessages) => [...prevMessages, userMessage])

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        isTemporary: true,
        isStreaming: true,
        user_question: messageContent
      }
      setMessages((prevMessages) => [...prevMessages, assistantMessage])

      const response = await sendMessage(
        messageContent,
        conversationId,
        chatHistory
      )

      if (response.conversationId && (!conversationId || conversationId !== response.conversationId)) {
        setConversationId(response.conversationId)
        refreshConversationList()
      }

      if (response.conversationId) {
        localStorage.setItem('aiCourseAssistantConversationId', response.conversationId)
      }

      setMessages((prevMessages) => {
        const assistantIndex = prevMessages.findIndex((msg) => msg.isTemporary)
        if (assistantIndex !== -1) {
          const updatedMessages = [...prevMessages]
          updatedMessages[assistantIndex] = {
            ...updatedMessages[assistantIndex],
            content: response.content,
            ...(response.image_url ? { image_url: response.image_url } : {}),
            isTemporary: false
          }
          return updatedMessages
        }
        return prevMessages
      })
    } catch (error) {
      console.error('重试发送失败:', error)
      handleMessageError(error)
    } finally {
      setIsLoading(false)
    }
  }

  // Markdown渲染组件
  const MarkdownContent = ({ content }: { content: string | any }) => {
    // 确保内容是字符串类型
    const safeContent = React.useMemo(() => {
      if (typeof content === 'string') {
        return content
      } else if (content === null || content === undefined) {
        return ''
      } else if (Array.isArray(content)) {
        // 如果是数组，尝试找到文本内容并连接
        return content
          .map((item) => {
            if (typeof item === 'string') return item
            if (item && typeof item === 'object' && 'text' in item)
              return item.text
            return JSON.stringify(item)
          })
          .join('\n')
      } else if (typeof content === 'object') {
        // 如果是对象，尝试提取文本内容或转为JSON字符串
        if ('text' in content) return String(content.text)
        if ('content' in content) return String(content.content)
        return JSON.stringify(content)
      }
      // 其他类型尝试转换为字符串
      return String(content)
    }, [content])

    return (
      <div className="markdown-content">
        <ReactMarkdown
          rehypePlugins={[rehypeSanitize]}
          remarkPlugins={[remarkGfm]}
        >
          {safeContent}
        </ReactMarkdown>
      </div>
    )
  }

  // 检查消息是否是文生图请求
  const isTextToImageRequest = (message: string): boolean => {
    const imageGenerationKeywords = [
      '生成图片',
      '生成一张图片',
      '生成一幅图',
      '画一张',
      '绘制一张',
      '图像生成',
      '图片生成',
      '生成示意图',
      '创建图片',
      '帮我画',
      '制作图片',
      '绘图',
      '做一张图'
    ]

    return imageGenerationKeywords.some((keyword) => message.includes(keyword))
  }

  // 从文生图请求中提取提示词
  const extractPromptFromRequest = (message: string): string => {
    const imageGenerationKeywords = [
      '生成图片',
      '生成一张图片',
      '生成一幅图',
      '画一张',
      '绘制一张',
      '图像生成',
      '图片生成',
      '生成示意图',
      '创建图片',
      '帮我画',
      '制作图片',
      '绘图',
      '做一张图'
    ]

    // 尝试提取关键词后面的内容作为图像描述
    for (const keyword of imageGenerationKeywords) {
      if (message.includes(keyword)) {
        const parts = message.split(keyword)
        if (parts.length > 1 && parts[1].trim()) {
          return parts[1].trim()
        }
      }
    }

    // 如果没有明确的描述，使用整个消息
    return message
  }

  return (
    <div className="chat-container">
      <div className="conversation-sidebar">
        <div className="conversation-sidebar-header">
          <h3>会话列表</h3>
          <button
            className="new-conversation-btn"
            onClick={handleCreateNewConversation}
            disabled={isLoading}
          >
            新建会话
          </button>
        </div>

        <div className="conversation-list">
          {conversations.length === 0 ? (
            <div className="empty-conversation">暂无历史会话</div>
          ) : (
            conversations.map((item) => (
              <div
                key={item.conversation_id}
                className={`conversation-item-container ${conversationId === item.conversation_id ? 'active' : ''}`}
              >
                <button
                  className="conversation-item"
                  onClick={() => handleSelectConversation(item.conversation_id)}
                  disabled={isLoading}
                  title={item.conversation_id}
                >
                  <div className="conversation-title">{item.title || '新会话'}</div>
                  <div className="conversation-updated-at">
                    {item.updated_at
                      ? `更新于 ${formatDateTime24(item.updated_at)}`
                      : '暂无更新时间'}
                  </div>
                </button>
                <div className="conversation-actions">
                  <button 
                    className="menu-btn" 
                    onClick={(e) => toggleMenu(e, item.conversation_id)}
                    disabled={isLoading}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="1"></circle>
                      <circle cx="19" cy="12" r="1"></circle>
                      <circle cx="5" cy="12" r="1"></circle>
                    </svg>
                  </button>
                  {activeMenuId === item.conversation_id && (
                    <div className="conversation-menu">
                      <button className="menu-item delete" onClick={(e) => handleDeleteConversation(e, item.conversation_id)}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="3 6 5 6 21 6"></polyline>
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                        <span>删除</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="chat-content-main">
        <div className="messages-container">
          {isLoadingHistory ? (
            <div className="loading-history">会话加载中...</div>
          ) : (
            <>
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`message ${message.role}`}
                >
                  <div className="message-bubble">
                    {message.image_url && (
                      <div className="message-image-container">
                        <img
                          src={message.image_url}
                          alt="AI Generated"
                          className="message-image"
                          onClick={() => window.open(message.image_url, '_blank')}
                        />
                      </div>
                    )}
                    {message.role === 'assistant' &&
                    isLoading &&
                    index === messages.length - 1 &&
                    !message.content ? (
                      <div className="typing">
                        <div className="dot"></div>
                        <div className="dot"></div>
                        <div className="dot"></div>
                      </div>
                    ) : (
                      <div className="message-content-wrapper">
                        <MarkdownContent content={message.content} />
                      </div>
                    )}
                  </div>
                  {/* 用户反馈按钮与来源展示 */}
                  {index > 0 && (message.role === 'user' || (message.role === 'assistant' && !message.isTemporary)) && (
                    <div className="message-actions-container">
                      <div className="actions-and-source-row">
                        <div className="message-actions">
                          {message.role === 'assistant' && (
                            <>
                              <button
                                className={`message-action-btn ${messageFeedback[index] === 'like' ? 'active' : ''}`}
                                onClick={() => handleFeedback(index, 'like')}
                                title="点赞"
                                type="button"
                              >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
                                </svg>
                              </button>
                              <button
                                className={`message-action-btn ${messageFeedback[index] === 'dislike' ? 'active' : ''}`}
                                onClick={() => handleFeedback(index, 'dislike')}
                                title="不赞成"
                                type="button"
                              >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path>
                                </svg>
                              </button>
                              <button
                                className="message-action-btn"
                                onClick={() => handleCopyMessage(message.content, index)}
                                title="复制"
                                type="button"
                              >
                                {copiedIndex === index ? (
                                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#48bb78" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                    <polyline points="20 6 9 17 4 12"></polyline>
                                  </svg>
                                ) : (
                                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                  </svg>
                                )}
                              </button>
                              <button
                                className="message-action-btn"
                                onClick={() => handleRetryMessage(message)}
                                title="重试"
                                type="button"
                              >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M23 4v6h-6"></path>
                                  <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                                </svg>
                              </button>
                            </>
                          )}
                        </div>

                        {message.role === 'assistant' && message.source_label && !message.isStreaming && (
                          <div className="source-label-inline">
                            来源：{message.source_label === '知识库' ? '知识库' : '大模型'}
                          </div>
                        )}
                      </div>

                      {message.role === 'assistant' && message.source_label === '知识库' && message.matched_docs && message.matched_docs.length > 0 && (
                        <div className="matched-docs-list">
                          {message.matched_docs.map((doc, docIndex) => (
                            <div key={docIndex} className="matched-doc-item">
                              <span className="doc-index">{docIndex + 1}.</span>
                              <span className="doc-filename">{doc.filename}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  {message.timestamp && (
                    <div className="message-info">
                      {new Date(message.timestamp).toLocaleTimeString([], {
                        hour12: false,
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        <div className="input-area-container">
          <div className="input-controls">
            <div className="chat-input-wrapper">
              <input
                type="text"
                className="chat-input-field"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="请输入您的问题..."
                disabled={isLoading || isLoadingHistory}
              />
            </div>

            <button
              className="send-btn"
              onClick={handleSendMessage}
              disabled={isLoading || isLoadingHistory || !input.trim()}
            >
              {isLoading ? '发送中...' : '发送'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ChatInterface
