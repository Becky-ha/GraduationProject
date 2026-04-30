import axios from 'axios';

// 定义Message类型
export interface Message {
  role: string;
  content: string;
  timestamp?: string;
  image_url?: string;  // 添加图片URL字段
  isTemporary?: boolean; // 标记是否为临时消息
}

// API基础URL
const API_BASE_URL = 'http://127.0.0.1:8088/api';

// 定义请求类型
export interface ChatRequest {
  message: string;
  chat_history?: any[];
}

// 定义多模态请求类型
export interface MultiModalChatRequest {
  message: string;
  chat_history?: any[];
  image_data?: string;  // Base64编码的图片数据
}

// 文生图请求类型
export interface TextToImageRequest {
  prompt: string;
  negative_prompt?: string;
  n?: number;
  size?: string;
}

// 文生图响应类型
export interface TextToImageResponse {
  image_urls: string[];
  conversation_id: string;
}

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  last_message: string;
  updated_at: string | null;
}

// 获取本地存储的Token
const getAuthHeaders = (): Record<string, string> => {
  const token = localStorage.getItem('token');
  return token ? { Authorization: `Bearer ${token}` } : {};
};

// 发送普通消息（非流式）
export async function sendMessage(
  message: string, 
  conversationId?: string,
  chatHistory?: any[],
  useStream: boolean = false,
  onTokenReceived?: (token: string) => void,
  onImageReceived?: (imageUrl: string) => void
): Promise<{ content: string, conversationId: string, image_url?: string, response_meta?: { source: string; source_label: string } }> {
  console.log(`发送消息: '${message}', conversationId: ${conversationId || '新会话'}, 历史消息数: ${chatHistory?.length || 0}, 使用流式响应: ${useStream}`);
  
  if (useStream) {
    return sendStreamMessage(message, conversationId, chatHistory, onTokenReceived, onImageReceived);
  }
  
  try {
    // 构建请求对象
    const request: ChatRequest = {
      message,
      chat_history: chatHistory || []
    };
    
    // 设置请求头
    const headers = {
      'Content-Type': 'application/json',
      ...getAuthHeaders()
    };
    
    // 如果有会话ID，添加到URL参数
    let url = `${API_BASE_URL}/chat`;
    if (conversationId) {
      url += `?conversation_id=${conversationId}`;
    }
    
    console.log(`发送POST请求到: ${url}`);
    
    // 发送请求
    const response = await axios.post(url, request, { headers });
    
    // 构建返回对象
    const result: { 
      content: string, 
      conversationId: string,
      image_url?: string,
      response_meta?: { source: string; source_label: string }
    } = {
      content: response.data.response,
      conversationId: response.data.conversation_id,
      response_meta: response.data.response_meta
    };
    
    // 如果响应中包含图片URL
    if (response.data.image_url) {
      result.image_url = response.data.image_url;
    }
    
    return result;
  } catch (error) {
    console.error(`发送消息失败:`, error);
    throw error;
  }
}

// 使用表单发送多模态消息（图片+文本）
export async function sendMultiModalMessage(
  message: string,
  imageFile: File,
  conversationId?: string,
  chatHistory?: any[]
) {
  try {
    console.log(`发送多模态消息: '${message}', 图片: ${imageFile.name}, conversationId: ${conversationId || '新会话'}, 历史消息数: ${chatHistory?.length || 0}`);
    
    // 构建FormData
    const formData = new FormData();
    formData.append('message', message);
    formData.append('file', imageFile);
    
    // 如果有会话ID，添加到URL参数
    let url = `${API_BASE_URL}/chat/multimodal`;
    if (conversationId) {
      url += `?conversation_id=${conversationId}`;
    }
    
    console.log(`发送多模态请求到: ${url}`);
    
    // 发送请求
    const response = await axios.post(url, formData, {
      headers: {
        ...getAuthHeaders()
      }
    });
    
    console.log('多模态响应:', response.data);
    
    return {
      content: response.data.response,
      conversationId: response.data.conversation_id
    };
  } catch (error) {
    console.error(`发送多模态消息失败:`, error);
    throw error;
  }
}

// 使用JSON发送多模态消息（Base64图片+文本）
export async function sendMultiModalJsonMessage(
  message: string,
  imageData: string,
  conversationId?: string,
  chatHistory?: any[]
): Promise<{ content: string, conversationId: string, image_url?: string }> {
  try {
    console.log(`发送多模态JSON消息: '${message}', 图片数据长度: ${imageData.length}, conversationId: ${conversationId || '新会话'}, 历史消息数: ${chatHistory?.length || 0}`);
    
    // 确保图片 base64 数据格式正确
    let processedImageData = imageData;
    if (imageData.startsWith('data:')) {
      // 已经是正确格式，保持不变
      console.log('图片数据已包含 data URI 前缀');
    } else {
      // 添加 data URI 前缀
      processedImageData = `data:image/jpeg;base64,${imageData}`;
      console.log('已添加 data URI 前缀到图片数据');
    }
    
    // 确保聊天历史中每个消息对象都有role和content字段
    const sanitizedChatHistory = chatHistory ? chatHistory.map(msg => {
      // 确保消息有必要的字段
      if (typeof msg === 'object' && msg !== null) {
        return {
          role: msg.role || 'user',
          content: msg.content || '',
          timestamp: msg.timestamp || new Date().toISOString()
        };
      }
      // 跳过无效消息
      console.warn('跳过无效历史消息', msg);
      return null;
    }).filter(Boolean) : [];
    
    // 构建请求对象
    const request: MultiModalChatRequest = {
      message,
      chat_history: sanitizedChatHistory,
      image_data: processedImageData
    };
    
    // 如果有会话ID，添加到URL参数
    let url = `${API_BASE_URL}/chat/multimodal-json`;
    if (conversationId) {
      url += `?conversation_id=${conversationId}`;
    }
    
    console.log(`发送多模态JSON请求到: ${url}`);
    console.log(`请求历史消息数: ${sanitizedChatHistory.length}`);
    
    // 发送请求
    const response = await axios.post(url, request, {
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders()
      }
    });
    
    console.log('多模态JSON响应:', response.data);
    
    // 确保响应内容是字符串
    let content = '';
    if (response.data && response.data.response) {
      if (typeof response.data.response === 'string') {
        content = response.data.response;
      } else {
        // 如果响应不是字符串，尝试转换
        try {
          content = JSON.stringify(response.data.response);
        } catch (err) {
          console.error('响应内容转换失败:', err);
          content = '收到响应，但格式无法处理。';
        }
      }
    } else {
      content = '未收到有效响应。';
    }
    
    // 构建返回对象
    const result: { 
      content: string, 
      conversationId: string,
      image_url?: string 
    } = {
      content: content,
      conversationId: response.data.conversation_id || conversationId || ''
    };
    
    // 如果响应中包含图片URL
    if (response.data.image_url) {
      result.image_url = response.data.image_url;
    }
    
    return result;
  } catch (error) {
    console.error(`发送多模态JSON消息失败:`, error);
    throw error;
  }
}

// 发送流式消息
export async function sendStreamMessage(
  message: string, 
  conversationId?: string,
  chatHistory?: any[],
  onTokenReceived?: (token: string) => void,
  onImageReceived?: (imageUrl: string) => void
): Promise<{ content: string, conversationId: string, image_url?: string, source_label?: string, matched_docs?: any[] }> {
  console.log(`发送流式消息: '${message}', conversationId: ${conversationId || '新会话'}, 历史消息数: ${chatHistory?.length || 0}`);

  try {
    // 1. 先发送POST请求，确保消息和历史记录被后端接收
    const request: ChatRequest = {
      message,
      chat_history: chatHistory || []
    };

    const postResponse = await axios.post(`${API_BASE_URL}/chat/stream`, request, {
      params: conversationId ? { conversation_id: conversationId } : {},
      headers: { 
        'Content-Type': 'application/json',
        ...getAuthHeaders()
      }
    });

    // 从POST响应中获取会话ID
    const newConversationId = postResponse.data.conversation_id || conversationId;
    console.log(`POST请求成功，获取到会话ID: ${newConversationId}`);

    // 2. 然后建立EventSource连接来接收流式响应
    return new Promise((resolve, reject) => {
      let receivedContent = '';
      let receivedConversationId = newConversationId || '';
      let receivedSourceLabel = '';
      let receivedMatchedDocs: any[] = [];

      let eventSourceUrl = `${API_BASE_URL}/chat/stream`;
      if (newConversationId) {
        eventSourceUrl += `?conversation_id=${newConversationId}`;
      }

      const token = localStorage.getItem('token');
      if (token) {
        eventSourceUrl += newConversationId ? `&token=${token}` : `?token=${token}`;
      }

      console.log(`建立EventSource连接: ${eventSourceUrl}`);
      const eventSource = new EventSource(eventSourceUrl);

      eventSource.onmessage = (event: MessageEvent) => {
        if (!event.data) return;
        const token = event.data;
        receivedContent += token;
        if (onTokenReceived) {
          onTokenReceived(token);
        }
      };

      eventSource.addEventListener('done', (event: Event) => {
        const messageEvent = event as MessageEvent;
        let data: {
          conversation_id?: string;
          image_url?: string;
          source_label?: string;
          matched_docs?: any[];
          response_meta?: { source_label?: string; matched_docs?: any[] };
        } = {};
        try {
          data = JSON.parse(messageEvent.data);
        } catch (e) {
          console.warn('Done event JSON parse error', e);
        }

        if (data.conversation_id) {
          receivedConversationId = data.conversation_id;
        }
        receivedSourceLabel =
          data.source_label || data.response_meta?.source_label || receivedSourceLabel;
        receivedMatchedDocs =
          data.matched_docs || data.response_meta?.matched_docs || receivedMatchedDocs;

        eventSource.close();
        resolve({
          content: receivedContent,
          conversationId: receivedConversationId,
          image_url: data.image_url,
          source_label: receivedSourceLabel,
          matched_docs: receivedMatchedDocs
        });
      });

      eventSource.onerror = (event: Event) => {
        console.error('EventSource error:', event);
        eventSource.close();
        reject(new Error('流式响应连接出错'));
      };
    });
  } catch (error) {
    console.error('初始化流式请求失败:', error);
    throw error;
  }
}

// 获取对话历史
export const getChatHistory = async (conversationId: string) => {
  const response = await axios.get(`${API_BASE_URL}/history/${conversationId}`, {
    headers: getAuthHeaders()
  });
  return response.data;
};

// 获取会话ID列表
export const getConversationIds = async (): Promise<string[]> => {
  const response = await axios.get(`${API_BASE_URL}/history`, {
    headers: getAuthHeaders()
  });
  if (Array.isArray(response.data?.conversations)) {
    return response.data.conversations.map(
      (item: ConversationSummary) => item.conversation_id
    );
  }
  return response.data?.conversation_ids || [];
};

// 获取会话摘要列表
export const getConversationList = async (): Promise<ConversationSummary[]> => {
  const response = await axios.get(`${API_BASE_URL}/history`, {
    headers: getAuthHeaders()
  });
  return response.data?.conversations || [];
};

export const syncConversationList = async (): Promise<ConversationSummary[]> => {
  return getConversationList();
};

// 删除会话
export const deleteConversation = async (conversationId: string): Promise<void> => {
  await axios.delete(`${API_BASE_URL}/history/${conversationId}`, {
    headers: getAuthHeaders()
  });
};

// 文生图API调用
export async function callTextToImage(
  prompt: string,
  options: {
    negative_prompt?: string;
    n?: number;
    size?: string;
    conversationId?: string;
  } = {}
): Promise<TextToImageResponse> {
  try {
    console.log(`发送文生图请求 - 提示词: '${prompt}', 会话ID: ${options.conversationId || '新会话'}`);
    
    // 构建请求对象
    const request: TextToImageRequest = {
      prompt: prompt
    };
    
    // 添加可选参数
    if (options.negative_prompt) request.negative_prompt = options.negative_prompt;
    if (options.n) request.n = options.n;
    if (options.size) request.size = options.size;
    
    // 设置请求头
    const headers = {
      'Content-Type': 'application/json',
      ...getAuthHeaders()
    };
    
    // 如果有会话ID，添加到URL参数
    let url = `${API_BASE_URL}/text2image`;
    if (options.conversationId) {
      url += `?conversation_id=${options.conversationId}`;
    }
    
    console.log(`发送文生图POST请求到: ${url}`);
    
    // 发送请求
    const response = await axios.post(url, request, { headers });
    
    console.log('文生图响应:', response.data);
    
    // 返回结果
    return {
      image_urls: response.data.image_urls,
      conversation_id: response.data.conversation_id
    };
  } catch (error) {
    console.error(`文生图请求失败:`, error);
    throw error;
  }
} 