import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './KnowledgeBaseManager.css';

interface KnowledgeFile {
  id: number;
  filename: string;
  file_type: string;
  file_size: number;
  status: string;
  progress: number;
  error_message?: string;
  created_at: string;
}

const KnowledgeBaseManager: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [knowledgeFiles, setKnowledgeFiles] = useState<KnowledgeFile[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [status, setStatus] = useState<{ type: 'success' | 'error' | 'none', message: string }>({ type: 'none', message: '' });
  const [isUploading, setIsLoading] = useState(false);
  const [isLoadingList, setIsLoadingList] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 分页状态
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 6;

  // WebSocket 连接处理
  useEffect(() => {
    const ws = new WebSocket('ws://127.0.0.1:8088/ws/knowledge-status');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('收到进度更新:', data);
      
      setKnowledgeFiles(prev => prev.map(file => {
        if (file.id === data.file_id) {
          return {
            ...file,
            status: data.status,
            progress: data.progress,
            error_message: data.error
          };
        }
        return file;
      }));

      // 如果有新文件完成解析，刷新列表以防万一
      if (data.status === 'completed') {
        // 适当延迟刷新，给向量库一点时间
        setTimeout(fetchKnowledgeFiles, 1000);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket 连接已关闭，尝试重新连接...');
      // 可以添加重连逻辑
    };

    return () => ws.close();
  }, []);

  const fetchKnowledgeFiles = async () => {
    setIsLoadingList(true);
    const token = localStorage.getItem('token');
    try {
      const response = await axios.get('http://127.0.0.1:8088/api/admin/knowledge-files', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setKnowledgeFiles(response.data);
    } catch (err) {
      console.error('获取知识库文件列表失败:', err);
    } finally {
      setIsLoadingList(false);
    }
  };

  useEffect(() => {
    fetchKnowledgeFiles();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFiles = Array.from(e.target.files);
      const allowedExtensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.md', '.markdown', '.txt', '.json'];
      
      const invalidFiles = selectedFiles.filter(f => {
        const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
        return !allowedExtensions.includes(ext);
      });

      if (invalidFiles.length > 0) {
        setStatus({ type: 'error', message: `包含不支持的文件格式: ${invalidFiles.map(f => f.name).join(', ')}` });
        return;
      }

      setFiles(prev => [...prev, ...selectedFiles]);
      setStatus({ type: 'none', message: '' });
      
      // 自动触发上传
      if (selectedFiles.length > 0) {
        setTimeout(() => handleUpload(selectedFiles), 100);
      }
    }
  };

  const handleUpload = async (filesToUpload: File[]) => {
    setIsLoading(true);
    setStatus({ type: 'none', message: '' });

    const formData = new FormData();
    filesToUpload.forEach(file => {
      formData.append('files', file);
    });

    const token = localStorage.getItem('token');

    try {
      await axios.post('http://127.0.0.1:8088/api/admin/upload-knowledge', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${token}`
        }
      });

      setStatus({ type: 'success', message: `成功上传 ${filesToUpload.length} 个文件！` });
      setFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = '';
      fetchKnowledgeFiles();
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || '上传失败，请稍后重试';
      setStatus({ type: 'error', message: `错误: ${errorMsg}` });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('确定要删除这个文件吗？')) return;

    const token = localStorage.getItem('token');
    try {
      await axios.delete(`http://127.0.0.1:8088/api/admin/knowledge-files/${id}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setKnowledgeFiles(prev => prev.filter(f => f.id !== id));
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || '删除失败';
      setStatus({ type: 'error', message: `错误: ${errorMsg}` });
    }
  };

  const formatFileSize = (bytes: number | null | undefined) => {
    if (bytes === null || bytes === undefined || isNaN(bytes)) return '未知大小';
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const getFileIcon = (ext: string) => {
    const e = ext.replace('.', '').toLowerCase();
    switch (e) {
      case 'pdf': 
        return (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
          </svg>
        );
      case 'docx': case 'doc': 
        return (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <path d="M9 15h6"></path>
            <path d="M9 12h6"></path>
            <path d="M9 9h1"></path>
          </svg>
        );
      case 'pptx': case 'ppt': 
        return (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
            <line x1="8" y1="21" x2="16" y2="21"></line>
            <line x1="12" y1="17" x2="12" y2="21"></line>
          </svg>
        );
      case 'md': case 'markdown': case 'txt': 
        return (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <line x1="10" y1="9" x2="8" y2="9"></line>
          </svg>
        );
      case 'json': 
        return (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
            <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
            <line x1="12" y1="22.08" x2="12" y2="12"></line>
          </svg>
        );
      default: 
        return (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
          </svg>
        );
    }
  };

  const filteredFiles = knowledgeFiles.filter(f => 
    f.filename.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // 计算当前页的数据
  const totalPages = Math.ceil(filteredFiles.length / itemsPerPage);
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentFiles = filteredFiles.slice(indexOfFirstItem, indexOfLastItem);

  const handlePageChange = (pageNumber: number) => {
    setCurrentPage(pageNumber);
  };

  return (
    <div className="kb-manager-page">
      <div className="kb-header">
        <h1>知识库中心</h1>
        <div className="header-actions">
          <input
            type="file"
            id="kb-file-input"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf,.docx,.doc,.pptx,.ppt,.md,.markdown,.txt,.json"
            multiple
            style={{ display: 'none' }}
          />
          <button 
            className="btn-upload" 
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            <svg className="icon-svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            上传文档
          </button>
        </div>
      </div>

      <div className="search-bar-section">
        <div className="search-input-wrapper">
          <svg className="search-icon-svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          <input
            type="text"
            placeholder="搜索文档名称、标签..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      {status.type !== 'none' && (
        <div className={`status-banner status-${status.type}`}>
          {status.message}
          <button onClick={() => setStatus({ type: 'none', message: '' })} className="close-banner">×</button>
        </div>
      )}

      {isLoadingList ? (
        <div className="loading-state">正在加载知识库文档...</div>
      ) : (
        <div className="knowledge-grid">
          {currentFiles.length === 0 ? (
            <div className="empty-state">
              <svg className="empty-icon-svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#cbd5e0" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
              </svg>
              <p>暂无相关文档，点击右上角上传</p>
            </div>
          ) : (
            currentFiles.map(file => (
              <div key={file.id} className="knowledge-card">
                <div className="card-top">
                  <div className={`file-icon-bg tag-${file.file_type.replace('.', '')}`}>
                    <span className="file-icon">{getFileIcon(file.file_type)}</span>
                  </div>
                  <div className={`status-badge status-${file.status}`}>
                    {file.status === 'processing' ? (
                      <div className="processing-wrapper">
                        <span className="dot processing"></span>
                        <span className="status-text">解析中 {file.progress}%</span>
                      </div>
                    ) : file.status === 'pending' ? (
                      <div className="pending-wrapper">
                        <span className="dot pending"></span>
                        <span className="status-text">待解析</span>
                      </div>
                    ) : file.status === 'failed' ? (
                      <div className="failed-wrapper" title={file.error_message}>
                        <span className="dot failed"></span>
                        <span className="status-text">解析失败</span>
                      </div>
                    ) : (
                      <div className="completed-wrapper">
                        <span className="dot completed"></span>
                        <span className="status-text">正常</span>
                      </div>
                    )}
                  </div>
                  <button className="card-delete-btn" onClick={() => handleDelete(file.id)}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6"></polyline>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                   </button>
                </div>
                
                <div className="card-content">
                  <h3 className="file-name" title={file.filename}>{file.filename}</h3>
                  <p className="file-desc">暂无描述信息...</p>
                </div>

                <div className="card-footer">
                  <div className="footer-info">
                    <span className="info-type">{file.file_type.replace('.', '').toUpperCase()}</span>
                    <span className="info-divider">·</span>
                    <span className="info-size">{formatFileSize(file.file_size)}</span>
                  </div>
                  <div className="footer-date">
                    {new Date(file.created_at).toLocaleDateString()}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* 分页组件 */}
      {knowledgeFiles.length > 0 && (
        <div className="pagination-container">
          <button 
            className="pagination-arrow" 
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
            title="上一页"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6"></polyline>
            </svg>
          </button>
          
          <div className="pagination-numbers">
            {Array.from({ length: totalPages || 1 }, (_, i) => i + 1).map(number => (
              <button
                key={number}
                className={`pagination-number ${currentPage === number ? 'active' : ''}`}
                onClick={() => handlePageChange(number)}
              >
                {number}
              </button>
            ))}
          </div>

          <button 
            className="pagination-arrow" 
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === (totalPages || 1)}
            title="下一页"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
          </button>
        </div>
      )}
    </div>
  );
};

export default KnowledgeBaseManager;
