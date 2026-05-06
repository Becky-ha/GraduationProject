import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface KeywordCoverage {
  keyword: string;
  is_covered: boolean;
  source_count: number;
  hot_count: number;
  suggested_action: string;
  priority: string;
}

interface KnowledgeAnalysisData {
  total_files: number;
  total_chunks: number;
  coverage_rate: number;
  keyword_details: KeywordCoverage[];
}

const KnowledgeAnalysis: React.FC = () => {
  const [data, setData] = useState<KnowledgeAnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAnalysis();
  }, []);

  const fetchAnalysis = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      const response = await axios.get('http://127.0.0.1:8088/api/admin/knowledge-analysis', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setData(response.data);
      setError(null);
    } catch (err) {
      setError('获取知识库分析数据失败');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="sub-loading">正在通过向量库分析知识库覆盖情况...</div>;
  if (error) return <div className="sub-error">{error}</div>;
  if (!data) return null;

  return (
    <div className="analysis-sub-content">
      <div className="section-header">
        <div />
        <button className="refresh-mini-btn" onClick={fetchAnalysis}>重新分析</button>
      </div>

      <div className="analysis-overview-cards">
        <div className="overview-card">
          <span className="card-label">已上传文件</span>
          <span className="card-value">{data.total_files}</span>
        </div>
        <div className="overview-card">
          <span className="card-label">知识分块总数</span>
          <span className="card-value">{data.total_chunks}</span>
        </div>
        <div className="overview-card highlight">
          <span className="card-label">核心关键词覆盖率</span>
          <span className="card-value">{data.coverage_rate}%</span>
        </div>
      </div>

      <div className="table-wrapper" style={{ marginTop: '2rem' }}>
        <h4 style={{ marginBottom: '1rem', color: '#606266' }}>人工智能核心关键词覆盖详情</h4>
        <table className="analysis-table">
          <thead>
            <tr>
              <th>关键词</th>
              <th>覆盖状态</th>
              <th>语义相关块数量</th>
              <th>优化建议</th>
            </tr>
          </thead>
          <tbody>
            {data.keyword_details.map((item, index) => (
              <tr key={index}>
                <td style={{ fontWeight: 600 }}>{item.keyword}</td>
                <td>
                  <span className={`status-badge ${item.is_covered ? 'covered' : 'missing'}`}>
                    {item.is_covered ? '已覆盖' : '知识盲区'}
                  </span>
                  {item.priority === 'urgent' && (
                    <span className="priority-badge urgent" style={{ marginLeft: '8px' }}>紧急</span>
                  )}
                </td>
                <td className="count-cell">
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <span>知识块: {item.source_count}</span>
                    <span style={{ fontSize: '0.8rem', color: '#909399' }}>高频提问: {item.hot_count}次</span>
                  </div>
                </td>
                <td style={{ color: item.priority === 'urgent' ? '#f56c6c' : (item.is_covered ? '#67c23a' : '#e6a23c') }}>
                  {item.suggested_action}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default KnowledgeAnalysis;
