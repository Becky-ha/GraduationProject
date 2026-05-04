import React, { useState, useEffect } from 'react';
import { getFeedbackOverview } from '../services/chatService';

interface FeedbackItem {
  message_id: number;
  question: string;
  answer: string;
  like_count: number;
  dislike_count: number;
}

const FeedbackOverview: React.FC = () => {
  const [data, setData] = useState<FeedbackItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<'like_count' | 'dislike_count'>('like_count');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setIsLoading(true);
      const result = await getFeedbackOverview();
      setData(result);
      setError(null);
    } catch (err) {
      setError('加载反馈数据失败');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSort = (field: 'like_count' | 'dislike_count') => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  const sortedData = [...data].sort((a, b) => {
    const valA = a[sortField];
    const valB = b[sortField];
    return sortOrder === 'desc' ? valB - valA : valA - valB;
  });

  const truncateText = (text: string, maxLength: number = 80) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div className="analysis-sub-content">
      <div className="section-header">
        <h2>反馈统计</h2>
        <button className="refresh-mini-btn" onClick={fetchData}>刷新</button>
      </div>
      <div className="table-wrapper">
        {isLoading ? (
          <div className="sub-loading">加载中...</div>
        ) : error ? (
          <div className="sub-error">{error}</div>
        ) : (
          <table className="analysis-table">
            <thead>
              <tr>
                <th>问题</th>
                <th>回答</th>
                <th onClick={() => handleSort('like_count')} className="sortable-header">
                  👍 点赞 {sortField === 'like_count' && (sortOrder === 'desc' ? '↓' : '↑')}
                </th>
                <th onClick={() => handleSort('dislike_count')} className="sortable-header">
                  👎 点踩 {sortField === 'dislike_count' && (sortOrder === 'desc' ? '↓' : '↑')}
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedData.map(item => (
                <tr key={item.message_id}>
                  <td title={item.question}>{truncateText(item.question, 40)}</td>
                  <td title={item.answer}>{truncateText(item.answer, 60)}</td>
                  <td className="count-cell like">{item.like_count}</td>
                  <td className="count-cell dislike">{item.dislike_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default FeedbackOverview;
