import React, { useState, useEffect } from 'react';
import { getHotQuestions } from '../services/chatService';

interface HotQuestionItem {
  question: string;
  count: number;
}

const HotQuestions: React.FC = () => {
  const [data, setData] = useState<HotQuestionItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setIsLoading(true);
      const result = await getHotQuestions();
      setData(result);
      setError(null);
    } catch (err) {
      setError('加载高频问题失败');
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) return <div className="sub-loading">加载中...</div>;
  if (error) return <div className="sub-error">{error}</div>;

  return (
    <div className="analysis-sub-content">
      <div className="section-header">
        <h2>高频问题</h2>
        <button className="refresh-mini-btn" onClick={fetchData}>刷新</button>
      </div>
      <div className="table-wrapper">
        <table className="analysis-table">
          <thead>
            <tr>
              <th style={{ width: '80px' }}>排名</th>
              <th>问题</th>
              <th style={{ width: '120px' }}>出现次数</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={index}>
                <td className="rank-cell">
                  <span className={`rank-badge rank-${index + 1}`}>
                    {index + 1}
                  </span>
                </td>
                <td title={item.question}>{item.question}</td>
                <td className="count-cell">{item.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default HotQuestions;
