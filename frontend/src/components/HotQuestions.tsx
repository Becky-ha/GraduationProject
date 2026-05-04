import React, { useState, useEffect } from 'react';
import { getHotQuestions } from '../services/chatService';

interface HotQuestionItem {
  represent_question: string;
  count: number;
  examples: string[];
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

  return (
    <div className="analysis-sub-content">
      <div className="section-header">
        <h2>高频问题</h2>
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
                <th style={{ width: '80px' }}>排名</th>
                <th>意图/代表性问题</th>
                <th>相似提问示例</th>
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
                  <td title={item.represent_question} style={{ fontWeight: 'bold' }}>{item.represent_question}</td>
                  <td>
                    <div className="example-list">
                      {item.examples.map((ex, i) => (
                        <div key={i} className="example-item">• {ex}</div>
                      ))}
                    </div>
                  </td>
                  <td className="count-cell">{item.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default HotQuestions;
