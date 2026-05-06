import React, { useState, useEffect } from 'react';
import { getHotQuestions } from '../services/chatService';

interface HotQuestionItem {
  cluster_id?: number;
  represent_question: string;
  count: number;
  examples: string[];
}

const HotQuestions: React.FC = () => {
  const [data, setData] = useState<HotQuestionItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedClusters, setExpandedClusters] = useState<Set<number>>(new Set());

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
      setError('加载高频问题分类失败');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleExpand = (index: number) => {
    const newExpanded = new Set(expandedClusters);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedClusters(newExpanded);
  };

  return (
    <div className="analysis-sub-content">
      <div className="section-header">
        <div />
        <button className="refresh-mini-btn" onClick={fetchData}>同步最新数据</button>
      </div>
      <div className="table-wrapper">
        {isLoading ? (
          <div className="sub-loading">正在进行语义聚类分析...</div>
        ) : error ? (
          <div className="sub-error">{error}</div>
        ) : (
          <div className="clusters-grid">
            {data.map((item, index) => {
              const isExpanded = expandedClusters.has(index);
              const displayExamples = isExpanded ? item.examples : item.examples.slice(0, 3);
              
              return (
                <div key={index} className="cluster-card">
                  <div className="cluster-card-header">
                    <h4 className="cluster-title">
                      <span className="icon">🤖</span>
                      {item.represent_question}
                    </h4>
                    <span className="cluster-count">🔥 出现 {item.count} 次</span>
                  </div>
                  
                  <div className="cluster-examples">
                    <span className="example-label">相似提问示例：</span>
                    <div className="example-tags">
                      {displayExamples.map((ex, i) => (
                        <div key={i} className="example-tag">{ex}</div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="cluster-card-footer">
                    {item.examples.length > 3 && (
                      <button className="btn-expand" onClick={() => toggleExpand(index)}>
                        {isExpanded ? '收起' : `展开更多 (${item.examples.length - 3})`}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default HotQuestions;
