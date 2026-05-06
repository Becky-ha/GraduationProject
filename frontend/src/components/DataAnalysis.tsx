import React, { useState } from 'react';
import FeedbackOverview from './FeedbackOverview';
import HotQuestions from './HotQuestions';
import KnowledgeAnalysis from './KnowledgeAnalysis';
import './DataAnalysis.css';

type AnalysisTab = 'feedback' | 'hot-questions' | 'knowledge-analysis';

const DataAnalysis: React.FC = () => {
  const [activeTab, setActiveTab] = useState<AnalysisTab>('feedback');

  return (
    <div className="analysis-page-container">
      <aside className="analysis-sidebar">
        <div className="sidebar-header">
          <h3>分析看板</h3>
        </div>
        <nav className="sidebar-nav">
          <button 
            className={`nav-item ${activeTab === 'feedback' ? 'active' : ''}`}
            onClick={() => setActiveTab('feedback')}
          >
            <span className="icon">📊</span> 反馈统计
          </button>
          <button 
            className={`nav-item ${activeTab === 'hot-questions' ? 'active' : ''}`}
            onClick={() => setActiveTab('hot-questions')}
          >
            <span className="icon">🔥</span> 问题统计
          </button>
          <button 
            className={`nav-item ${activeTab === 'knowledge-analysis' ? 'active' : ''}`}
            onClick={() => setActiveTab('knowledge-analysis')}
          >
            <span className="icon">📚</span> 知识库分析
          </button>
        </nav>
      </aside>

      <main className="analysis-main-content">
        <div className="tab-content-wrapper">
          {activeTab === 'feedback' && <FeedbackOverview />}
          {activeTab === 'hot-questions' && <HotQuestions />}
          {activeTab === 'knowledge-analysis' && <KnowledgeAnalysis />}
        </div>
      </main>
    </div>
  );
};

export default DataAnalysis;
