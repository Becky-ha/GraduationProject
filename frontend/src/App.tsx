import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import './App.css';
import ChatInterface from './components/ChatInterface';
import Login from './components/Login';
import Register from './components/Register';
import ForgotPassword from './components/ForgotPassword';
import KnowledgeBaseManager from './components/KnowledgeBaseManager';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [username, setUsername] = useState<string | null>(null);
  const [userRole, setUserRole] = useState<string | null>(null);

  useEffect(() => {
    const syncAuthState = () => {
      const token = localStorage.getItem('token');
      setIsAuthenticated(!!token);
      setUsername(localStorage.getItem('username'));
      setUserRole(localStorage.getItem('role'));
    };

    const handleVisibilityAndFocus = () => {
      syncAuthState();
      const token = localStorage.getItem('token');
      if (!token) {
        setIsAuthenticated(false);
        setUsername(null);
        setUserRole(null);
      }
    };

    syncAuthState();
    window.addEventListener('storage', syncAuthState);
    window.addEventListener('focus', handleVisibilityAndFocus);
    document.addEventListener('visibilitychange', handleVisibilityAndFocus);
    return () => {
      window.removeEventListener('storage', syncAuthState);
      window.removeEventListener('focus', handleVisibilityAndFocus);
      document.removeEventListener('visibilitychange', handleVisibilityAndFocus);
    };
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    localStorage.removeItem('role');
    localStorage.removeItem('aiCourseAssistantConversationId');
    setIsAuthenticated(false);
    setUsername(null);
    setUserRole(null);
    window.location.href = '/login';
  };

  return (
    <Router>
      <div className="App">
        {isAuthenticated && (
          <header className="App-header-minimal">
            <div className="header-left">
              <span className="logo-icon">🤖</span>
              <span className="app-name">QAI Bot</span>
              <Link to="/" className="nav-link">对话</Link>
              {userRole === 'admin' && (
                <Link to="/admin/knowledge" className="nav-link">知识库管理</Link>
              )}
            </div>
            <div className="user-menu">
              <span className="username">欢迎, {username}</span>
              <button onClick={handleLogout} className="logout-button">退出登录</button>
            </div>
          </header>
        )}
        <main className={!isAuthenticated ? 'auth-main' : (window.location.pathname.includes('/admin') ? 'admin-main' : 'chat-main')}>
          <Routes>
            <Route 
              path="/" 
              element={isAuthenticated ? (
                <div className="chat-interface-wrapper">
                  <ChatInterface onLogout={handleLogout} />
                </div>
              ) : <Navigate to="/login" />} 
            />
            <Route 
              path="/admin/knowledge" 
              element={isAuthenticated && userRole === 'admin' ? <KnowledgeBaseManager /> : <Navigate to="/" />} 
            />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
