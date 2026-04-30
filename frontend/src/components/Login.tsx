import React, { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import './Login.css';

const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      navigate('/');
    }
  }, [navigate]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      console.log('正在发送登录请求到: http://127.0.0.1:8088/api/login');
      const response = await axios.post('http://127.0.0.1:8088/api/login', {
        username,
        password,
      });
      console.log('登录响应成功:', response.data);

      const { access_token, username: loggedInUsername, role } = response.data;
      
      // 保存到本地存储
      localStorage.setItem('token', access_token);
      localStorage.setItem('username', loggedInUsername);
      localStorage.setItem('role', role);
      
      // 强制刷新应用状态
      window.location.href = '/';
    } catch (err: any) {
      setError(err.response?.data?.detail || '登录失败，请检查用户名和密码');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header-section">
          <h1 className="login-app-name">QAI Bot</h1>
          <p className="login-app-subtitle">智能辅助问答助手</p>
        </div>
        
        <h2 className="login-title">用户登录</h2>
        
        {error && <div className="error-message">{error}</div>}
        
        <form onSubmit={handleLogin}>
          <div className="form-group">
            <input
              type="text"
              id="username"
              placeholder="请输入用户名"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div className="form-group password-group">
            <div className="password-input-wrapper">
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                placeholder="请输入密码"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <button 
                type="button" 
                className="password-toggle" 
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>
                )}
              </button>
            </div>
          </div>
          <button type="submit" className="login-submit-btn" disabled={isLoading}>
            {isLoading ? '登录中...' : '登录'}
          </button>
        </form>
        
        <div className="auth-footer">
          <span className="footer-text">还没有账号？</span>
          <Link to="/register" className="footer-link">立即注册</Link>
          <span className="footer-divider">|</span>
          <Link to="/forgot-password" style={{ color: '#a0aec0', textDecoration: 'none', fontSize: '0.85rem' }}>忘记密码？</Link>
        </div>
      </div>
    </div>
  );
};

export default Login;
