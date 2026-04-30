import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import './Login.css';

const Register: React.FC = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [role, setRole] = useState('student');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    if (password.length < 6) {
      setError('密码长度不能小于 6 位');
      return;
    }
    
    if (password !== confirmPassword) {
      setError('两次密码输入不一致');
      return;
    }

    setIsLoading(true);

    try {
      console.log('正在发送注册请求到: http://127.0.0.1:8088/api/register');
      const response = await axios.post('http://127.0.0.1:8088/api/register', {
        username,
        email,
        password,
        role, // 发送角色
      });
      console.log('注册响应成功:', response.data);

      setSuccess('注册成功，正在跳转登录页...');
      setTimeout(() => {
        navigate('/login');
      }, 1500);
    } catch (err: any) {
      setError(err.response?.data?.detail || '注册失败，请尝试其他用户名');
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
        
        <h2 className="login-title">用户注册</h2>
        
        {error && <div className="error-message">{error}</div>}
        {success && <div className="success-message">{success}</div>}
        
        <form onSubmit={handleRegister}>
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
          <div className="form-group">
            <input
              type="email"
              id="email"
              placeholder="请输入邮箱"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className="form-group password-group">
            <div className="password-input-wrapper">
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                className={password.length > 0 && password.length < 6 ? 'input-error' : ''}
                placeholder="请输入密码（至少6位）"
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
            {password.length > 0 && password.length < 6 && (
              <p className="field-hint">密码长度不能小于 6 位</p>
            )}
          </div>
          <div className="form-group">
            <div className="password-input-wrapper">
              <input
                type={showPassword ? 'text' : 'password'}
                id="confirmPassword"
                placeholder="请再次输入密码"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </div>
          </div>
          <div className="form-group">
            <select
              id="role"
              value={role}
              onChange={(e) => setRole(e.target.value)}
              className="role-select"
            >
              <option value="student">普通用户</option>
              <option value="admin">管理员</option>
            </select>
          </div>
          <button type="submit" className="login-submit-btn" disabled={isLoading}>
            {isLoading ? '注册中...' : '注册'}
          </button>
        </form>
        
        <div className="auth-footer">
          <span className="footer-text">已有账号？</span>
          <Link to="/login" className="footer-link">立即登录</Link>
        </div>
      </div>
    </div>
  );
};

export default Register;
