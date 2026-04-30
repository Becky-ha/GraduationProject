import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import './Login.css';

const ForgotPassword: React.FC = () => {
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [step, setStep] = useState(1); // 1: 输入邮箱, 2: 输入验证码和新密码
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  // 请求发送验证码
  const handleRequestCode = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:8088/api/forgot-password', { email });
      setSuccess('验证码发送请求已处理。请检查后端控制台查看模拟邮件内容（6位数字）');
      setStep(2);
    } catch (err: any) {
      const errorDetail = err.response?.data?.detail;
      setError(errorDetail || '发送失败，请检查邮箱地址');
    } finally {
      setIsLoading(false);
    }
  };

  // 重置密码
  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (newPassword !== confirmPassword) {
      setError('两次密码输入不一致');
      return;
    }

    setIsLoading(true);

    try {
      await axios.post('http://127.0.0.1:8088/api/reset-password', {
        email,
        code,
        new_password: newPassword,
      });
      setSuccess('密码重置成功！正在跳转登录页...');
      setTimeout(() => {
        navigate('/login');
      }, 2000);
    } catch (err: any) {
      setError(err.response?.data?.detail || '重置失败，验证码错误或已过期');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <h2>找回密码</h2>
        {error && <div className="error-message">{error}</div>}
        {success && <div className="success-message">{success}</div>}

        {step === 1 ? (
          <form onSubmit={handleRequestCode}>
            <div className="form-group">
              <label htmlFor="email">请输入注册时使用的邮箱</label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <button type="submit" disabled={isLoading}>
              {isLoading ? '发送中...' : '发送验证码'}
            </button>
          </form>
        ) : (
          <form onSubmit={handleResetPassword}>
            <div className="form-group">
              <label htmlFor="code">验证码</label>
              <input
                type="text"
                id="code"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="请输入6位验证码"
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="newPassword">新密码</label>
              <input
                type="password"
                id="newPassword"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="confirmPassword">确认新密码</label>
              <input
                type="password"
                id="confirmPassword"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </div>
            <button type="submit" disabled={isLoading}>
              {isLoading ? '重置中...' : '重置密码'}
            </button>
          </form>
        )}
        
        <p className="auth-link">
          <Link to="/login">返回登录</Link>
        </p>
      </div>
    </div>
  );
};

export default ForgotPassword;
