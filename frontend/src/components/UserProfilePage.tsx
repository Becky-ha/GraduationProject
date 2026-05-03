import React, { useState, useEffect } from 'react';
import { getMyInfo, updateUserInfo, changePassword } from '../services/userService';
import './UserProfilePage.css';

interface UserProfile {
  id: number;
  username: string;
  email: string | null;
  role: string;
  status: string;
  created_at: string | null;
}

type ModalType = 'none' | 'username' | 'email' | 'password';

const UserProfilePage: React.FC = () => {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeModal, setActiveModal] = useState<ModalType>('none');
  
  // 表单状态
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    oldPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  
  const [message, setMessage] = useState<{ text: string, type: 'success' | 'error' } | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const fetchProfile = async () => {
    try {
      setLoading(true);
      const data = await getMyInfo();
      setUser(data);
      setFormData(prev => ({
        ...prev,
        username: data.username,
        email: data.email || ''
      }));
    } catch (error) {
      console.error('获取个人信息失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProfile();
  }, []);

  const handleCloseModal = () => {
    setActiveModal('none');
    setMessage(null);
    setFormData(prev => ({
      ...prev,
      oldPassword: '',
      newPassword: '',
      confirmPassword: ''
    }));
  };

  const showMessage = (text: string, type: 'success' | 'error') => {
    setMessage({ text, type });
    if (type === 'success') {
      setTimeout(() => handleCloseModal(), 1500);
    }
  };

  const handleUpdateUsername = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.username.trim()) return showMessage('用户名不能为空', 'error');
    
    setSubmitting(true);
    try {
      await updateUserInfo({ username: formData.username });
      // 如果用户名变了，更新本地存储以保持顶部导航同步
      localStorage.setItem('username', formData.username);
      showMessage('用户名修改成功！', 'success');
      fetchProfile();
    } catch (error: any) {
      showMessage(error.response?.data?.detail || '修改失败', 'error');
    } finally {
      setSubmitting(false);
    }
  };

  const handleUpdateEmail = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.email.trim()) return showMessage('邮箱不能为空', 'error');
    
    setSubmitting(true);
    try {
      await updateUserInfo({ email: formData.email });
      showMessage('邮箱修改成功！', 'success');
      fetchProfile();
    } catch (error: any) {
      showMessage(error.response?.data?.detail || '修改失败', 'error');
    } finally {
      setSubmitting(false);
    }
  };

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.oldPassword || !formData.newPassword) return showMessage('密码不能为空', 'error');
    if (formData.newPassword !== formData.confirmPassword) return showMessage('两次输入的新密码不一致', 'error');
    
    setSubmitting(true);
    try {
      await changePassword({ 
        old_password: formData.oldPassword, 
        new_password: formData.newPassword 
      });
      showMessage('密码修改成功！', 'success');
    } catch (error: any) {
      showMessage(error.response?.data?.detail || '修改失败', 'error');
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) return <div className="profile-loading">加载中...</div>;
  if (!user) return <div className="profile-error">无法加载用户信息</div>;

  return (
    <div className="profile-container">
      <div className="profile-card">
        <h1>个人中心</h1>
        
        <div className="info-list">
          <div className="info-item">
            <div className="info-left">
              <span className="label">用户名</span>
              <span className="value">{user.username}</span>
            </div>
            <button className="edit-btn" onClick={() => setActiveModal('username')}>修改</button>
          </div>

          <div className="info-item">
            <div className="info-left">
              <span className="label">邮箱地址</span>
              <span className="value">{user.email || '未设置'}</span>
            </div>
            <button className="edit-btn" onClick={() => setActiveModal('email')}>修改</button>
          </div>

          <div className="info-item">
            <div className="info-left">
              <span className="label">登录密码</span>
              <span className="value password-dots">******</span>
            </div>
            <button className="edit-btn" onClick={() => setActiveModal('password')}>修改</button>
          </div>

          <div className="info-item no-border">
            <div className="info-left">
              <span className="label">账户角色</span>
              <span className="value role-text">{user.role === 'admin' ? '管理员' : '普通用户'}</span>
            </div>
          </div>

          <div className="info-item no-border">
            <div className="info-left">
              <span className="label">注册时间</span>
              <span className="value date-text">
                {user.created_at ? new Date(user.created_at).toLocaleString() : '-'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 修改用户名弹窗 */}
      {activeModal === 'username' && (
        <div className="modal-overlay" onClick={handleCloseModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h3>修改用户名</h3>
            {message && <div className={`modal-msg ${message.type}`}>{message.text}</div>}
            <form onSubmit={handleUpdateUsername}>
              <div className="form-item">
                <label>新用户名</label>
                <input 
                  type="text" 
                  value={formData.username} 
                  onChange={e => setFormData({...formData, username: e.target.value})}
                  autoFocus
                />
              </div>
              <div className="modal-actions">
                <button type="button" onClick={handleCloseModal} className="btn-cancel">取消</button>
                <button type="submit" className="btn-confirm" disabled={submitting}>
                  {submitting ? '提交中...' : '确认修改'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* 修改邮箱弹窗 */}
      {activeModal === 'email' && (
        <div className="modal-overlay" onClick={handleCloseModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h3>修改邮箱</h3>
            {message && <div className={`modal-msg ${message.type}`}>{message.text}</div>}
            <form onSubmit={handleUpdateEmail}>
              <div className="form-item">
                <label>新邮箱地址</label>
                <input 
                  type="email" 
                  value={formData.email} 
                  onChange={e => setFormData({...formData, email: e.target.value})}
                  autoFocus
                />
              </div>
              <div className="modal-actions">
                <button type="button" onClick={handleCloseModal} className="btn-cancel">取消</button>
                <button type="submit" className="btn-confirm" disabled={submitting}>
                  {submitting ? '提交中...' : '确认修改'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* 修改密码弹窗 */}
      {activeModal === 'password' && (
        <div className="modal-overlay" onClick={handleCloseModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h3>修改密码</h3>
            {message && <div className={`modal-msg ${message.type}`}>{message.text}</div>}
            <form onSubmit={handleChangePassword}>
              <div className="form-item">
                <label>当前密码</label>
                <input 
                  type="password" 
                  value={formData.oldPassword} 
                  onChange={e => setFormData({...formData, oldPassword: e.target.value})}
                  placeholder="请输入当前使用的密码"
                />
              </div>
              <div className="form-item">
                <label>新密码</label>
                <input 
                  type="password" 
                  value={formData.newPassword} 
                  onChange={e => setFormData({...formData, newPassword: e.target.value})}
                  placeholder="请输入新密码"
                />
              </div>
              <div className="form-item">
                <label>确认新密码</label>
                <input 
                  type="password" 
                  value={formData.confirmPassword} 
                  onChange={e => setFormData({...formData, confirmPassword: e.target.value})}
                  placeholder="请再次输入新密码"
                />
              </div>
              <div className="modal-actions">
                <button type="button" onClick={handleCloseModal} className="btn-cancel">取消</button>
                <button type="submit" className="btn-confirm" disabled={submitting}>
                  {submitting ? '提交中...' : '确认修改'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default UserProfilePage;
