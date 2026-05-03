import React, { useState, useEffect } from 'react';
import { getAllUsers, updateUserStatus } from '../services/userService';
import './AdminUserPage.css';

interface User {
  id: number;
  username: string;
  email: string | null;
  role: string;
  status: string;
  created_at: string | null;
}

const AdminUserPage: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const fetchUsers = async () => {
    try {
      setLoading(true);
      const data = await getAllUsers();
      setUsers(data);
    } catch (error) {
      console.error('获取用户列表失败:', error);
      alert('获取用户列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleStatusToggle = async (user: User) => {
    const newStatus = user.status === 'active' ? 'disabled' : 'active';
    const confirmMsg = `确定要${newStatus === 'disabled' ? '封禁' : '解封'}用户 "${user.username}" 吗？`;
    
    if (window.confirm(confirmMsg)) {
      try {
        await updateUserStatus(user.id, newStatus);
        fetchUsers(); // 刷新列表
      } catch (error: any) {
        alert(error.response?.data?.detail || '操作失败');
      }
    }
  };

  const showDetail = (user: User) => {
    setSelectedUser(user);
    setIsModalOpen(true);
  };

  if (loading) return <div className="admin-loading">加载中...</div>;

  return (
    <div className="admin-user-container">
      <div className="admin-user-header">
        <h1>用户管理</h1>
        <button onClick={fetchUsers} className="refresh-btn">刷新列表</button>
      </div>

      <div className="table-wrapper">
        <table className="user-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>用户名</th>
              <th>邮箱</th>
              <th>角色</th>
              <th>状态</th>
              <th>注册时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {users.map(user => (
              <tr key={user.id}>
                <td>{user.id}</td>
                <td>{user.username}</td>
                <td>{user.email || '未设置'}</td>
                <td>
                  <span className={`role-badge ${user.role}`}>
                    {user.role === 'admin' ? '管理员' : '普通用户'}
                  </span>
                </td>
                <td>
                  <span className={`status-badge ${user.status}`}>
                    {user.status === 'active' ? '活跃' : '已封禁'}
                  </span>
                </td>
                <td>{user.created_at ? new Date(user.created_at).toLocaleString() : '-'}</td>
                <td>
                  <div className="action-btns">
                    <button onClick={() => showDetail(user)} className="detail-btn">详情</button>
                    <button 
                      onClick={() => handleStatusToggle(user)} 
                      className={`status-btn ${user.status === 'active' ? 'ban' : 'unban'}`}
                    >
                      {user.status === 'active' ? '封禁' : '解封'}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {isModalOpen && selectedUser && (
        <div className="modal-overlay" onClick={() => setIsModalOpen(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2>用户详情</h2>
            <div className="detail-grid">
              <div className="detail-item"><label>ID:</label> <span>{selectedUser.id}</span></div>
              <div className="detail-item"><label>用户名:</label> <span>{selectedUser.username}</span></div>
              <div className="detail-item"><label>邮箱:</label> <span>{selectedUser.email || '-'}</span></div>
              <div className="detail-item"><label>角色:</label> <span>{selectedUser.role}</span></div>
              <div className="detail-item"><label>状态:</label> <span>{selectedUser.status}</span></div>
              <div className="detail-item"><label>注册时间:</label> <span>{selectedUser.created_at}</span></div>
            </div>
            <button className="close-modal" onClick={() => setIsModalOpen(false)}>关闭</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminUserPage;
