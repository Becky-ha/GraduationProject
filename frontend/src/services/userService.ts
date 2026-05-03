import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8088/api';

const getAuthHeaders = (): Record<string, string> => {
  const token = localStorage.getItem('token');
  return token ? { Authorization: `Bearer ${token}` } : {};
};

// 获取所有用户列表 (Admin)
export const getAllUsers = async () => {
  const response = await axios.get(`${API_BASE_URL}/admin/users`, {
    headers: getAuthHeaders()
  });
  return response.data;
};

// 封禁/解封用户 (Admin)
export const updateUserStatus = async (userId: number, status: 'active' | 'disabled') => {
  const response = await axios.post(`${API_BASE_URL}/admin/ban_user`, {
    user_id: userId,
    status: status
  }, {
    headers: getAuthHeaders()
  });
  return response.data;
};

// 获取当前用户信息
export const getMyInfo = async () => {
  const response = await axios.get(`${API_BASE_URL}/user/me`, {
    headers: getAuthHeaders()
  });
  return response.data;
};

// 修改当前用户信息
export const updateUserInfo = async (data: { username?: string, email?: string }) => {
  const response = await axios.put(`${API_BASE_URL}/user/update`, data, {
    headers: getAuthHeaders()
  });
  return response.data;
};

// 修改密码
export const changePassword = async (data: { old_password: string, new_password: string }) => {
  const response = await axios.post(`${API_BASE_URL}/user/change-password`, data, {
    headers: getAuthHeaders()
  });
  return response.data;
};
