import sqlite3
import os

db_path = 'ai_assistant.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 检查并添加列
    columns = [info[1] for info in cursor.execute("PRAGMA table_info(knowledge_files)").fetchall()]
    
    if 'progress' not in columns:
        cursor.execute('ALTER TABLE knowledge_files ADD COLUMN progress INTEGER DEFAULT 0')
        print("Added column 'progress'")
    
    if 'error_message' not in columns:
        cursor.execute('ALTER TABLE knowledge_files ADD COLUMN error_message TEXT')
        print("Added column 'error_message'")
    
    # 更新旧状态
    cursor.execute("UPDATE knowledge_files SET status = 'completed' WHERE status = 'active'")
    
    conn.commit()
    conn.close()
    print("Database schema updated successfully.")
else:
    print(f"Database file {db_path} not found.")
