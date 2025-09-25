# Text-to-SQL API

This project is a **FastAPI-based service** that converts natural language questions into SQL queries, executes them on connected databases, and returns human-readable answers.  
It uses **LangChain** and **LlamaCpp** models for query generation and answer rephrasing.  

---

## 🚀 Features
- Connect to multiple databases (MySQL, PostgreSQL, SQLite, Oracle, SQL Server).  
- Ask natural language questions → Get SQL + final answer.  
- Conversation memory for follow-up queries.  
- Pre-configured prompt templates per database dialect.  
- REST API endpoints for integration with any frontend.  

---

## 📦 Requirements
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Supported databases require additional drivers:
- **MySQL** → `pymysql`  
- **PostgreSQL** → `psycopg2-binary`  
- **SQLite** → works with built-in driver  
- **Oracle** → `cx_Oracle`  
- **SQL Server** → `pyodbc`  

---

## ⚙️ Configuration
Update the model paths in `nicsi_db_gpt.py`:

```python
model_path = "path to text to sql model"
dst_path   = "path to nlp chatting model"
```

Replace with the correct `.gguf` paths for your system.  

---

## ▶️ Running the API
Run the FastAPI app with Uvicorn:

```bash
uvicorn nicsi_db_gpt:app --reload --host 0.0.0.0 --port 8000
```

Open docs in your browser:  
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  

---

## 📡 API Endpoints
### 1. Health Check
```
GET /health
```

### 2. Connect to Database
```
POST /connect-database
```
Example request:
```json
{
  "db_type": "mysql",
  "db_user": "root",
  "db_password": "password",
  "db_host": "localhost",
  "db_port": 3306,
  "db_name": "mydb",
  "table_names": ["users", "orders"]
}
```

### 3. Ask Question
```
POST /ask-question
```
Example request:
```json
{
  "session_id": "mysql_localhost_mydb_0",
  "question": "What are the first 5 users?"
}
```

### 4. Manage Sessions
- `GET /sessions` → list active sessions  
- `DELETE /session/{session_id}` → close a session  

---

## 🛠️ Development Notes
- Uses **LangChain Runnables** to build the pipeline:  
  Question → SQL → Query Execution → Answer Rephrasing.  
- Supports **follow-up queries** with conversation memory.  
- Customize prompts in `get_sql_prompt_template` and `answer_prompt`.  

---

## 📜 License
This project is licensed under the **MIT License**.  
