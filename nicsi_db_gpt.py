from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os
import re
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.llms import LlamaCpp
from langchain_community.tools import QuerySQLDatabaseTool
from langchain.memory import ConversationBufferMemory
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text-to-SQL API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (loaded once at startup)
llm1 = None  # Text2SQL model
llm2 = None  # Answer generation model

# Store active database connections and memories per session
active_sessions: Dict[str, Dict] = {}

# Pydantic models for request/response
class DatabaseConfig(BaseModel):
    db_type: str = Field(..., description="Database type: mysql, postgresql, sqlite, oracle, mssql")
    db_user: str = Field(default="", description="Database username (not required for SQLite)")
    db_password: str = Field(default="", description="Database password (not required for SQLite)")
    db_host: str = Field(default="localhost", description="Database host (not required for SQLite)")
    db_port: Optional[int] = Field(default=None, description="Database port (optional)")
    db_name: str = Field(..., description="Database name or file path for SQLite")
    table_names: List[str] = Field(..., description="List of table names to include")


class ReConfigDB(BaseModel):
    db_type: str = Field(..., description="Database type: mysql, postgresql, sqlite, oracle, mssql")
    db_user: str = Field(default="", description="Database username (not required for SQLite)")
    db_password: str = Field(default="", description="Database password (not required for SQLite)")
    db_host: str = Field(default="localhost", description="Database host (not required for SQLite)")
    db_port: Optional[int] = Field(default=None, description="Database port (optional)")
    db_name: str = Field(..., description="Database name or file path for SQLite")
    table_names: List[str] = Field(..., description="List of table names to include")
    session_id: str = Field(..., description="Session ID to reconnect")
    
class QuestionRequest(BaseModel):
    session_id: str
    question: str

class DatabaseResponse(BaseModel):
    session_id: str
    status: str
    message: str
    table_info: Optional[str] = None

class AnswerResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    generated_sql: str
    status: str

# Initialize models at startup
@app.on_event("startup")
async def load_models():
    global llm1, llm2
    
    try:
        # Update these paths according to your setup
        model_path = "D:/models/text2sql/models--tensorblock--Text2SQL-1.5B-GGUF/snapshots/a905daffc22ca5a83083f669b9020ae287bb5d88/text2sql-1.5b-q4_K_M.gguf"
        dst_path = "D:/models/google gemma/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
        
        logger.info("Loading Text2SQL model...")
        llm1 = LlamaCpp(
            model_path=model_path,
            n_threads=6,
            n_ctx=4096,
            temperature=0,
            max_tokens=128
        )
        
        logger.info("Loading Answer generation model...")
        llm2 = LlamaCpp(
            model_path=dst_path,
            n_threads=6,
            n_ctx=4096,
            temperature=0,
            max_tokens=3000
        )
        
        logger.info("Models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

def create_database_uri(config: DatabaseConfig) -> str:
    """Create database URI based on database type and configuration"""
    db_type = config.db_type.lower()
    
    if db_type == "sqlite":
        # For SQLite, db_name should be the file path
        return f"sqlite:///{config.db_name}"
    
    elif db_type == "mysql":
        port = config.db_port or 3306
        return f"mysql+pymysql://{config.db_user}:{config.db_password}@{config.db_host}:{port}/{config.db_name}"
    
    elif db_type == "postgresql":
        port = config.db_port or 5432
        return f"postgresql+psycopg2://{config.db_user}:{config.db_password}@{config.db_host}:{port}/{config.db_name}"
    
    elif db_type == "oracle":
        port = config.db_port or 1521
        return f"oracle+cx_oracle://{config.db_user}:{config.db_password}@{config.db_host}:{port}/{config.db_name}"
    
    elif db_type == "mssql":
        port = config.db_port or 1433
        return f"mssql+pyodbc://{config.db_user}:{config.db_password}@{config.db_host}:{port}/{config.db_name}?driver=ODBC+Driver+17+for+SQL+Server"
    
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_sql_prompt_template(db_type: str) -> PromptTemplate:
    """Get SQL prompt template based on database type"""
    db_type = db_type.lower()
    
    if db_type == "mysql":
        dialect_info = "MySQL"
        specific_instructions = "- Use backticks for table/column names if they contain spaces or special characters.\n- Use LIMIT for result limiting."
    elif db_type == "postgresql":
        dialect_info = "PostgreSQL"
        specific_instructions = "- Use double quotes for table/column names if they contain spaces or special characters.\n- Use LIMIT for result limiting.\n- Use OFFSET for pagination."
    elif db_type == "sqlite":
        dialect_info = "SQLite"
        specific_instructions = "- Use square brackets or double quotes for table/column names if needed.\n- Use LIMIT for result limiting.\n- Use OFFSET for pagination."
    elif db_type == "oracle":
        dialect_info = "Oracle"
        specific_instructions = "- Use double quotes for table/column names if they contain spaces or special characters.\n- Use ROWNUM or FETCH FIRST for result limiting."
    elif db_type == "mssql":
        dialect_info = "SQL Server"
        specific_instructions = "- Use square brackets for table/column names if they contain spaces or special characters.\n- Use TOP for result limiting.\n- Use OFFSET...FETCH for pagination."
    else:
        dialect_info = "SQL"
        specific_instructions = "- Follow standard SQL syntax."

    return PromptTemplate.from_template(
f"""You are a {dialect_info} expert. ONLY output a single valid SQL statement that begins with SELECT, INSERT, UPDATE, or DELETE.
- Output exactly one SQL statement and nothing else.
- End the statement with a semicolon.
- Do NOT include CREATE TABLE, sample data, schema dumps, explanations, or comments.
- If this is a follow-up (like 'remaining rows'), use appropriate pagination syntax for {dialect_info}.
{specific_instructions}
- Use the schema below only to help write column/table names.

Conversation so far:
{{chat_history}}

Schema:
{{table_info}}
x
Question: {{input}}
"""
    )

answer_prompt = PromptTemplate.from_template(
    """You are given a user question, a SQL query, and the SQL result. 
Return **only** the final answer to the question as a short human-readable sentence, 
without any explanation, details, or restating the query.

Conversation so far:
{chat_history}

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:"""
)

def extract_sql(text: str) -> str:
    """Extract SQL statement from model output"""
    # Capture first statement starting with SELECT|INSERT|UPDATE|DELETE (case-insensitive)
    m = re.search(r"(?is)\b(SELECT|INSERT|UPDATE|DELETE)\b.*?;", text)
    if m:
        return m.group(0).strip()
    
    # Fallback: capture if there's a statement but no semicolon
    m2 = re.search(r"(?is)\b(SELECT|INSERT|UPDATE|DELETE)\b.*", text)
    if m2:
        return m2.group(0).strip()
    
    # No SQL found
    return text.strip()

@app.post("/connect-database", response_model=DatabaseResponse)
async def connect_database(config: DatabaseConfig):
    """Connect to database and initialize session"""
    try:
        session_id = f"{config.db_type}_{config.db_host}_{config.db_name}_{len(active_sessions)}_{uuid.uuid4()}"
        
        # Create database connection URI based on database type
        db_uri = create_database_uri(config)
        db = SQLDatabase.from_uri(db_uri, include_tables=config.table_names)
        
        # Test connection
        table_names = db.get_usable_table_names()
        if not table_names:
            raise HTTPException(status_code=400, detail="No accessible tables found")
        
        # Initialize memory for this session
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Store session data
        active_sessions[session_id] = {
            "db": db,
            "memory": memory,
            "table_names": table_names,
            "db_type": config.db_type
        }
        
        logger.info(f"Database connected for session {session_id}")
        logger.info(f"Available tables: {table_names}")
        
        return DatabaseResponse(
            session_id=session_id,
            status="success",
            message=f"Connected successfully. Available tables: {', '.join(table_names)}",
            table_info=db.table_info
        )
        
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=400, detail=f"Database connection failed: {str(e)}")
    




@app.post("/reconnect-database", response_model=DatabaseResponse)
async def reconnect_database(config: ReConfigDB):
    """Connect to database and initialize session"""
    try:
        session_id = config.session_id
        
        # Create database connection URI based on database type
        db_uri = create_database_uri(config)
        db = SQLDatabase.from_uri(db_uri, include_tables=config.table_names)
        
        # Test connection
        table_names = db.get_usable_table_names()
        if not table_names:
            raise HTTPException(status_code=400, detail="No accessible tables found")
        
        # Initialize memory for this session
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Store session data
        active_sessions[session_id] = {
            "db": db,
            "memory": memory,
            "table_names": table_names,
            "db_type": config.db_type
        }
        
        logger.info(f"Database connected for session {session_id}")
        logger.info(f"Available tables: {table_names}")
        
        return DatabaseResponse(
            session_id=session_id,
            status="reconnected successfully",
            message=f"Connected successfully. Available tables: {', '.join(table_names)}",
            table_info=db.table_info
        )
        
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=400, detail=f"Database connection failed: {str(e)}")
    



@app.post("/ask-question", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Process user question and return SQL result"""
    try:
        # Check if session exists
        if request.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found. Please connect to database first.")
        
        session_data = active_sessions[request.session_id]
        db = session_data["db"]
        memory = session_data["memory"]
        db_type = session_data["db_type"]
        
        # Handle follow-up questions based on database type
        question = request.question
        if re.search(r"\bremaining rows\b", question, re.I):
            if db_type.lower() in ["mysql", "postgresql", "sqlite"]:
                question = "Return all rows from the table except the first 2 (use OFFSET 2)."
            elif db_type.lower() == "mssql":
                question = "Return all rows from the table except the first 2 (use OFFSET 2 ROWS FETCH NEXT rows)."
            elif db_type.lower() == "oracle":
                question = "Return all rows from the table except the first 2 (use ROWNUM or OFFSET)."
        
        # Add user message to memory
        memory.chat_memory.add_user_message(question)
        
        # Prepare inputs
        inputs = {
            "input": question,
            "table_info": db.table_info,
            "question": question,
            "chat_history": memory.load_memory_variables({})["chat_history"]
        }
        
        # Create processing pipeline with database-specific prompt
        sql_prompt = get_sql_prompt_template(db_type)
        extract_sql_runnable = RunnableLambda(lambda x: extract_sql(x))
        generate_query = sql_prompt | llm1
        clean_query = generate_query | extract_sql_runnable
        execute_query = QuerySQLDatabaseTool(db=db)
        rephrase_answer = answer_prompt | llm2 | StrOutputParser()
        
        # Build and execute pipeline
        chain = (
            RunnablePassthrough.assign(query=clean_query)
            .assign(result=itemgetter("query") | execute_query)
            | rephrase_answer
        )
        
        # Execute the chain
        result = chain.invoke(inputs)
        
        # Get the generated SQL (need to run the query generation part again to capture it)
        generated_sql_output = generate_query.invoke(inputs)
        generated_sql = extract_sql(generated_sql_output)
        
        # Add AI response to memory
        memory.chat_memory.add_ai_message(result)
        
        logger.info(f"Question processed for session {request.session_id}")
        logger.info(f"Generated SQL: {generated_sql}")
        
        return AnswerResponse(
            session_id=request.session_id,
            question=question,
            answer=result,
            generated_sql=generated_sql,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/supported-databases")
async def get_supported_databases():
    """Get list of supported database types"""
    return {
        "supported_databases": [
            {
                "type": "mysql",
                "name": "MySQL",
                "default_port": 3306,
                "requires_credentials": True,
                "description": "MySQL database server"
            },
            {
                "type": "postgresql", 
                "name": "PostgreSQL",
                "default_port": 5432,
                "requires_credentials": True,
                "description": "PostgreSQL database server"
            },
            {
                "type": "sqlite",
                "name": "SQLite",
                "default_port": None,
                "requires_credentials": False,
                "description": "SQLite database file (provide full file path)"
            },
            {
                "type": "oracle",
                "name": "Oracle",
                "default_port": 1521,
                "requires_credentials": True,
                "description": "Oracle database server"
            },
            {
                "type": "mssql",
                "name": "SQL Server",
                "default_port": 1433,
                "requires_credentials": True,
                "description": "Microsoft SQL Server"
            }
        ]
    }

@app.get("/sessions")
async def get_active_sessions():
    """Get list of active sessions"""
    sessions = []
    for session_id, data in active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "table_names": data["table_names"],
            "db_type": data["db_type"]
        })
    return {"active_sessions": sessions}

@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    """Close a database session"""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": f"Session {session_id} closed successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": llm1 is not None and llm2 is not None,
        "active_sessions": len(active_sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)