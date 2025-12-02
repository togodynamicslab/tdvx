@echo off
REM Set environment variable to fix OpenMP library conflict
set KMP_DUPLICATE_LIB_OK=TRUE

REM Activate virtual environment and start server
cd /d %~dp0
call venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
