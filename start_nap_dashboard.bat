@echo off
setlocal
cd /d "%~dp0"
"D:\Anacoda\Scripts\streamlit.exe" run src\nap_dashboard.py --server.port 8522 --browser.gatherUsageStats false
