@echo off
rem Double-click or run from cmd. Extra args are passed to install.py (e.g. setup.bat --yes --test)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
pause
