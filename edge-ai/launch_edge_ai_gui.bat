@echo off
cd /d %~dp0

call env\Scripts\activate
python gui\app.py
