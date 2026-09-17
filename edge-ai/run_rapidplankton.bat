@echo off
title Rapid Plankton

echo(
echo    Loading plankton gui
echo(

echo           .-.
echo        '-( o )-'
echo       (          )
echo        '-(    )-'
echo       (          )
echo        '-(____)-'
echo             \
echo              \~~~~
echo(

cd /d %~dp0
call env\Scripts\activate
python gui\app.py