@echo off
setlocal enabledelayedexpansion

:: Get current date and time
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do (
    set year=%%c
    set month=%%a
    set day=%%b
)

if !month! lss 10 set month=0!month:~-1!
if !day! lss 10 set day=0!day:~-1!

set filename=logs\!year!-!month!-!day!.md
set handofffile=logs\handoff.md
set tempfile=logs\temp_handoff.txt

if not exist !filename! (
    echo No log file found for today. Run start-session.bat first.
    pause
    exit /b
)

for /f "tokens=1-4 delims=:. " %%a in ('echo %time%') do (
    set hour=%%a
    set min=%%b
    set sec=%%c
)

set hour=!hour: =!
set timestamp=!hour!:!min!:!sec!

:: Add session end template to log
echo. >> !filename!
echo ### Session Ended: !timestamp! >> !filename!
echo. >> !filename!
echo #### What I accomplished: >> !filename!
echo - >> !filename!
echo. >> !filename!
echo #### Handoff note for next session: >> !filename!
echo [HANDOFF_START] >> !filename!
echo. >> !filename!
echo [HANDOFF_END] >> !filename!

:: Open log for editing
echo.
echo Please document:
echo 1. What you accomplished this session
echo 2. IMPORTANT: Write your handoff note between [HANDOFF_START] and [HANDOFF_END]
echo.
notepad !filename!

:: Extract handoff note automatically
echo Extracting handoff note...
set in_handoff=0
type nul > !tempfile!

for /f "tokens=*" %%a in ('type !filename!') do (
    set line=%%a
    if "!line!"=="[HANDOFF_START]" set in_handoff=1
    if !in_handoff!==1 if not "!line!"=="[HANDOFF_START]" if not "!line!"=="[HANDOFF_END]" echo %%a >> !tempfile!
    if "!line!"=="[HANDOFF_END]" set in_handoff=0
)

:: Create handoff file
echo # Handoff from !date! at !timestamp! > !handofffile!
echo. >> !handofffile!
type !tempfile! >> !handofffile!
del !tempfile!

echo.
echo Session ended at !timestamp!
echo Handoff note saved for next session.
echo.
echo Don't forget to update tasks.md if you completed any tasks!
choice /C YN /M "Open tasks.md now?"
if errorlevel 1 if not errorlevel 2 notepad tasks.md