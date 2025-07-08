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
set tasksfile=tasks.md
set handofffile=logs\handoff.md

for /f "tokens=1-4 delims=:. " %%a in ('echo %time%') do (
    set hour=%%a
    set min=%%b
    set sec=%%c
)

set hour=!hour: =!
set timestamp=!hour!:!min!:!sec!

if not exist logs mkdir logs

:: Initialize tasks.md if it doesn't exist
if not exist !tasksfile! (
    echo # Tasks > !tasksfile!
    echo. >> !tasksfile!
    echo ## Active >> !tasksfile!
    echo. >> !tasksfile!
    echo ## Completed >> !tasksfile!
    echo. >> !tasksfile!
    echo ## On Hold >> !tasksfile!
    echo. >> !tasksfile!
)

:: Check if daily log exists
if exist !filename! (
    :: Add new session to existing log
    echo. >> !filename!
    echo --- >> !filename!
    echo. >> !filename!
    echo ## Session Started: !timestamp! >> !filename!
    echo. >> !filename!
    
    echo Session started at !timestamp!
    
    :: Show handoff note if exists
    if exist !handofffile! (
        echo.
        echo ===== HANDOFF NOTE FROM LAST SESSION =====
        type !handofffile!
        echo ==========================================
        echo.
        pause
    )
    
    :: Open tasks and log
    start notepad !tasksfile!
    timeout /t 1 /nobreak >nul
    notepad !filename!
    exit /b
)

:: Create new daily log
echo # Work Log - !date! > !filename!
echo. >> !filename!
echo ## Day Started: !timestamp! >> !filename!
echo. >> !filename!

echo New day! Log created at !timestamp!

:: Show handoff note if exists
if exist !handofffile! (
    echo.
    echo ===== HANDOFF NOTE FROM LAST SESSION =====
    type !handofffile!
    echo ==========================================
    echo.
    
    :: Add handoff note to new log
    echo ## Previous Session Summary >> !filename!
    type !handofffile! >> !filename!
    echo. >> !filename!
    echo --- >> !filename!
    echo. >> !filename!
    
    pause
)

:: Open tasks and log
start notepad !tasksfile!
timeout /t 1 /nobreak >nul
notepad !filename!

pause