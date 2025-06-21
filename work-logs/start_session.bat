@echo off
setlocal enabledelayedexpansion

:: Get current date in YYYY-MM-DD format
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do (
    set year=%%c
    set month=%%a
    set day=%%b
)

:: Pad single digits with zero
if !month! lss 10 set month=0!month:~-1!
if !day! lss 10 set day=0!day:~-1!

set filename=logs\!year!-!month!-!day!.md

:: Get current time
for /f "tokens=1-4 delims=:. " %%a in ('echo %time%') do (
    set hour=%%a
    set min=%%b
    set sec=%%c
)

:: Remove leading space from hour if present
set hour=!hour: =!
set timestamp=!hour!:!min!:!sec!

:: Create logs directory if it doesn't exist
if not exist logs mkdir logs

:: Check if file already exists
if exist !filename! (
    :: Add new session marker
    echo. >> !filename!
    echo --- >> !filename!
    echo. >> !filename!
    echo ## Session Started: !timestamp! >> !filename!
    echo. >> !filename!
    
    echo Session started at !timestamp!
    notepad !filename!
    exit /b
)

:: Create new log file for the day
echo # Work Log - !date! > !filename!
echo. >> !filename!
echo ## Day Started: !timestamp! >> !filename!
echo. >> !filename!
echo ## Tasks >> !filename!
echo - [ ] >> !filename!
echo - [ ] >> !filename!
echo - [ ] >> !filename!
echo. >> !filename!

:: Check for previous logs and open the most recent one
set lastlog=
for /f "tokens=*" %%f in ('dir /b /o-d logs\*.md 2^>nul') do (
    set testlog=%%f
    if not "!testlog!"=="!filename:logs\=!" (
        set lastlog=%%f
        goto :found_previous
    )
)
:found_previous

echo New day! Log created at !timestamp!

if defined lastlog (
    echo.
    echo Opening previous log (!lastlog:~0,-3!) for reference...
    start notepad "logs\!lastlog!"
    timeout /t 1 /nobreak >nul
)

:: Open today's new log
notepad !filename!

pause