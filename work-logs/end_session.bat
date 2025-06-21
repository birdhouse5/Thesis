@echo off
setlocal enabledelayedexpansion

:: Get current date and time
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do (
    set year=%%c
    set month=%%a
    set day=%%b
)

:: Pad single digits with zero
if !month! lss 10 set month=0!month:~-1!
if !day! lss 10 set day=0!day:~-1!

set filename=logs\!year!-!month!-!day!.md

:: Check if today's log exists
if not exist !filename! (
    echo No log file found for today. Run start-day.bat first.
    pause
    exit /b
)

:: Get current time with better formatting
for /f "tokens=1-4 delims=:. " %%a in ('echo %time%') do (
    set hour=%%a
    set min=%%b
    set sec=%%c
)

:: Remove leading space from hour if present
set hour=!hour: =!

:: Format timestamp
set timestamp=!hour!:!min!:!sec!

:: Add session header to log
echo. >> !filename!
echo --- >> !filename!
echo. >> !filename!
echo ## Session Ended: !timestamp! >> !filename!
echo. >> !filename!
echo ### What I worked on: >> !filename!
echo - >> !filename!

:: Open file for editing
echo Opening log to add session notes...
notepad !filename!

echo.
echo Session logged at !timestamp!
pause