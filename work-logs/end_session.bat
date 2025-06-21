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
    echo No log file found for today. Run start-session.bat first.
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

:: Add session ended marker
echo. >> !filename!
echo ### Session Ended: !timestamp! >> !filename!
echo - >> !filename!

:: Open file for editing
echo.
echo Update your task checkboxes and add session notes:
echo   [ ] = pending
echo   [x] = done
echo   [~] = in progress
echo.
notepad !filename!

echo.
echo Session ended at !timestamp!
pause