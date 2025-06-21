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

:: Create logs directory if it doesn't exist
if not exist logs mkdir logs

:: Check if file already exists
if exist !filename! (
    echo Today's log already exists. Opening existing file...
    notepad !filename!
    exit /b
)

:: Create new log file with header
echo # Work Log - !date! > !filename!
echo. >> !filename!
echo ## Day Started: !hour!:!min!:!sec! >> !filename!
echo. >> !filename!
echo ## Daily Goals >> !filename!
echo. >> !filename!

:: Open editor for user to add goals
echo Opening log file to add today's goals...
notepad !filename!

echo.
echo Daily log created: !filename! at !hour!:!min!:!sec!
echo Add your goals and save the file.
pause