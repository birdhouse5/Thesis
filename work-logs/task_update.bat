@echo off
setlocal enabledelayedexpansion

:: Get current date
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do (
    set year=%%c
    set month=%%a
    set day=%%b
)

if !month! lss 10 set month=0!month:~-1!
if !day! lss 10 set day=0!day:~-1!

set filename=logs\!year!-!month!-!day!.md

:: Check if today's log exists
if not exist !filename! (
    echo No log file found for today.
    echo.
    echo Would you like to:
    echo 1. Start a new session
    echo 2. Open yesterday's log
    echo 3. Exit
    echo.
    set /p choice="Select (1-3): "
    
    if "!choice!"=="1" (
        start-session.bat
        exit /b
    )
    if "!choice!"=="2" (
        :: Try to find yesterday's log
        for /f "tokens=*" %%f in ('dir /b /o-d logs\*.md 2^>nul') do (
            set lastlog=%%f
            goto :found_last
        )
        :found_last
        if defined lastlog (
            notepad logs\!lastlog!
        ) else (
            echo No previous logs found.
            pause
        )
        exit /b
    )
    exit /b
)

:: Show quick status
echo.
echo Today's Status:
echo ---------------
set /a pending=0
set /a done=0
for /f "tokens=*" %%a in ('findstr /C:"[ ]" !filename! 2^>nul') do set /a pending+=1
for /f "tokens=*" %%a in ('findstr /C:"[x]" !filename! 2^>nul') do set /a done+=1
echo Completed: !done!
echo Pending: !pending!
echo.

echo Opening today's log...
notepad !filename!