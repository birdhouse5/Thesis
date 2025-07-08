@echo off
setlocal enabledelayedexpansion

set tasksfile=tasks.md

if not exist !tasksfile! (
    echo No tasks.md found. Creating new file...
    echo # Tasks > !tasksfile!
    echo. >> !tasksfile!
    echo ## Active >> !tasksfile!
    echo. >> !tasksfile!
    echo ## Completed >> !tasksfile!
    echo. >> !tasksfile!
    echo ## On Hold >> !tasksfile!
    echo. >> !tasksfile!
)

echo Opening tasks.md...
echo.
echo TASK FORMAT GUIDE:
echo - [ ] [2025-06-24] Main task description
echo   - [ ] Subtask 1
echo   - [x] Subtask 2 (completed)
echo   - [ ] Subtask 3
echo.
echo Move completed tasks to the Completed section with completion date.
echo.

notepad !tasksfile!