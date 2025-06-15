@echo off
echo ========================================
echo VariBAD Trading Experiment
echo ========================================
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate
)

REM Test setup first
echo Testing setup...
python test_setup.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Setup test failed! Fix errors above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting Training
echo ========================================
echo.

python train_varibad.py

echo.
echo ========================================
echo Training Complete
echo ========================================
echo.

REM Analyze results
python -c "from src.utils.analyze_logs import analyze_latest_experiment; analyze_latest_experiment()"

echo.
pause