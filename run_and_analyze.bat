@echo off
echo ========================================
echo Running Temporal Split Experiment
echo ========================================
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate
)

REM Run the experiment
echo Running experiment...
python experiments/test_temporal_split.py

echo.
echo ========================================
echo Analyzing Results
echo ========================================
echo.

REM Analyze the results
python -c "from src.utils.analyze_logs import analyze_latest_experiment; analyze_latest_experiment()"

echo.
echo Check the results folder for detailed logs!
echo.
pause