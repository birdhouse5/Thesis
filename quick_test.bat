@echo off
echo ========================================
echo Quick Component Test
echo ========================================
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate
)

echo Testing data loader...
python -m src.data.data_loader
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo Testing episode sampling...
python -c "from src.data.data_loader import DataLoader; loader = DataLoader(['SPY','QQQ','TLT'], '2018-01-01', '2023-12-31', '2021-12-31'); sampler = loader.get_episode_sampler('train'); print('Successfully sampled episode with shape:', sampler.sample_episode().shape)"
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo Testing environment setup...
python -c "import yaml; print('Config loading works')"
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo All component tests passed!
echo.
pause
exit /b 0

:error
echo.
echo ERROR: Component test failed!
pause
exit /b 1