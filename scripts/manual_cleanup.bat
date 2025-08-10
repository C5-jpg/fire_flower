@echo off
echo Starting manual file organization...

rem Move temporary Excel files
echo Moving temporary Excel files...
move "~$*.xlsx" "temp" >nul 2>&1

rem Move the specific log file
echo Moving specific log file...
move "交互日志_20250810.md" "logs" >nul 2>&1

rem Move the oddly named file (likely a remnant from a previous move)
echo Cleaning up odd file names...
if exist "移动特定文件" (
    del "移动特定文件"
)

echo Manual organization complete.
pause