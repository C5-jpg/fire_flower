@echo off
python analyze_bike_data_final.py > output.log 2>&1
type output.log
pause