@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo 开始整理 C:\Users\hk\fire_flower 文件夹...
echo ==========================================

rem 定义目标文件夹
set "dest_scripts=scripts"
set "dest_data=data"
set "dest_results=results"
set "dest_docs=docs"
set "dest_figures=figures"
set "dest_logs=logs"
set "dest_temp=temp"
set "dest_attachments=attachments"

rem 创建目标文件夹（如果不存在）
echo 创建目标文件夹...
for %%D in ("%dest_scripts%" "%dest_data%" "%dest_results%" "%dest_docs%" "%dest_figures%" "%dest_logs%" "%dest_temp%" "%dest_attachments%") do (
    if not exist "%%~D" (
        echo   创建文件夹: %%~D
        mkdir "%%~D"
    ) else (
        echo   文件夹已存在: %%~D
    )
)

rem 移动文件到对应文件夹
echo.
echo 开始移动文件...

rem 移动 Python 脚本
echo   ^> 移动 Python 脚本...
move /Y "*.py" "%dest_scripts%" >nul 2>&1

rem 移动批处理文件
echo   ^> 移动批处理文件...
move /Y "*.bat" "%dest_scripts%" >nul 2>&1

rem 移动 Excel 结果文件
echo   ^> 移动 Excel 结果文件...
move /Y "*.xlsx" "%dest_results%" >nul 2>&1

rem 移动临时Excel文件 (以~$开头)
echo   ^> 移动临时Excel文件...
move /Y "~$*.xlsx" "%dest_temp%" >nul 2>&1

rem 移动图片文件
echo   ^> 移动图片文件...
move /Y "*.png" "%dest_figures%" >nul 2>&1

rem 移动日志文件
echo   ^> 移动日志文件...
move /Y "*.log" "%dest_logs%" >nul 2>&1

rem 移动 Markdown 文档
echo   ^> 移动 Markdown 文档...
move /Y "*.md" "%dest_docs%" >nul 2>&1

rem 移动 CSV 数据文件
echo   ^> 移动 CSV 数据文件...
move /Y "*.csv" "%dest_data%" >nul 2>&1

rem 移动特定命名的日志文件 (交互日志)
for %%F in (交互日志_*.md) do (
    echo   ^> 移动特定文件: %%F
    move /Y "%%F" "%dest_logs%" >nul 2>&1
)

echo.
echo ==========================================
echo 文件整理完成！
echo ==========================================
echo.
echo 整理后的文件夹结构:
echo   - %dest_scripts%      (Python 脚本, .bat 脚本)
echo   - %dest_data%         (CSV 数据文件)
echo   - %dest_results%      (Excel 结果文件)
echo   - %dest_docs%         (Markdown 文档)
echo   - %dest_figures%      (PNG 图片文件)
echo   - %dest_logs%         (LOG 日志文件, 交互日志)
echo   - %dest_temp%         (临时文件, 如 ~$*.xlsx)
echo   - %dest_attachments%  (其他未分类附件)
echo   - .conda, .git        (配置文件夹，保持原位)
echo   - 其他已知文件夹 (2025年度..., copt, qwencli) 保持原位
echo.
echo 请检查以上文件夹内容是否符合预期。
echo 按任意键退出...
pause >nul