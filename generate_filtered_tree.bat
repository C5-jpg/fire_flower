@echo off
setlocal EnableDelayedExpansion

REM 设置根目录和输出文件
set "ROOT_DIR=C:\Users\hk\fire_flower"
set "OUTPUT_FILE=%ROOT_DIR%\filtered_directory_tree.txt"

REM 清空或创建输出文件
echo Folder PATH listing > "%OUTPUT_FILE%"
echo Volume serial number is >> "%OUTPUT_FILE%"
dir /-C "%ROOT_DIR%" | findstr "Volume Serial Number" >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

REM 递归遍历目录并过滤
call :tree_level "%ROOT_DIR%" "" >> "%OUTPUT_FILE%"

goto :eof

:tree_level
set "current_dir=%~1"
set "prefix=%~2"

REM 列出当前目录下的文件和文件夹
for /F "tokens=*" %%F in ('dir /A /B "%current_dir%" 2^>NUL') do (
    set "item_full_path=%current_dir%\%%F"
    set "skip_item="

    REM 检查是否需要跳过特定文件夹
    if /I "%%F"==".conda" set "skip_item=1"
    if /I "%%F"==".git" set "skip_item=1"
    if /I "%%F"=="__pycache__" set "skip_item=1"

    if not defined skip_item (
        if exist "!item_full_path!\*" (
            REM 是目录
            echo !prefix!%%F
            call :tree_level "!item_full_path!" "!prefix!   "
        ) else (
            REM 是文件
            echo !prefix!   %%F
        )
    )
)
goto :eof