@echo off
setlocal EnableExtensions

cd /d "%~dp0\.."

set "FAILED="

echo Cleaning generated artifacts in "%CD%"

call :remove_dir ".pytest_cache"
call :remove_dir ".mypy_cache"
call :remove_dir ".ruff_cache"
call :remove_dir ".test_output"
call :remove_dir "build"
call :remove_dir "dist"
call :remove_dir "htmlcov"

call :remove_glob_dirs ".\*.egg-info"
call :remove_glob_dirs ".\src\*.egg-info"
call :remove_glob_dirs_warn ".\pytest-cache-files-*"

call :remove_glob_files ".\.coverage"
call :remove_glob_files ".\.coverage.*"
call :remove_glob_files ".\coverage.xml"

call :remove_dir "__pycache__"
call :remove_named_dirs ".\src" "__pycache__"
call :remove_named_dirs ".\tests" "__pycache__"
call :remove_named_dirs ".\examples" "__pycache__"
call :remove_named_dirs ".\scripts" "__pycache__"

if defined FAILED (
    echo Clean finished with errors.
    exit /b 1
)

echo Clean finished successfully.
exit /b 0

:remove_dir
set "TARGET=%~1"
if exist "%TARGET%" (
    echo Removing directory "%TARGET%"
    rmdir /s /q "%TARGET%" >nul 2>&1
    if exist "%TARGET%" (
        echo Failed to remove directory "%TARGET%"
        set "FAILED=1"
    )
)
exit /b 0

:remove_glob_dirs
for /d %%D in (%~1) do (
    call :remove_dir "%%~fD"
)
exit /b 0

:remove_glob_dirs_warn
for /d %%D in (%~1) do (
    call :remove_dir_warn "%%~fD"
)
exit /b 0

:remove_glob_files
for %%F in (%~1) do (
    if exist "%%~fF" (
        echo Removing file "%%~fF"
        del /f /q "%%~fF" >nul 2>&1
        if exist "%%~fF" (
            echo Failed to remove file "%%~fF"
            set "FAILED=1"
        )
    )
)
exit /b 0

:remove_dir_warn
set "TARGET=%~1"
if exist "%TARGET%" (
    echo Removing directory "%TARGET%"
    rmdir /s /q "%TARGET%" >nul 2>&1
    if exist "%TARGET%" (
        echo Warning: could not remove directory "%TARGET%"
    )
)
exit /b 0

:remove_named_dirs
set "BASE_DIR=%~1"
set "DIR_NAME=%~2"
if not exist "%BASE_DIR%" exit /b 0
for /d /r "%BASE_DIR%" %%D in (%DIR_NAME%) do (
    call :remove_dir "%%~fD"
)
exit /b 0
