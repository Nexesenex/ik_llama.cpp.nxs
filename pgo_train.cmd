@echo off
REM PGO (Profile-Guided Optimization) workflow for clang-cl builds.
REM
REM Prerequisites:
REM   1. LLVM 18+ with llvm-profdata in PATH
REM   2. A model file (GGUF) at the path you set below
REM   3. cmake + ninja in PATH
REM
REM Usage:
REM   Edit MODEL and PROMPT below, then:
REM       pgo_train.cmd
REM
REM The optimized binary ends up in build_pgo_opt\bin\llama-cli.exe

setlocal enabledelayedexpansion

REM === CONFIGURE THESE ===
set MODEL=C:\models\q4_k_m.gguf
set PROMPT=The quick brown fox jumps over the lazy dog
set N_TOKENS=200
set PGO_DIR=%~dp0pgo

echo === PGO: Step 1/4 - Instrumented build ===
cmake --preset cpu-avx2-pgo-instr
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cmake --build --preset cpu-avx2-pgo-instr
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo === PGO: Step 2/4 - Training run (TG, %N_TOKENS% tokens) ===
if not exist "%PGO_DIR%" mkdir "%PGO_DIR%"
set LLVM_PROFILE_FILE=%PGO_DIR%\train-%p.profraw
"%~dp0build_pgo_instr\bin\llama-cli.exe" -m "%MODEL%" -p "%PROMPT%" -n %N_TOKENS% -ngl 0
if %ERRORLEVEL% neq 0 (
    echo Training run finished (exit code %ERRORLEVEL%, may be normal for CLI mode)
)

echo === PGO: Step 3/4 - Merge profiles ===
if exist "%PGO_DIR%\train.profdata" del "%PGO_DIR%\train.profdata"
llvm-profdata merge -output="%PGO_DIR%\train.profdata" "%PGO_DIR%\*.profraw"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo Profiles merged to %PGO_DIR%\train.profdata

echo === PGO: Step 4/4 - Optimized build with profile data ===
cmake --preset cpu-avx2-pgo-opt
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cmake --build --preset cpu-avx2-pgo-opt
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo === PGO: Done ===
echo Optimized binary: %~dp0build_pgo_opt\bin\llama-cli.exe