@echo off
rem ===========================================================================
rem  CR-RL Training Launcher (Windows)
rem  Starts: league training (5 deck models + main) + dashboard web UI + browser
rem
rem  Usage:
rem    start_training.bat                                default run
rem    start_training.bat --config aggressive            named config (reward)
rem    start_training.bat --config aggressive --resume   resume from checkpoint
rem    start_training.bat --device cuda                  use GPU (needs cu130 torch)
rem    start_training.bat --n-envs 4                     parallel envs (batched GPU)
rem    start_training.bat --setup                        create/install .venv deps
rem    start_training.bat --setup-cuda                   create/install .venv + cu130 torch
rem    start_training.bat --selftest                     run selftest first
rem    start_training.bat --help
rem  Configs: standard / aggressive / defensive / elixir / fast
rem  NOTE: ASCII-only + CRLF line endings (cmd.exe codepage/parsing safety).
rem ===========================================================================
setlocal
title CR-RL Training Launcher

rem ---- paths ----
set "ROOT=%~dp0"
set "SRC=%ROOT%src\clasher_new"
set "VENV=%ROOT%.venv"

rem ---- defaults ----
set "CONFIG=standard"
set "CONFIG_NAME="
set "OUT_DIR=runs"
set "DEVICE=auto"
set "N_ENVS=1"
set "TOTAL_STEPS=20000"
set "STEPS_PER_EVAL=2000"
set "N_EVAL_GAMES=4"
set "MAX_EP_STEPS=600"
set "DECKS_PATH="
set "PORT=8090"
set "WITH_DASHBOARD=1"
set "ONLY_VS_MAIN=0"
set "KEEP_SNAPSHOT=0"
set "RESUME=0"
set "NO_REPLAYS=0"
set "DO_SETUP=0"
set "DO_SETUP_CUDA=0"
set "DO_SELFTEST=0"
set "SHOW_HELP=0"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"

rem ---- parse args (block-free, most robust form) ----
:parse
if "%~1"=="" goto parse_done
if /i "%~1"=="--help"           set "SHOW_HELP=1"
if /i "%~1"=="--config"         set "CONFIG=%~2"
if /i "%~1"=="--config"         shift
if /i "%~1"=="--config-name"    set "CONFIG_NAME=%~2"
if /i "%~1"=="--config-name"    shift
if /i "%~1"=="--out-dir"        set "OUT_DIR=%~2"
if /i "%~1"=="--out-dir"        shift
if /i "%~1"=="--device"         set "DEVICE=%~2"
if /i "%~1"=="--device"         shift
if /i "%~1"=="--n-envs"         set "N_ENVS=%~2"
if /i "%~1"=="--n-envs"         shift
if /i "%~1"=="--total-steps"    set "TOTAL_STEPS=%~2"
if /i "%~1"=="--total-steps"    shift
if /i "%~1"=="--steps-per-eval" set "STEPS_PER_EVAL=%~2"
if /i "%~1"=="--steps-per-eval" shift
if /i "%~1"=="--n-eval-games"   set "N_EVAL_GAMES=%~2"
if /i "%~1"=="--n-eval-games"   shift
if /i "%~1"=="--max-ep-steps"   set "MAX_EP_STEPS=%~2"
if /i "%~1"=="--max-ep-steps"   shift
if /i "%~1"=="--decks-path"     set "DECKS_PATH=%~2"
if /i "%~1"=="--decks-path"     shift
if /i "%~1"=="--port"           set "PORT=%~2"
if /i "%~1"=="--port"           shift
if /i "%~1"=="--no-dashboard"   set "WITH_DASHBOARD=0"
if /i "%~1"=="--only-vs-main"   set "ONLY_VS_MAIN=1"
if /i "%~1"=="--keep-snapshot"  set "KEEP_SNAPSHOT=1"
if /i "%~1"=="--resume"         set "RESUME=1"
if /i "%~1"=="--no-replays"     set "NO_REPLAYS=1"
if /i "%~1"=="--setup"          set "DO_SETUP=1"
if /i "%~1"=="--setup-cuda"     set "DO_SETUP=1"
if /i "%~1"=="--setup-cuda"     set "DO_SETUP_CUDA=1"
if /i "%~1"=="--selftest"       set "DO_SELFTEST=1"
shift
goto parse
:parse_done

if "%SHOW_HELP%"=="1" goto show_help
goto help_done

:show_help
echo Options:
echo   --config NAME         named config: standard/aggressive/defensive/elixir/fast
echo                         (each config = own reward weights + folder under OUT_DIR)
echo   --config-name NAME    override output folder name (default = config name)
echo   --out-dir DIR         run output root (default runs)
echo   --device DEV          cpu / cuda / auto (default auto; needs cu130 torch for cuda)
echo   --n-envs N            parallel envs (default 1; >1 uses batched GPU inference)
echo   --total-steps N       total training steps   (default 20000)
echo   --steps-per-eval N    evaluate every N steps (default 2000; also saves league replays)
echo   --n-eval-games N      games per pair         (default 4)
echo   --max-ep-steps N      max decision steps/game(default 600)
echo   --decks-path PATH     three-category decks   (default auto-detect)
echo   --resume              resume from run_state.json checkpoint
echo   --no-replays          do not save league replays
echo   --port N              dashboard port         (default 8090)
echo   --no-dashboard        do not open dashboard
echo   --only-vs-main        eval only main vs others
echo   --keep-snapshot       keep main_ckpt slot in league
echo   --setup               create .venv and install CPU deps
echo   --setup-cuda          create .venv and install CUDA 13 (cu130) torch
echo   --selftest            run selftest first
exit /b 0

:help_done

rem ---- cuda torch index ----
if "%DO_SETUP_CUDA%"=="1" set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"

rem ---- locate python ----
set "PY="
if exist "%VENV%\Scripts\python.exe" set "PY=%VENV%\Scripts\python.exe"
if not defined PY if exist "%VENV%\bin\python.exe" set "PY=%VENV%\bin\python.exe"
if not defined PY set "PY=python"

rem ---- optional: setup venv ----
if "%DO_SETUP%"=="1" goto do_setup
if not exist "%VENV%\Scripts\python.exe" if not exist "%VENV%\bin\python.exe" goto do_setup
goto setup_done

:do_setup
echo [setup] creating venv: %VENV%
python -m venv "%VENV%"
set "PY=%VENV%\Scripts\python.exe"
if not exist "%PY%" set "PY=%VENV%\bin\python.exe"
if not exist "%PY%" goto setup_failed
echo [setup] installing pip / torch(%TORCH_INDEX%) / gymnasium / stable-baselines3 / tqdm ...
"%PY%" -m pip install --upgrade pip
rem force-reinstall: if a CPU torch is already installed, plain install would skip
rem the CUDA index and keep the CPU build (cuda.is_available() stays False).
"%PY%" -m pip install --force-reinstall torch --index-url %TORCH_INDEX%
"%PY%" -m pip install "gymnasium" "stable-baselines3>=2.6" "tqdm"
goto setup_done

:setup_failed
echo [setup] FAILED to create venv. Install Python 3.10+ and add it to PATH.
pause
exit /b 1

:setup_done
echo.
echo [info] Python : %PY%
echo [info] Project: %ROOT%
echo [info] Config : %CONFIG%  (folder: %OUT_DIR%\%CONFIG%)

rem ---- optional: selftest ----
if "%DO_SELFTEST%"=="1" goto run_selftest
goto selftest_done

:run_selftest
echo [selftest] running full selftest...
"%PY%" "%ROOT%scripts\rl\selftest.py"
if errorlevel 1 goto selftest_failed
goto selftest_done

:selftest_failed
echo [selftest] FAILED. Training not started.
pause
exit /b 1

:selftest_done

rem ---- build training args (--opt=value form, avoids nested quotes) ----
set "TRAIN_ARGS=--mode run --config=%CONFIG% --out-dir=%OUT_DIR% --device=%DEVICE% --n-envs=%N_ENVS% --total-steps=%TOTAL_STEPS% --steps-per-eval=%STEPS_PER_EVAL% --n-eval-games=%N_EVAL_GAMES% --max-ep-steps=%MAX_EP_STEPS%"
if not "%CONFIG_NAME%"=="" set "TRAIN_ARGS=%TRAIN_ARGS% --config-name=%CONFIG_NAME%"
if not "%DECKS_PATH%"=="" set "TRAIN_ARGS=%TRAIN_ARGS% --decks-path=%DECKS_PATH%"
if "%ONLY_VS_MAIN%"=="1" set "TRAIN_ARGS=%TRAIN_ARGS% --only-vs-main"
if "%KEEP_SNAPSHOT%"=="1" set "TRAIN_ARGS=%TRAIN_ARGS% --keep-snapshot"
if "%RESUME%"=="1" set "TRAIN_ARGS=%TRAIN_ARGS% --resume"
if "%NO_REPLAYS%"=="1" set "TRAIN_ARGS=%TRAIN_ARGS% --no-replays"

echo.
echo ============================================================
echo   Starting CR-RL league training (5 deck models + main)
echo   Args: %TRAIN_ARGS%
echo ============================================================

rem ---- dashboard state path lives inside the config folder ----
set "STATE_DIR=%CONFIG%"
if not "%CONFIG_NAME%"=="" set "STATE_DIR=%CONFIG_NAME%"
set "STATE_JSON=%OUT_DIR%\%STATE_DIR%\league_state.json"

rem ---- training window ----
start "CR-RL Training" cmd /k ""%PY%" "%ROOT%scripts\rl\run_league.py" %TRAIN_ARGS%"

rem ---- dashboard window (optional) ----
if "%WITH_DASHBOARD%"=="1" goto start_dashboard
goto started

:start_dashboard
echo [info] Dashboard: http://127.0.0.1:%PORT%/
start "CR-RL Dashboard" cmd /k ""%PY%" "%ROOT%scripts\rl\dashboard.py" --state=%STATE_JSON% --port %PORT%"
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:%PORT%/"

:started
echo [info] Training started in a new window. Close that window to stop.
echo [info] Eval every %STEPS_PER_EVAL% steps, replays + dashboard refresh every 3s.
pause
endlocal
