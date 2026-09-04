@echo off
rem ===========================================================================
rem  CR-RL Unified Launcher (Windows)
rem  One entry for ALL training modes + dashboard + browser:
rem    solo  : self-play (fixed deck mirror + frozen copy)      [default]
rem    run   : league (5 deck models + main PPO)
rem    flow  : full pairwise archetype league (6 models)
rem
rem  Usage:
rem    start_rl.bat                                  -> solo economy, parallel eval 16
rem    start_rl.bat --mode run --config aggressive   -> league training
rem    start_rl.bat --mode flow                      -> flow pairwise league
rem    start_rl.bat --mode solo --resume             -> resume solo from checkpoint
rem    start_rl.bat --eval-workers 0                 -> serial eval (parallel off)
rem    start_rl.bat --no-dashboard                   -> train only, no web UI
rem    start_rl.bat --setup / --setup-cuda           -> create/install .venv deps
rem    start_rl.bat --selftest                       -> run selftest first
rem    start_rl.bat --help
rem
rem  NOTE: ASCII-only + CRLF line endings (cmd.exe codepage/parsing safety).
rem ===========================================================================
setlocal
title CR-RL Launcher

rem ---- paths ----
set "ROOT=%~dp0"
set "SRC=%ROOT%src\clasher_new"
set "VENV=%ROOT%.venv"

rem ---- defaults ----
set "MODE=solo"
set "CONFIG=economy"
set "CONFIG_NAME="
set "OUT_DIR=runs"
set "DEVICE=auto"
set "N_ENVS=1"
set "TOTAL_STEPS=20000"
set "STEPS_PER_EVAL=2000"
set "N_EVAL_GAMES=16"
set "MAX_EP_STEPS=360"
set "EVAL_WORKERS=16"
set "SOLO_COPY_EVERY=2000"
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
set "DRY_RUN=0"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"

rem ---- parse args (block-free, most robust form) ----
:parse
if "%~1"=="" goto parse_done
if /i "%~1"=="--help"           set "SHOW_HELP=1"
if /i "%~1"=="--mode"           set "MODE=%~2"
if /i "%~1"=="--mode"           shift
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
if /i "%~1"=="--eval-workers"   set "EVAL_WORKERS=%~2"
if /i "%~1"=="--eval-workers"   shift
if /i "%~1"=="--solo-copy-every" set "SOLO_COPY_EVERY=%~2"
if /i "%~1"=="--solo-copy-every" shift
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
if /i "%~1"=="--dry-run"        set "DRY_RUN=1"
shift
goto parse
:parse_done

if "%SHOW_HELP%"=="1" goto show_help
goto help_done

:show_help
echo Modes:
echo   --mode MODE      solo (self-play, default) / run (league) / flow (pairwise league)
echo Config:
echo   --config NAME          standard/aggressive/defensive/elixir/economy/fast
echo   --config-name NAME     override output folder name (default = config name)
echo   --out-dir DIR          run output root (default runs)
echo   --device DEV           cpu / cuda / auto (default auto; needs cu130 torch for cuda)
echo   --n-envs N             parallel envs (default 1; >1 uses batched GPU inference)
echo Training:
echo   --total-steps N        total training steps      (default 20000)
echo   --steps-per-eval N     evaluate every N steps    (default 2000; saves replays)
echo   --n-eval-games N       games per eval            (default 16)
echo   --max-ep-steps N       max decision steps/game   (default 360)
echo   --eval-workers N       parallel eval processes   (default 16; 0 = serial)
echo   --solo-copy-every N    solo: frozen copy sync    (default 2000)
echo   --resume               resume from run_state.json checkpoint
echo   --no-replays           do not save eval replays
echo   --only-vs-main         league: eval main vs others only
echo   --keep-snapshot        league: keep main_ckpt slot
echo Dashboard:
echo   --port N               dashboard port            (default 8090)
echo   --no-dashboard         do not open dashboard/browser
echo Setup:
echo   --setup                create .venv and install CPU deps
echo   --setup-cuda           create .venv and install CUDA 13 (cu130) torch
echo   --selftest             run selftest first
echo   --dry-run              print commands only, do not start anything
exit /b 0

:help_done

rem ---- cuda torch index ----
if "%DO_SETUP_CUDA%"=="1" set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"

rem ---- validate mode ----
if /i not "%MODE%"=="solo" if /i not "%MODE%"=="run" if /i not "%MODE%"=="flow" (
  echo [error] unknown --mode "%MODE%"  [use solo/run/flow]
  pause
  exit /b 1
)

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
echo [info] Mode   : %MODE%
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

rem ---- build common training args (--opt=value form, avoids nested quotes) ----
set "TRAIN_ARGS=--mode=%MODE% --config=%CONFIG% --out-dir=%OUT_DIR% --device=%DEVICE% --n-envs=%N_ENVS% --total-steps=%TOTAL_STEPS% --steps-per-eval=%STEPS_PER_EVAL% --n-eval-games=%N_EVAL_GAMES% --max-ep-steps=%MAX_EP_STEPS%"
if not "%CONFIG_NAME%"=="" set "TRAIN_ARGS=%TRAIN_ARGS% --config-name=%CONFIG_NAME%"
if not "%DECKS_PATH%"=="" set "TRAIN_ARGS=%TRAIN_ARGS% --decks-path=%DECKS_PATH%"
if "%ONLY_VS_MAIN%"=="1" set "TRAIN_ARGS=%TRAIN_ARGS% --only-vs-main"
if "%KEEP_SNAPSHOT%"=="1" set "TRAIN_ARGS=%TRAIN_ARGS% --keep-snapshot"
if "%RESUME%"=="1" set "TRAIN_ARGS=%TRAIN_ARGS% --resume"
if "%NO_REPLAYS%"=="1" set "TRAIN_ARGS=%TRAIN_ARGS% --no-replays"

rem ---- mode-specific args ----
set "STATE_DIR=%CONFIG%"
if not "%CONFIG_NAME%"=="" set "STATE_DIR=%CONFIG_NAME%"
set "DASH_ARGS="
if /i "%MODE%"=="solo" (
  set "TRAIN_ARGS=%TRAIN_ARGS% --eval-workers=%EVAL_WORKERS% --solo-copy-every=%SOLO_COPY_EVERY%"
  set "DASH_ARGS=--solo=%OUT_DIR%\%STATE_DIR%"
) else if /i "%MODE%"=="run" (
  set "DASH_ARGS=--state=%OUT_DIR%\%STATE_DIR%\league_state.json"
) else (
  set "DASH_ARGS=--sweep=%OUT_DIR%\%STATE_DIR%"
)

echo.
echo ============================================================
echo   Starting CR-RL training (mode=%MODE%, config=%CONFIG%)
echo   Args: %TRAIN_ARGS%
echo ============================================================

rem ---- dry-run: print commands only, never start ----
if "%DRY_RUN%"=="1" goto dry_run

rem ---- training window ----
start "CR-RL Training [%MODE%]" cmd /k ""%PY%" "%ROOT%scripts\rl\run_league.py" %TRAIN_ARGS%"

rem ---- dashboard window (optional) ----
if "%WITH_DASHBOARD%"=="1" goto start_dashboard
goto started

:start_dashboard
echo [info] Dashboard: http://127.0.0.1:%PORT%/
start "CR-RL Dashboard [%MODE%]" cmd /k ""%PY%" "%ROOT%scripts\rl\dashboard.py" %DASH_ARGS% --port %PORT%"
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:%PORT%/"
goto started

:dry_run
echo.
echo [dry-run] would run:
echo   "%PY%" "%ROOT%scripts\rl\run_league.py" %TRAIN_ARGS%
if "%WITH_DASHBOARD%"=="1" (
  echo   "%PY%" "%ROOT%scripts\rl\dashboard.py" %DASH_ARGS% --port %PORT%
  echo   browser: http://127.0.0.1:%PORT%/
)
echo [dry-run] nothing started.
pause
exit /b 0

:started
echo [info] Training started in a new window. Close that window to stop.
echo [info] Eval every %STEPS_PER_EVAL% steps; dashboard auto-refreshes.
pause
endlocal
