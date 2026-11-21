@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ============================================================
rem  HarvestCast launcher
rem  1. locate Python
rem  2. fetch initial market data if missing (real quotes, no fake data)
rem  3. run a first forecast if none exists
rem  4. start local server, wait until the port is ready
rem  5. open the UI in the default browser
rem  Set HC_NO_BROWSER=1 to skip step 5 (headless / CI).
rem ============================================================

set "PORT=8848"
set "URL=http://127.0.0.1:%PORT%/"

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (
  where py >nul 2>nul && set "PY=py -3"
)
if not defined PY (
  echo [ERROR] Python 3.9+ not found in PATH.
  echo         Install it from https://www.python.org/downloads/ and run again.
  pause
  exit /b 1
)

rem ---- already listening? just open the UI ----
netstat -ano | findstr /C:"127.0.0.1:%PORT%" | findstr /C:"LISTENING" >nul 2>nul
if not errorlevel 1 (
  echo [INFO] Server already running on port %PORT%.
  goto open_ui
)

rem ---- step 1: initial market data ----
if exist "data\market\*.csv" goto pick_variety
echo [SETUP] No market data found. Fetching initial quotes for all varieties ...
echo         This may take a few minutes on first run.
%PY% -m src.cli init
if errorlevel 1 (
  echo.
  echo [ERROR] Could not fetch initial data.
  echo         Please check your network and run start.bat again.
  echo         Offline demo mode:  python -m src.cli init --allow-demo
  pause
  exit /b 1
)

rem ---- step 2: pick any variety that actually has data ----
:pick_variety
set "FIRST="
for %%f in ("data\market\*.csv") do if not defined FIRST set "FIRST=%%~nf"
if not defined FIRST set "FIRST=corn"

if exist "data\output\forecast_%FIRST%.json" goto start_server
echo [SETUP] Running a first forecast for %FIRST%, please wait ...
%PY% -m src.cli predict --variety %FIRST% --origins 6

:start_server
echo [START] Launching HarvestCast server on port %PORT% ...
rem Output goes to logs\server.log so this window can close without killing the server,
rem and so the child process never holds the caller's stdout handle.
if not exist "logs" md "logs"
start "HarvestCast Server" /min cmd /c "%PY% -u -m src.cli serve --port %PORT% > logs\server.log 2>&1"

rem ---- wait for the port, max ~30 seconds ----
set /a TRIES=0
:wait_port
set /a TRIES+=1
powershell -NoProfile -Command "try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',%PORT%);$c.Close();exit 0}catch{exit 1}" >nul 2>nul
if not errorlevel 1 goto open_ui
if %TRIES% GEQ 30 (
  echo [WARN] Server did not respond within 30s.
  echo        Check the minimized "HarvestCast Server" window for details.
  goto open_ui
)
rem sleep about one second without relying on timeout.exe
ping -n 2 127.0.0.1 >nul
goto wait_port

:open_ui
if defined HC_NO_BROWSER (
  echo [DONE] UI ready at %URL%  ^(browser disabled by HC_NO_BROWSER^)
  exit /b 0
)
echo [DONE] Opening %URL%
start "" "%URL%"
exit /b 0
