@ECHO OFF

SET ExecuteCommand=nvidia-smi.exe

SET ExecutePeriod=1

SETLOCAL EnableDelayedExpansion

:loop
  cls
  %ExecuteCommand%
  timeout /t %ExecutePeriod% > nul
goto loop