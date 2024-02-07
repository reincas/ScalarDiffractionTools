@echo off
if exist build\ rmdir /s /q build 
if exist dist\ rmdir /s /q dist
for /d %%i in (*.egg-info) do rmdir /s /q %%i