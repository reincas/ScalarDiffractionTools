@echo off
call clean.bat
python -m build
pip uninstall -y ScalarDiffractionTools
pip install .
rem python -m twine upload dist/*
