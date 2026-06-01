param (
    [string]$WorkingDirectory,
    [string]$PythonInstallationPath
)

if ($WorkingDirectory) {
    Write-Host "Setting working directory..."
    Set-Location -Path $WorkingDirectory
}

if ($PythonInstallationPath) {
    Write-Host "Registering Python installation in PATH..."
    $Env:PATH = "$PythonInstallationPath;" + $Env:PATH
    $Env:PATH = "$PythonInstallationPath\Scripts;" + $Env:PATH
}

Write-Host "Current directory: $(Get-Location)"
Write-Host "Python installation path: $PythonInstallationPath"

Write-Host "Verifying Python installation..."
if (-not (Get-Command python -ErrorAction SilentlyContinue))
{
    Write-Host "Python is not installed or not available in the PATH."
    Write-Host "Please run this script in a terminal where a system Python installation has been registered."
    exit 1
}

Write-Host "Python installation found."
$pythonVersion = python --version
$pythonSourcePath = (Get-Command python).Source
Write-Host "Python version: $pythonVersion"
Write-Host "Python source path: $pythonSourcePath"

Write-Host "Installing UV..."
python -m pip install --upgrade pip uv

Write-Host "Upgrading lock file..."
python -m uv lock --upgrade

Write-Host "Installing project dependencies with UV..."
python -m uv sync --dev

Write-Host "Press Enter to exit..."
[void][System.Console]::ReadLine()
