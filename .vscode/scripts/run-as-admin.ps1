param (
    [string]$Command
)

Start-Process -FilePath "powershell.exe" -Verb RunAs -Wait -ArgumentList "-NoProfile", "-ExecutionPolicy", "ByPass", "-Command", $Command
