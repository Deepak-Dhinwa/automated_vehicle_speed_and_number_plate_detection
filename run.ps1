Set-Location $PSScriptRoot
Set-Location "$PSScriptRoot\app"
& "$PSScriptRoot\plate_detection_env\Scripts\python.exe" -m uvicorn main:app --port 8000
