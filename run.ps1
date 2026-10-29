Set-Location $PSScriptRoot
Set-Location "$PSScriptRoot\app"
Write-Host "Open http://127.0.0.1:8000"
Write-Host "That page shows the stream. http://127.0.0.1:8000/video_feed is the raw feed."
& "$PSScriptRoot\plate_detection_env\Scripts\python.exe" -m uvicorn main:app --port 8000
