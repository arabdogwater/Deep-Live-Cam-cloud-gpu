@echo off
echo.
echo =====================================================
echo   Deep-Live-Cam -- Push Webcam to GPU
echo =====================================================
echo.

echo Detecting video devices...
echo.
ffmpeg -list_devices true -f dshow -i dummy 2>&1 | findstr /i "video"
echo.

set /p CAM="Enter your webcam name exactly as shown above (e.g. Camo): "

echo.
set /p RTSP_URL="Enter the RTSP push URL from vast.ai (e.g. rtsp://1.2.3.4:48207/webcam): "

echo.
echo Streaming "%CAM%" to %RTSP_URL%...
echo Press Ctrl+C to stop.
echo.

ffmpeg -f dshow -i video="%CAM%" -vcodec libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -b:v 2M -maxrate 2M -bufsize 4M -f rtsp %RTSP_URL%

pause
