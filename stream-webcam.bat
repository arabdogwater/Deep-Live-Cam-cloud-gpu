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

set /p CAM="Enter your webcam name exactly as shown above (e.g. HD Webcam): "

echo.
echo Streaming "%CAM%" to GPU...
echo Press Ctrl+C to stop.
echo.
echo Once running, open: http://77.48.24.250:48253/vnc.html
echo.

ffmpeg -f dshow -i video="%CAM%" -vcodec libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -b:v 2M -maxrate 2M -bufsize 4M -f rtsp rtsp://77.48.24.250:48207/webcam

pause
