import cv2

video_path = "for cv-project.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps if fps > 0 else "Unknown"

print(f"Resolution: {frame_width}x{frame_height}")
print(f"Frame Rate: {fps} FPS")
print(f"Duration: {duration} seconds")

cap.release()
