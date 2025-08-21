import cv2
import os
from datetime import datetime

# USB 웹캠 장치 번호 (일반적으로 0번)
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 한 프레임만 읽기
ret, frame = cam.read()
if not ret:
    print("캡쳐 실패")
    cam.release()
    exit()

# 현재 날짜시간 → 문자열 변환
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 파일 이름 생성
filename = f"capture_{timestamp}.jpg"
save_path = os.path.join(os.getcwd(), filename)

# 저장
cv2.imwrite(save_path, frame)
print(f"이미지 저장 완료: {save_path}")

cam.release()
