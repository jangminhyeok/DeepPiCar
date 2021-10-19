import cv2

# 파이썬 파일 실행에 따른 코드
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        cv2.imshow("Cam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()
            break
