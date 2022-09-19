import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(
            frame,
            1  # 1：水平镜像，-1：垂直镜像
        )
    cv2.imshow('frame', frame)
    # 这一步必须有，否则图像无法显示
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 当一切完成时，释放捕获
cap.release()
cv2.destroyAllWindows()
