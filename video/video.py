import cv2
import os

path = "（刘嘉林）腹腔镜全腹膜外补片植入术（TEP）.mpg"
count = 0

if not os.path.exists(path.split('.')[0]):
    os.mkdir(path.split('.')[0])

cap = cv2.VideoCapture(path)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if count % 10 == 0:
            if count >= 1600 and count <= 8000:
            # if True:
                save_path = path.split('.')[0] + "/" + path + '_' + str(count) + ".png"
                cv2.imwrite(save_path, frame)
                cv2.destroyAllWindows()
        count += 1
    else:
        cap.release()

print(count)
cv2.destroyAllWindows()
