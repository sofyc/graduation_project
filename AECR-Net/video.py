import cv2

path = "video/（刘嘉林）腹腔镜全腹膜外补片植入术（TEP）.mpg"
index = 0
count = 0

cap = cv2.VideoCapture(path)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # if count % 1000 == 0:
        if 0:
            save_path = "video_image/" + path + '_' + str(index) + ".png"
            cv2.imwrite(save_path, frame)
            cv2.destroyAllWindows()
            index += 1
        count += 1
    else:
        cap.release()

print(count)
cv2.destroyAllWindows()
