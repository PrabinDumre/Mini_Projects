
import cv2

source = ("Prabin.jpg")
destination = ("Resized.jpg")

src = cv2.imread(source,cv2.IMREAD_UNCHANGED)
scale_percent = 40
new_width = int(src.shape[1] * scale_percent / 100)
new_height = int(src.shape[0] * scale_percent / 100)

output = cv2.resize(src,(new_width,new_height))

cv2.imwrite(destination,output)