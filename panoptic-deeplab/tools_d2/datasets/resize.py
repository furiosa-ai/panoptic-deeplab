import os
import glob
import cv2

for img_file in glob.glob("./cityscapes_256_512/*/*/*.png"):
    
    img = cv2.imread(img_file)
    img_resized = cv2.resize(img, dsize=(512,256), interpolation=cv2.INTER_AREA)

    #cv2.imshow("src",img)
    #cv2.imshow("ds", img_resized)
    #cv2.waitKey(0) press any key to move on to next img
    #cv2.destroyAllWindows()

    cv2.imwrite(img_file, img_resized)