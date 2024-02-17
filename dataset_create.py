from PIL import Image
import os, glob
import numpy as np
from PIL import ImageFile
# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

# classes = ["Anmitsu", "Castella", "Daifuku", "Dango", "Dorayaki", "Imagawayaki", "Manju", "monaka", "Taiyaki", "Warabimochi", "Youkan", "Zenzai"]
classes = [ "Castella","Daifuku", "Dango", "Dorayaki",  "Imagawayaki", "Taiyaki", "Youkan", "monaka"]
num_classes = len(classes)
image_size = 128
num_testdata = 20

X_train = []
X_test  = []
y_train = []
y_test  = []

for index, classlabel in enumerate(classes):
    photos_dir = "./drive/MyDrive/prodacts_ver_jpg/" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))

        data = np.asarray(image)

        if i < num_testdata:
            X_test.append(data)
            y_test.append(index)


        else:

            # angleに代入される値
            # -10
            #  -5
            # 0
            # 5
            # 10
            # 画像を5度ずつ回転
            for angle in range(-10, 10, 5):

                img_r = image.rotate(angle)
                data = np.array(img_r)
                X_train.append(data)
                y_train.append(index) 




X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)



xy = (X_train, X_test, y_train, y_test)

# X_train, X_test, y_train, y_test を個別に保存
np.save("./X_train.npy", X_train)
np.save("./X_test.npy", X_test)
np.save("./y_train.npy", y_train)
np.save("./y_test.npy", y_test)