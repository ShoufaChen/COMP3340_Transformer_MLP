import numpy as np
import os


data_root = "/home/chenshoufa/share_data/comp3340/flowers"

all_images = os.listdir(data_root)
all_images = [i for i in all_images if "jpg" in i]
all_images = sorted(all_images)

# there are 1360 images totally with 17 classes, each class has 80 images
# 1-80 belongs to class 1; 81-160 belongs to class 2...
train_images = []
val_images = []
for i in range(17):
    permutated_images = np.random.permutation(all_images[i * 80: (i + 1) * 80])
    train_images.append(permutated_images[:7 * len(permutated_images) // 8])
    val_images.append(permutated_images[7 * len(permutated_images) // 8:])

os.makedirs(os.path.join(data_root, "train"), exist_ok=True)
os.makedirs(os.path.join(data_root, "val"), exist_ok=True)

cnt = 0
for class_images in train_images:
    for img in class_images:
        class_i = "class_" + str(cnt)
        os.makedirs(f"{data_root}/train/{class_i}", exist_ok=True)
        os.system(f"cp {data_root}/{img} {data_root}/train/{class_i}")
    print("train", class_i)
    cnt += 1

cnt = 0
for class_images in val_images:
    for img in class_images:
        class_i = "class_" + str(cnt)
        os.makedirs(f"{data_root}/val/{class_i}", exist_ok=True)
        os.system(f"cp {data_root}/{img} {data_root}/val/{class_i}")
    print("val", class_i)
    cnt += 1
