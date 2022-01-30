import os

data_root = "/home/chenshoufa/share_data/comp3340/flowers"
f = open(os.path.join(data_root, 'meta/train.txt'), 'w')

classes = os.listdir(os.path.join(data_root, 'train'))
classes = sorted(classes)
for c in classes:
    if '.ipynb' not in c:
        print(c)
        class_id = c.split("_")[-1]
        cur_files = os.listdir(os.path.join(data_root, 'train', c))
        for cur_file in cur_files:
            line = c + '/' + cur_file + ' ' + class_id + '\n'
            f.write(line)
f.close()

f = open(os.path.join(data_root, 'meta/val.txt'), 'w')
classes = os.listdir(os.path.join(data_root, 'val'))
classes = sorted(classes)
for c in classes:
    if '.ipynb' not in c:
        print(c)
        class_id = c.split("_")[-1]
        cur_files = os.listdir(os.path.join(data_root, 'val', c))
        for cur_file in cur_files:
            line = c + '/' + cur_file + ' ' + class_id + '\n'
            f.write(line)
f.close()
