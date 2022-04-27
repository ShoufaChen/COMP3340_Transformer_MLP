import torch
import torch.nn as nn
import numpy as np
from icecream import ic
import os
import sys
import argparse
import time
import json
def generate_meta_data(model_config, checkpoint, output_dir):
    dir_path = output_dir + model_config.split('/')[-1][:-3] + '/' + checkpoint.split('/')[-1][:-4] + '/'
    ic(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(dir_path + 'embeddings.npy'):
        os.system(
            f"python tools/extract_backbone.py  {model_config} {checkpoint} --out-items all --metrics accuracy --out {dir_path}embeddings.npy")
    else:
        print("embeddings.npy already exists")

    if not os.path.exists(dir_path + 'eval_result.json'):
        os.system(
            f"python tools/test.py  {model_config} {checkpoint} --out-items all --metrics accuracy --out {dir_path}eval_result.json")
    else:
        print("eval_result.json already exists")

    path = dir_path + "embeddings.npy"
    output = np.load(path, allow_pickle=True)

    path = dir_path + "eval_result.json"
    f = open(path)
    data = json.load(f)
    embeddings = []
    m = torch.nn.AdaptiveAvgPool1d(1)
    for i in output:
        tensor_i = torch.tensor(i)
        if (len(tensor_i.shape) == 4):
            tensor_i = tensor_i.reshape(tensor_i.shape[0], tensor_i.shape[1], -1)
        y = m(tensor_i)
        real_i = y.numpy().reshape([-1, y.shape[1]])
        for j in real_i:
            embeddings.append(j)
    embeddings = np.array(embeddings)

    return embeddings, data, dir_path

def generate_badcase_pair(embeddings, data, output_dir):
    classes = ['daffodil', 'snowdrop', 'daisy', 'colts foot', 'dandelion', 'cowslip',
               'buttercup', 'wind flower', 'pansy', 'lilyValley', 'bluebell', 'crocys',
               'iris', 'tigerlily', 'tulip', 'fritillary', 'sunflower']
    gt_labels = []
    for i in classes:
        for j in range(10):
            gt_labels.append(i)

    badcase_list = []
    for i in range(len(data["pred_class"])):
        if data["pred_class"][i] != gt_labels[i]:
            badcase = {}
            badcase["image_idx"] = i
            badcase["gt_label"] = gt_labels[i]
            badcase["pred_label"] = data["pred_class"][i]
            badcase["gt_embedding"] = list(embeddings[i])
            badcase["class_scores"] = data['class_scores'][i]

            most_similar_idx = -1
            most_similar_score = 0
            for j in range(len(data["pred_class"])):
                if data["pred_class"][j] == data["pred_class"][i] and gt_labels[j]==data["pred_class"][i]:
                    from numpy import dot
                    from numpy.linalg import norm
                    current_score = dot(embeddings[i], embeddings[j])/(norm(embeddings[i])*norm(embeddings[j]))
                    if current_score > most_similar_score or most_similar_idx == -1:
                        most_similar_idx = j
                        most_similar_score = current_score
            if most_similar_idx == -1:
                continue

            badcase["most_similar_idx"] = most_similar_idx
            badcase["most_similar_score"] = most_similar_score
            badcase["most_similar_embedding"] = list(embeddings[most_similar_idx])
            badcase_list.append(badcase)

    path = output_dir + "badcases.jsonl"
    with open(path, 'w') as f:
        for badcase in badcase_list:
            f.write(str(badcase) + '\n')


def main():
    model_config = "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/configs/flowers/pertrain/resnet18_flowers_bs128_sda.py"
    checkpoint = "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/output/resnet18_sda_pertrain/epoch_100.pth"
    output_dir = "./output/PCA/"
    embeddings, data, dir_path = generate_meta_data(model_config, checkpoint, output_dir)
    generate_badcase_pair(embeddings, data, dir_path)


if __name__ == "__main__":
    main()