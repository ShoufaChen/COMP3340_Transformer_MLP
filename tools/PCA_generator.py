import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import json
import os
from icecream import ic

def generate_meta_data(model_config, checkpoint, output_dir):
    dir_path = output_dir+model_config.split('/')[-1][:-3]+'/'+checkpoint.split('/')[-1][:-4]+'/'
    ic(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(dir_path+'embeddings.npy'):
        os.system(f"python tools/extract_backbone.py  {model_config} {checkpoint} --out-items all --metrics accuracy --out {dir_path}embeddings.npy")
    else:
        print("embeddings.npy already exists")

    if not os.path.exists(dir_path+'eval_result.json'):
        os.system(f"python tools/test.py  {model_config} {checkpoint} --out-items all --metrics accuracy --out {dir_path}eval_result.json")
    else:
        print("eval_result.json already exists")
        
    path = dir_path+"embeddings.npy"
    output = np.load(path, allow_pickle=True)

    path = dir_path+"eval_result.json"
    f = open(path)
    data = json.load(f)
    embeddings = []
    m = torch.nn.AdaptiveAvgPool1d(1)
    for i in output:
        tensor_i = torch.tensor(i)
        if (len(tensor_i.shape) == 4):
            tensor_i = tensor_i.reshape(tensor_i.shape[0],tensor_i.shape[1], -1)
        y = m(tensor_i)
        real_i = y.numpy().reshape([-1,y.shape[1]])
        for j in real_i:
            embeddings.append(j)
    embeddings = np.array(embeddings)

    return embeddings, data, dir_path

pca = None
def generate_PCA_graph(embeddings, data, output_dir):
    classes = ['daffodil', 'snowdrop', 'daisy', 'colts foot', 'dandelion', 'cowslip',
               'buttercup', 'wind flower', 'pansy', 'lilyValley', 'bluebell', 'crocys',
               'iris', 'tigerlily', 'tulip', 'fritillary', 'sunflower']
    gt_labels = []
    for i in classes:
        for j in range(10):
            gt_labels.append(i)

    is_true = []
    for i in range(len(data["pred_class"])):
        if data["pred_class"][i] == gt_labels[i]:
            is_true.append(True)
        else:
            is_true.append(False)

    #scaler = StandardScaler()
    #scaler.fit(embeddings)
    #scaled = scaler.transform(embeddings)

    global pca
    if (pca is None):
        pca = PCA().fit(embeddings)

    pc = pca.transform(embeddings)
    pc1 = pc[:, 0]
    pc2 = pc[:, 1]

    data_df = pd.DataFrame.from_dict({"pc1": pc1, "pc2": pc2, "label": gt_labels, "is_true": is_true})
    sns.set(style='darkgrid', )
    ax = sns.scatterplot(x="pc1", y="pc2", data=data_df, hue='label', size="is_true")
    ax.set_title("PCA")
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])

    ax.legend(loc=2, bbox_to_anchor=(1.1, 1.05), borderaxespad=0.)

    fig = ax.get_figure()
    fig.savefig(output_dir+'PCA.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir+'PCA.png', dpi=300, bbox_inches='tight')
    plt.close(fig)



def main():
    model_configs_list = [
        "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/configs/flowers/pertrain/resnet18_flowers_bs128_sda.py",
        "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/configs/flowers/pertrain/resnet18_flowers_bs128_sda.py",
        "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/configs/flowers/pertrain/resnet18_flowers_bs128_sda.py",
        "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/configs/flowers/pertrain/resnet18_flowers_bs128_sda.py",
        "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/configs/flowers/pertrain/resnet18_flowers_bs128_sda.py",
        "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/configs/flowers/pertrain/resnet18_flowers_bs128_sda.py",
    ]
    checkpoint_list = [
        "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/output/resnet18_sda_pertrain/epoch_100.pth",
        "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/output/resnet18_sda_pertrain/epoch_80.pth",
        "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/output/resnet18_sda_pertrain/epoch_60.pth",
        "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/output/resnet18_sda_pertrain/epoch_40.pth",
        "/userhome/30/zhlyu/COMP3340_new/COMP3340_Transformer_MLP/output/resnet18_sda_pertrain/epoch_20.pth",
        "/userhome/30/zhlyu/cogito/resnet18_8xb32_in1k_20210831-fbbb1da6.pth",
    ]
    output_dir = "./output/PCA/"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in range(len(model_configs_list)):
        model_config = model_configs_list[i]
        checkpoint = checkpoint_list[i]
        #ic(model_config, checkpoint, output_dir)
        embeddings, data, dir_path = generate_meta_data(model_config, checkpoint, output_dir)
        generate_PCA_graph(embeddings, data, dir_path)

if __name__ == "__main__":
    main()