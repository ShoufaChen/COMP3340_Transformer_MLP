_base_ = [
    '../_base_/models/vit_base_flowers.py',
    '../_base_/datasets/flowers_bs32.py',
    '../_base_/schedules/flowers_bs32.py',
    '../_base_/default_runtime.py'
]

#Now this Vit incoporates WDA
load_from = "/userhome/30/zhlyu/cogito/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth"

