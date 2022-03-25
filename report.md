* Check the size of model

```bash
python tools/analysis_tools/get_flops.py configs/vision_transformer/vit_base_patch16_224_flowers.py
```

* do test inference

```bash
python tools/test.py  configs/vision_transformer/vit_base_patch16_224_flowers.py  work_dir/vit_base_patch16_224_flowers_pertrain/epoch_100.pth --out-items all --metrics accuracy --out 2.json
```

* Check the output data

```bash
python tools/visualizations/vis_cam.py  data/flowers/val/class_2/image_0202.jpg  configs/vision_transformer/vit_base_patch16_224_flowers.py work_dir/vit_base_patch16_224_flowers_pertrain/epoch_100.pth --vit-like --save-path ./1.jpg

```

```json

[0.9340776801109314, 0.00552458455786109, 0.003630232298746705, 0.0036970514338463545, 0.0047983271069824696, 0.0035009977873414755, 0.004527595825493336, 0.0029648838099092245, 0.0045443386770784855, 0.00393935339525342, 0.004238779656589031, 0.004138697404414415, 0.0038005071692168713, 0.004411126021295786, 0.0037998403422534466, 0.00473420973867178, 0.003671826794743538] 
[0.47576189041137695, 0.0040862709283828735, 0.003532903967425227, 0.005965675227344036, 0.005556718911975622, 0.013261193409562111, 0.004833100363612175, 0.019689539447426796, 0.017422014847397804, 0.01648862473666668, 0.005982196889817715, 0.08169852197170258, 0.3195144832134247, 0.008719236589968204, 0.006533948238939047, 0.005563376005738974, 0.005390321835875511]

```



* What we need to do
  * save the model performence for each testset
  * list a **TN, FP ** table between different classes
  * use **CAM** to analyze where error comes from

Our Analyze code:

```bash
https://colab.research.google.com/drive/1uyJiMzOyTCZ4oDhN6b_zaKO2bnqVwlDv?usp=sharing
```

