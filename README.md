# Reference-then-Supervision Framework for Infrared and Visible Image Fusion

## Abstract
Infrared and visible image fusion has drawn increasing attention of researchers in recent years, wherein the complementary information between two source images is extracted to synthesize a new fusion image with richer information. Deep neural network is the latest technique to tackle various image fusion problems. However, the natural absence of ground truth makes it very challenging to optimize deep learning models. While the current methods mainly consider to uses the two source images themselves or their visual features, to provide supervision for learning, which is easy to result in the imbalance between detail preservation and brightness distribution. To boost the performance of the fusion models, it is meaningful and necessary to establish a flexible framework that can combine the advantages of existing models to produce reliable supervision for model training. In this work, we propose a novel reference-then-supervision framework, which aims to fully leverage and exploit the available favorable reference information based on the performance of existing methods and then construct high-quality reliable supervision to assist in model building. For this purpose, we design an automatic filter to produce favorable reference and devise an adaptive enhancement method to construct reliable supervision, which helps to aggregate the advantages of various existing fusion methods for yielding visually pleasing results adapting to different complex scenarios. Extensive experiments on two commonly used datasets and our built challenging test set demonstrate that our framework can greatly improve the performance of existing fusion methods. Ablation study and empirical analysis also present the efficacy of our framework design. Furthermore, the applications on downstream pedestrian detection and object tracking tasks indicate the great potential of our framework. 

# Dataset
baiduYun:
Train: https://pan.baidu.com/s/1smUQDH9TMKZldUiJf_uZkA?pwd=769m : [769m] 
Test: https://pan.baidu.com/s/1VJK8h4mB5_hFLgJXAIu9ow?pwd=qxfy : [qxfy] 

## Environment Installation
pytorch:

python 3.6

torch 1.6.0

torchvision 0.7.0

scipy 1.2.1

numpy 1.18.2

scikit-image 0.16.2

tensorboard 2.11.2

tensorflow:

python 3.6

tensorflow-gpu 1.15.0 

scipy 1.2.1 

open-python 1.1.1

scikit-image  0.17.2

h5py 3.1.0

numpy 1.19.5

## Train
Baselines:
   - [DIDFuse](https://github.com/Zhaozixiang1228/IVIF-DIDFuse)
   - [RFN](https://github.com/hli1221/imagefusion-rfn-nest)
   - [PMGI](https://github.com/HaoZhang1018/PMGI_AAAI2020)
   - [SDNet](https://github.com/HaoZhang1018/SDNet)
   - [FusionGAN](https://github.com/jiayi-ma/FusionGAN)
   - [DDcGAN](https://github.com/hanna-xu/DDcGAN)
   - [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion)
   - [MGT](https://github.com/Vibashan/Image-Fusion-Transformer)
  
    Go to the folder for each method and run "bash train.sh"

## Test
Place the pre-trained model in the appropriate folder and run "bash test.sh"

### Results
[View PDF](image/IVF.pdf)

## Citation
If you use this data for your research, please cite our paper:

```
Li, Guihui and Shi, Zhensheng and Gu, Zhaorui and Zheng, Bing and Zheng, Haiyong, “Reference-then-Supervision Framework for Infrared and Visible Image Fusion”, pattern recognition, 2024. 
```

## Contact
Should you have any question, please contact guihuilee@163.com.

**Acknowledgment:** This code is based on the [DIDFuse](https://github.com/Zhaozixiang1228/IVIF-DIDFuse), [RFN](https://github.com/hli1221/imagefusion-rfn-nest), [PMGI](https://github.com/HaoZhang1018/PMGI_AAAI2020), [SDNet](https://github.com/HaoZhang1018/SDNet), [FusionGAN](https://github.com/jiayi-ma/FusionGAN), [DDcGAN](https://github.com/hanna-xu/DDcGAN), [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion), [MGT](https://github.com/Vibashan/Image-Fusion-Transformer).
