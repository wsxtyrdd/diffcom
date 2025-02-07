![Python >=3.8](https://img.shields.io/badge/Python->=3.7-yellow.svg)
![PyTorch >=1.9](https://img.shields.io/badge/PyTorch->=1.7-blue.svg)

# DiffCom: Channel Received Signal is a Natural Condition to Guide Diffusion Posterior Sampling [[pdf]](https://arxiv.org/abs/2406.07390)

Here is the implementation of the
paper "[DiffCom: Channel Received Signal is a Natural Condition to Guide Diffusion Posterior Sampling](https://semcomm.github.io/DiffCom/)".

Project website: [https://semcomm.github.io/DiffCom/](https://semcomm.github.io/DiffCom/)

## Overview of the DiffCom system architecture

<img src="imgs/Fig_framework.png"  style="width: 70%;" />

## RDP curves on [FFHQ](https://github.com/NVlabs/ffhq-dataset) testset

<img src="imgs/Fig_RDP.png"  style="width: 88%;" />

## Generalization to unseen wireless conditions

<img src="imgs/Fig_generalization.png"  style="width: 88%;" />

## Blind-DiffCom Achieves Pilot-Free Transmission

<img src="imgs/Fig_ce_free.png"  style="width: 88%;" />

## Requirements

Clone the repo and create a conda environment (we use PyTorch 1.9, CUDA 11.1).

TODO: check the dependencies

## Model Download

We provide 3 pre-trained ADJSCC models in [this link](https://drive.google.com/drive/folders/1N0EzzxCv1wh6JeFr0g8vkmB0Qj23ozZJ?usp=sharing), please download them and put them in the `_djscc/ckpt` folder.

The pre-trained Diffusion models are available at the following links, please download them and put them in the `model_zoo` folder.

| Model                                                       |                                               Download link                                                |
|-------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------:|
| 256x256_diffusion_uncond.pt(ILSVRC 2012 subset of ImageNet) | [download link](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) |
| ffhq_10m.pt                                                 |   [download link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing)    |

TODO: provide implementations of DiffCom based on \`SwinJSCC\` and \`NTSCC+\`.

## Inference Code

```bash
python main_diffcom.py --opt ./configs/diffcom.yaml
```

Please check the `diffcom.yaml` file for more details. 
The codebase now supports `diffcom`, `hifi_diffcom`, and `blind_diffcom`.

You can change the hyperparameters in the YAML file to run the corresponding method.

## Acknowledgement

The implementation is based on [DPS](https://github.com/DPS2022/diffusion-posterior-sampling), [PSLD](https://github.com/LituRout/PSLD).  
We thank the authors for sharing their code.

## Citation

If you find this code useful for your research, please cite our paper

```
@article{wang2024diffcom,
  title={DiffCom: Channel Received Signal is a Natural Condition to Guide Diffusion Posterior Sampling},
  author={Wang, Sixian and Dai, Jincheng and Tan, Kailin and Qin, Xiaoqi and Niu, Kai and Zhang, Ping},
  journal={arXiv preprint arXiv:2406.07390},
  year={2024}
}
```