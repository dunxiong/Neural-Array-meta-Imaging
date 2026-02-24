# Neural Array Meta-Imaging

This repository contains the official implementation of the paper:

> **Neural Array Meta-Imaging**  
> Xiong Dun, Jian Zhang, Fansheng Chen, Zhanyi Zhang, Xuquan Wang, Yujie Xing, Siyu Dong, Zeying Fan, Yuzhi Shi, Gordon Wetzstein, Zhanshan Wang, Xinbin Cheng
> **elight, 2025**  
---

**Neural Array Meta-Imaging (NAMI)** introduces a new computational imaging paradigm that integrates a **neural array mapping metalens** with **end-to-end learning-based reconstruction**.  
The system enables large-aperture, wide-FOV, and high-quality imaging through joint optical and algorithm co-design.

## Function
This repository provides the design codes, simulation models, and reconstruction algorithms for the **Neural Array Meta-Imaging (NAMI)** system.  
The code integrates a *metalens unit design*, *neural array optical mapping*, and a *multi-scale feature-aware Wiener deconvolution deep fusion network (MFWDFNet)* for high-fidelity image reconstruction.

## Structure
The workflow includes four main parts:

| Folder | Module | Description |
|:-------|:--------|:-------------|
| `0Metalens-unit-design` | **Lens Unit Design** | Design and simulation of individual metalens units used in the neural array meta-system. |
| `1Neural-array-design` | **Array Optimization** | Construct and optimize the neural array layout and optical mapping using the designed lens units. |
| `2MFWDFNet-finetuning` | **Algorithm Development** | Implementation and fine-tuning of the proposed MFWDFNet reconstruction network, compared with existing methods. |
| `3Experiment` | **Experimental Validation** | Real captured data reconstruction and comparison with simulation results. |


## Usage
**Metalens Unit Design**

cd 0Metalens-unit-design

conda env create -f environment.yml

sh run_train.sh

**Neural array design**

cd 1Neural-array-design

sh run_train_all.sh

**MFWDFNet finetuning**

cd 2MFWDFNet-finetuning

conda env create -f pytorch_2_2.yaml

sh run_train_MFWDFNet_CPSF.sh

**Experiment**

cd 3Experiment

python inference.py

## Citation
If you find our work useful in your research, please cite:
```
@article{dun2025neuralarray,
  title={Neural Array Meta-Imaging},
  author={Dun, Xiong and Zhang, Jian and Chen, Fansheng and Zhang, Zhanyi and Wang, Xuquan and Xing, Yujie and Dong, Siyu and Fan, Zeying and Shi, Yuzhi and Wetzstein, Gordon and Wang, Zhanshan and Cheng, Xinbin},
  year={2025},
  journal={To be announced}
}

```

## License
Our code is licensed ****
