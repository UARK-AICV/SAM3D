<h1 align="center">SAM3D: Segment Anything Model in Volumetric Medical Images </h1>
<p align="center">
  <p align="center">
    <a href="https://tanbuinhat.github.io/"><strong>Nhat-Tan Bui</strong></a>
    ·
    <a href="https://dblp.org/pid/253/9950.html"><strong>Dinh-Hieu Hoang</strong></a>
    ·
    <a href="https://www.fit.hcmus.edu.vn/~tmtriet/"><strong>Minh-Triet Tran</strong></a>
    ·
    <a href="https://vision.csee.wvu.edu/~doretto/"><strong>Gianfranco Doretto</strong></a>
    .
    <a href="https://community.wvu.edu/~daadjeroh/"><strong>Donald Adjeroh</strong></a>
    .
    <a href="https://directory.hsc.wvu.edu/Profile/60996"><strong>Brijesh Patel</strong></a>
    .
    <a href="https://uamshealth.com/provider/arabinda-k-choudhary/"><strong>Arabinda Choudhary</strong></a>
    .
    <a href="https://engineering.uark.edu/directory/index/uid/magda/name/Magda-El-Shenawee/"><strong>Magda El-Shenawee</strong></a>
    .
    <a href="https://www.nganle.net/"><strong>Ngan Le</strong></a>
  </p>

  <h4 align="center"><a href="https://arxiv.org/abs/2309.03493">arXiv</a></h4>
  <div align="center"></div>
</p>

## Introduction
<image src="images/architecture.png">
  
Image segmentation remains a pivotal component in medical image analysis, aiding in the extraction of critical information for precise diagnostic practices. With the advent of deep learning, automated image segmentation methods have risen to prominence, showcasing exceptional proficiency in processing medical imagery. Motivated by the Segment Anything Model (SAM)—a foundational model renowned for its remarkable precision and robust generalization capabilities in segmenting 2D natural images—we introduce SAM3D, an innovative adaptation tailored for 3D volumetric medical image analysis. Unlike current SAM-based methods that segment volumetric data by converting the volume into separate 2D slices for individual analysis, our SAM3D model processes the entire 3D volume image in a unified approach. Extensive experiments are conducted on multiple medical image datasets to demonstrate that our network attains competitive results compared with other state-of-the-art methods in 3D medical segmentation tasks while being significantly efficient in terms of parameters.
<p align="center">
<image src="images/decoder.png">
</p>

## Datasets and Trained Models
<ul>
  <li>To validate the effectiveness of our model, we conduct the experiments benchmark in four datasets: Synapse, ACDC, BRaTs, and Decathlon-Lung. We follow the same dataset preprocessing as in <a href="https://github.com/Amshaker/unetr_plus_plus">UNETR++</a> and <a href="https://github.com/282857341/nnFormer">nnFormer</a>. Please refer to their repositories for more details about organizing the dataset folders. Alternatively, you can download the preprocessed dataset at <a href="https://drive.google.com/drive/folders/1N8FAxEH0QExkqQbPT2oy2DrzUfIaRDMx?usp=drive_link">Google Drive</a>.</li>
  <li>The pre-trained SAM model can be downloaded at its <a href="https://github.com/facebookresearch/segment-anything">original repository</a>. Put the checkpoint in the ./checkpoints folder.</li>
  <li>The Synapse weight can be downloaded at <a href="https://drive.google.com/file/d/1jxWSlK1Zy_gBY_XO3xh6ydaLthqDh5Tm/view?usp=sharing">Google Drive</a>.</li>
  <li>The ADCD weight can be downloaded at <a href="https://drive.google.com/file/d/1a4fWzwEC9jKBKcZ_kj9wsrtoL8Ha1qpx/view?usp=drive_link">Google Drive</a>.</li>
  <li>The BraTS weight can be downloaded at <a href="https://drive.google.com/file/d/1jxWSlK1Zy_gBY_XO3xh6ydaLthqDh5Tm/view?usp=drive_link">Google Drive</a>.</li>
  <li>The Lung weight can be downloaded at <a href="https://drive.google.com/file/d/1jraG6uXrXEUyj-tFOMiEIGoDxDJeTQ_X/view?usp=sharing">Google Drive</a>.</li>
</ul>

## Usage

### Installation

```
conda create --name sam3d python=3.8
conda activate sam3d
pip install -r requirements.txt
```

The code is implemented based on ```pytorch 2.0.1``` with ```torchvision 0.15.2```. Please follow the instructions from the official PyTorch <a href="https://pytorch.org/get-started/locally/">website</a> to install the Pytorch, Torchvision and CUDA versions.

### Training

```
bash training_scripts/run_training_synapse.sh
bash training_scripts/run_training_acdc.sh
bash training_scripts/run_training_lung.sh
bash training_scripts/run_training_tumor.sh
```

### Evaluation

```
bash evaluation_scripts/run_evaluation_synapse.sh
bash evaluation_scripts/run_evaluation_acdc.sh
bash evaluation_scripts/run_evaluation_lung.sh
bash evaluation_scripts/run_evaluation_tumor.sh
```

## Predictions
<ul>
  <li>The pre-computed maps and scores of the Synapse dataset can be downloaded at <a href="https://drive.google.com/file/d/1Eb2o2b4TGNUyFpvdCQo3RRcxd2vB5J3x/view?usp=sharing">Google Drive</a>.</li>
  <li>The pre-computed maps and scores of the ACDC dataset can be downloaded at <a href="https://drive.google.com/file/d/19vVWDRnhSGFxFVcIzgF0LyJOyVGOs1UU/view?usp=sharing">Google Drive</a>.</li>
  <li>The pre-computed maps and scores of the BraTS dataset can be downloaded at <a href="https://drive.google.com/file/d/1tTbhgaBOcQ8Ww_rSCFvOyReDCaUcjue1/view?usp=sharing">Google Drive</a>.</li>
  <li>The pre-computed maps and scores of the Lung dataset can be downloaded at <a href="https://drive.google.com/file/d/1LqM0ZVwk6RLzqEodVACzTBZUAtNPdFT5/view?usp=sharing">Google Drive</a>.</li>
</ul>

## Citation
```
@article{sam3d,
      title={SAM3D: Segment Anything Model in Volumetric Medical Images}, 
      author={Nhat-Tan Bui and Dinh-Hieu Hoang and Minh-Triet Tran and Ngan Le},
      journal={arXiv:2309.03493},
      year={2023}
}
```

## Acknowledgment
A part of this code is adapted from these previous works: [UNETR++](https://github.com/Amshaker/unetr_plus_plus), [nnFormer](https://github.com/282857341/nnFormer), [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [SAM](https://github.com/facebookresearch/segment-anything).

## FAQ
If you have any questions, please feel free to create an issue on this repository or contact us at <tanb@uark.edu> / <hieu.hoang2020@ict.jvn.edu.vn>.
