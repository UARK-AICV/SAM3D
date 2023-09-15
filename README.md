# SAM3D
Authors: Nhat-Tan Bui, Dinh-Hieu Hoang, Minh-Triet Tran, Ngan Le

This is the official implementation of <a href="https://arxiv.org/pdf/2309.03493v1.pdf">"SAM3D: Segment Anything Model in Volumetric Medical Images"</a>.

## Introduction
<image src="images/architecture.png">

Image segmentation is a critical task in medical image analysis, providing valuable information that helps to make an accurate diagnosis. In recent years, deep learning-based automatic image segmentation methods have achieved outstanding results in medical images. In this paper, inspired by the Segment Anything Model (SAM), a foundation model that has received much attention for its impressive accuracy and powerful generalization ability in 2D still image segmentation, we propose a SAM3D that targets at 3D volumetric medical images and utilizes the pre-trained features from the SAM encoder to capture meaningful representations of input images. Different from other existing SAM-based volumetric segmentation methods that perform the segmentation by dividing the volume into a set of 2D slices, our model takes the whole 3D volume image as input and processes it simply and effectively that avoids training a significant number of parameters. Extensive experiments are conducted on multiple medical image datasets to demonstrate that our network attains competitive results compared with other state-of-the-art methods in 3D medical segmentation tasks while being significantly efficient in terms of parameters.

<image src="images/decoder.png">

## Datasets and Trained Models
<ul>
  <li>To validate the effectiveness of our model, we conduct the experiments benchmark in four datasets: Synapse, ACDC, BRaTs, and Decathlon-Lung. We follow the same dataset preprocessing as in <a href="https://github.com/Amshaker/unetr_plus_plus">UNETR++</a> and <a href="https://github.com/282857341/nnFormer">nnFormer</a>. Please refer to their repositories for more details about organizing the dataset folders. Alternatively, you can download the preprocessed dataset </li>
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

The code is implemented based on ```pytorch 2.0.1``` with ```torchvision 0.15.2```. Please follow the instructions from the official PyTorch <a href="https://pytorch.org/get-started/locally/">website</a> to install the Pytorch, Torchvision and CUDA version.

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
A part of this code is adapted from these previous works: [UNETR++](https://github.com/Amshaker/unetr_plus_plus), [nnFormer](https://github.com/282857341/nnFormer) and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

## FAQ
If you have any questions, please feel free to create an issue on this repository or contact us at <tanb@uark.edu> / <hieu.hoang2020@ict.jvn.edu.vn>.
