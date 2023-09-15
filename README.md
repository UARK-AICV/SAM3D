# SAM3D
Authors: Nhat-Tan Bui, Dinh-Hieu Hoang, Minh-Triet Tran, Ngan Le

This is the official implementation of <a href="https://arxiv.org/pdf/2309.03493v1.pdf">"SAM3D: Segment Anything Model in Volumetric Medical Images"</a>.

## Introduction
<image src="images/architecture.png">

Image segmentation is a critical task in medical image analysis, providing valuable information that helps to make an accurate diagnosis. In recent years, deep learning-based automatic image segmentation methods have achieved outstanding results in medical images. In this paper, inspired by the Segment Anything Model (SAM), a foundation model that has received much attention for its impressive accuracy and powerful generalization ability in 2D still image segmentation, we propose a SAM3D that targets at 3D volumetric medical images and utilizes the pre-trained features from the SAM encoder to capture meaningful representations of input images. Different from other existing SAM-based volumetric segmentation methods that perform the segmentation by dividing the volume into a set of 2D slices, our model takes the whole 3D volume image as input and processes it simply and effectively that avoids training a significant number of parameters. Extensive experiments are conducted on multiple medical image datasets to demonstrate that our network attains competitive results compared with other state-of-the-art methods in 3D medical segmentation tasks while being significantly efficient in terms of parameters.

<image src="images/decoder.png">

## Datasets and Trained Models
<ul>
  <li>To validate the effectiveness of our model, we conduct the experiments benchmark in four datasets: Synapse, ACDC, BRaTs, and Decathlon-Lung. We follow the same dataset preprocessing as in <a href="https://github.com/Amshaker/unetr_plus_plus">UNETR++</a> and <a href="https://github.com/282857341/nnFormer">nnFormer</a>. Please refer to their repositories for more details about downloading and organizing the dataset folders.</li>
  <li>.</li>
  <li>.</li>
  <li>.</li>
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
