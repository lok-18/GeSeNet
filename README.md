# GeSeNet

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/UMF-CMGR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.7.0-orange)](https://pytorch.org/)

### GeSeNet: A General Semantic-guided Network with Couple Mask Ensemble for Medical Image Fusion
in IEEE Transactions on Neural Networks and Learning Systems (***IEEE TNNLS***)  
by Jiawei Li, Jinyuan Liu, Shihua Zhou, Qiang Zhang and Nikola K. Kasabov

<div align=center>
<img src="https://github.com/lok-18/GeSeNet/blob/main/fig/network.png" width="100%">
</div>

### Requirements  
> - python 3.7  
> - torch 1.7.0
> - torchvision 0.8.0
> - opencv 4.5
> - numpy 1.21.6
> - pillow 9.4.0

### Dataset setting
> We give 5 test image pairs as examples in three modalities (i.e., MRI-CT, MRI-PET, MRI-SPECT), respectively.
> 
> Moreover, you can set your own test datasets of different modalities under ```./test_images/...```, like:   
> ```
> test_images
> ├── MRI_CT
> |   ├── CT
> |   |   ├── 1.png
> |   |   ├── 2.png
> |   |   └── ...
> |   ├── MRI
> |   |   ├── 1.png
> |   |   ├── 2.png
> |   |   └── ...
> ```
> The datasets in our paper are all from: [Harvard medical images](http://www.med.harvard.edu/AANLIB/)

### Test
> The pre-trained model has given in ```./model/GeSeNet.pth```.
> Please run ```test.py``` to get fused results, and you can check them in:
> ```
> results
> ├── MRI_CT
> |   ├── 1.png
> |   ├── 2.png
> |   └── ...
> ├── MRI_PET
> |   ├── 1.png
> |   ├── 2.png
> |   └── ...
> ├── MRI_SPECT
> |   ├── 1.png
> |   ├── 2.png
> |   └── ...
> ```

### Experimental results
> The qualitative comparison results of our GeSeNet with nine state-of-the-art methods on MRI-CT, MRI-PET and MRI-SPECT image pairs.
> <div align=center>
> <img src="https://github.com/lok-18/GeSeNet/blob/main/fig/MRI_CT.png" width="100%">
> </div> 
> <div align=center>
> <img src="https://github.com/lok-18/GeSeNet/blob/main/fig/MRI_PET.png" width="100%">
> </div>
> <div align=center>
> <img src="https://github.com/lok-18/GeSeNet/blob/main/fig/MRI_SPECT.png" width="100%">
> </div>
> Please refer to the paper for more experimental results and details.

### Citation
> ```
> We will updata the BibTex after it is published.
> ```

### Contact
If you have any questions, please create an issue or email to me ([Jiawei Li](ljw19970218@163.com)).
