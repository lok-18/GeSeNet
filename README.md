# GeSeNet

[![IEEE](https://img.shields.io/badge/IEEE-TNNLS-purple)]([https://github.com/lok-18/GeSeNet/blob/main/LICENSE](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385))
[![LICENSE](https://img.shields.io/badge/License-MIT-green)](https://github.com/lok-18/GeSeNet/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7.0-orange)](https://pytorch.org/)

### *GeSeNet: A General Semantic-guided Network with Couple Mask Ensemble for Medical Image Fusion* [[*Paper*]](https://ieeexplore.ieee.org/document/10190200) [[*Google Scholar*]](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=GeSeNet%3A+A+General+Semantic-Guided+Network+With+Couple+Mask+Ensemble+for+Medical+Image+Fusion&btnG=)
in IEEE Transactions on Neural Networks and Learning Systems (**IEEE TNNLS**)  
by Jiawei Li, Jinyuan Liu, Shihua Zhou, Qiang Zhang and Nikola K. Kasabov
<div align=center>
<img src="https://github.com/lok-18/GeSeNet/blob/main/fig/network.png" width="100%">
</div>

### *Requirements* 
> - python 3.7  
> - torch 1.7.0
> - torchvision 0.8.0
> - opencv 4.5
> - numpy 1.21.6
> - pillow 9.4.0

### *Dataset setting*
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

### *Test*
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

### *Experimental results*
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

### *Citation*
> ```
> @article{li2023gesenet,
>   title={GeSeNet: A General Semantic-guided Network with Couple Mask Ensemble for Medical Image Fusion},
>   author={Li, Jiawei and Liu, Jinyuan and Zhou, Shihua and Zhang, Qiang and Kasabov, Nikola},
>   journal={IEEE Transactions on Neural Networks and Learning Systems},
>   year={2023},
>   publisher={IEEE}
> }
> ```

### *Realted works*
> - Jiawei Li, Jinyuan Liu, Shihua Zhou, Qiang Zhang and Nikola K. Kasabov. ***Learning a Coordinated Network for Detail-refinement Multi-exposure Image Fusion***. IEEE Transactions on Circuits and Systems for Video Technology (**IEEE TCSVT**), 2022, 33(2): 713-727. [[*Paper*]](https://ieeexplore.ieee.org/abstract/document/9869621)
> - Jiawei Li, Jinyuan Liu, Shihua Zhou, Qiang Zhang and Nikola K. Kasabov. ***Infrared and visible image fusion based on residual dense network and gradient loss***. Infrared Physics & Technology, 2023, 128: 104486. [[*Paper*]](https://www.sciencedirect.com/science/article/pii/S1350449522004674)
> - Jia Lei, Jiawei Li, Jinyuan Liu, Shihua Zhou, Qiang Zhang and Nikola K. Kasabov. ***GALFusion: Multi-exposure Image Fusion via a Global-local Aggregation Learning Network***. IEEE Transactions on Instrumentation and Measurement (**IEEE TIM**), 2023, 72: 1-15. [[*Paper*]](https://ieeexplore.ieee.org/abstract/document/10106641) [[*Code*]](https://github.com/jialei-sc/GALFusion)

### *Contact*
> If you have any questions, please create an issue or email to me ([Jiawei Li](ljw19970218@163.com)).
