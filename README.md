<div align="center">   
  
# 명시적 감독 기반 이벤트-카메라 융합을 통한 3D 의미론적 장면 완성
</div>


</br>


## Abstract
Huma


## Method

| ![space-1.jpg](teaser/arch.png) | 
|:--:| 
| ***Figure 1. Overall framework of VoxFormer**. * |

## ⚙️ Dependencies & Installation
```bash
git clone https://github.com/mchaewon/masters_paper.git
cd masters_paper
```

**Docker** (recommended):

```bash
docker build -t event_ssc .
docker run -it --gpus '"device=0"' --shm-size 16G \
  -v $(pwd):/masters_paper -v /path/to/masters_paper/data:/masters_paper/dataset \
  event_ssc /bin/bash
```


## Model Zoo
Please download the trained models based on the following table.

| Backbone | Method | Lr Schd | IoU| mIoU | Config | Download |
| :---: | :---: | :---: | :---: | :---:| :---: | :---: |
| [R50](https://drive.google.com/file/d/1A4Efx7OQ2KVokM1XTbZ6Lf2Q5P-srsyE/view?usp=share_link) | VoxFormer-T | 20ep | 44.15| 13.35|[config](projects/configs/voxformer/voxformer-T.py) |[model](coming soon) |

 
## Dataset

- [x] SemanticKITTI

## Acknowledgement

Many thanks to these excellent open source projects:
- [VoxFormer](https://github.com/NVlabs/VoxFormer)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)