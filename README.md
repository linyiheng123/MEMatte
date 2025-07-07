<div align="center">
<!-- <h2>Click2Trimap</h2> -->
<h3>Memory Efficient Matting with Adaptive Token Routing </h3>

Yiheng Lin, Yihan Hu, Chenyi Zhang, Ting Liu, Xiaochao Qu, Luoqi Liu, Yao Zhao, Yunchao Wei

Institute of Information Science, Beijing Jiaotong University  
Visual Intelligence + X International Joint Laboratory of the Ministry of Education  
Pengcheng Laboratory, Shenzhen, China  
MT Lab, Meitu Inc

<p align="center">
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg"/>
    </a>
    <a href="https://arxiv.org/pdf/2412.10702.pdf">
        <img src="https://img.shields.io/badge/arxiv-2412.10702-red"/>
    </a>   
    <a href="https://huggingface.co/datasets/dafbgd/UHRIM">
      <img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace">
    </a>
</p>
</div>

## 📮 News
- [2025.07] Release our interactive image matting model - [MattePro](https://github.com/ChenyiZhang007/MattePro).

## Introduction
Transformer-based models have recently achieved outstanding performance in image matting. However, their application to high-resolution images remains challenging due to the quadratic complexity of global self-attention. To address this issue, we propose MEMatte, a memory-efficient matting framework for processing high-resolution images. MEMatte incorporates a router before each global attention block, directing informative tokens to the global attention while routing other tokens to a Lightweight Token Refinement Module (LTRM). Specifically, the router employs a local-global strategy to predict the routing probability of each token, and the LTRM utilizes efficient modules to simulate global attention. Additionally, we introduce a Batch-constrained Adaptive Token Routing (BATR) mechanism, which allows each router to dynamically route tokens based on image content and the stages of attention block in the network.

## Dataset
Our proposed ultra high-resolution image matting datasets:
[`huggingface: dafbgd/UHRIM`](https://huggingface.co/datasets/dafbgd/UHRIM) 


## Quick Installation 
Run the following command to install required packages. 
```
pip install -r requirements.txt
```
Install [detectron2](https://github.com/facebookresearch/detectron2) please following its [document](https://detectron2.readthedocs.io/en/latest/), you can also run following command
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Results
Quantitative Results on [Composition-1k](https://paperswithcode.com/dataset/composition-1k)
| Model      | SAD   | MSE | Grad | Conn   | checkpoints |
| ---------- | ----- | --- | ---- | -----  | ----------- |
| MEMatte-ViTS | 21.90 | 3.37 | 7.43 | 16.77  | [GoogleDrive](https://drive.google.com/file/d/122p3sdhJVb7vg4IXELeC9C3HEG9Mlh5z/view?usp=sharing) |
| MEMatte-ViTB | 21.06 | 3.11 | 6.70 | 15.71  | [GoogleDrive](https://drive.google.com/file/d/1NOV64zMSFtoKPASqvEvxQKI_PRY9m5IA/view?usp=sharing) |

We also train a robust model for real-world images AIM-500 using mixed data:
| Model      | SAD   | MSE | Grad | Conn   | checkpoints |
| ---------- | ----- | --- | ---- | -----  | ----------- |
| MEMatte-ViTS | 13.90 | 11.17 | 10.94 | 12.78  | [GoogleDrive](https://drive.google.com/file/d/1R5NbgIpOudKjvLz1V9M9SxXr1ovAmu3u/view?usp=drive_link) |

## Train
1. Download the official checkpoints of ViTMatte ([ViTMatte_S_Com.pth](https://drive.google.com/file/d/12VKhSwE_miF9lWQQCgK7mv83rJIls3Xe/view), [ViTMatte_B_Com.pth](https://drive.google.com/file/d/1mOO5MMU4kwhNX96AlfpwjAoMM4V5w3k-/view?pli=1)), and then process the checkpoints using `pretrained/preprocess.py`.
2. Set `train.init_checkpoint` in the configs to specify the processed checkpoint.
3. Train the model with the following command:
```
python main.py \
        --config-file configs/MEMatte_S_topk0.25_win_global_long.py \
        --num-gpus 2
```


## Inference
```
python inference.py \
    --config-dir ./configs/CONFIG.py \
    --checkpoint-dir ./CHECKPOINT_PATH \
    --inference-dir ./SAVE_DIR \
    --data-dir /DataDir \
    --max-number-token Max_number_token 
```
For example:
```
python inference.py \
    --config-dir ./configs/MEMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/MEMatte_ViTS_DIM.pth \
    --inference-dir ./predAlpha/test_aim500 \
    --data-dir ./Datasets/AIM-500 \
    --max-number-token 18000
# Reducing the maximum number of tokens lowers memory usage.
```

## ToDo
- [x] release UHRIM dataset
- [x] release code and checkpoint

## Citation
If you have any questions, please feel free to open an issue. If you find our method or dataset helpful, we would appreciate it if you could give our project a star ⭐️ on GitHub and cite our paper:
```bibtex
@inproceedings{lin2025memory,
  title={Memory Efficient Matting with Adaptive Token Routing},
  author={Lin, Yiheng and Hu, Yihan and Zhang, Chenyi and Liu, Ting and Qu, Xiaochao and Liu, Luoqi and Zhao, Yao and Wei, Yunchao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={5298--5306},
  year={2025}
}
```

## License
The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source. 

## Acknowledgement
Our project is developed based on [ViTMatte](https://github.com/hustvl/ViTMatte), [DynamicViT](https://github.com/raoyongming/DynamicViT), [Matteformer](https://github.com/webtoon/matteformer), [ToMe](https://github.com/facebookresearch/ToMe), [EViT](https://github.com/youweiliang/evit). Thanks for their wonderful work!<div align="center">

