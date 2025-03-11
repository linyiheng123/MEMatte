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


## Introduction
Transformer-based models have recently achieved outstanding performance in image matting. However, their application to high-resolution images remains challenging due to the quadratic complexity of global self-attention. To address this issue, we propose MEMatte, a memory-efficient matting framework for processing high-resolution images. MEMatte incorporates a router before each global attention block, directing informative tokens to the global attention while routing other tokens to a Lightweight Token Refinement Module (LTRM). Specifically, the router employs a local-global strategy to predict the routing probability of each token, and the LTRM utilizes efficient modules to simulate global attention. Additionally, we introduce a Batch-constrained Adaptive Token Routing (BATR) mechanism, which allows each router to dynamically route tokens based on image content and the stages of attention block in the network.

## Dataset
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
| MEMatte-ViTS | 13.90 | 11.17 | 10.94 | 12.78  | Coming Soon |


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


## License
The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source. 

## Acknowledgement
Our project is developed based on [ViTMatte](https://github.com/hustvl/ViTMatte). Thanks for their wonderful work!<div align="center">

