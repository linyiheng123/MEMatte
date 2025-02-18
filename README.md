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


## ToDo
- [x] release UHRIM dataset
- [ ] release code and checkpoint


