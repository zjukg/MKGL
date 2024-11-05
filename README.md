## [MGKL: Mastery of a Three-Word Language](https://openreview.net/forum?id=eqMNwXvOqn)
![](https://img.shields.io/badge/version-1.0.0-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/MKGL)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-orange)](https://huggingface.co/)
[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-%23bd9f65?labelColor=%23bea066&color=%23ffffff)](https://neurips.cc/)

Welcome to the repository for the model of KG Language (MKGL). This project investigates the potential of LLMs in understanding and interacting with knowledge graphs, a domain that has received limited exploration in the context of NLP.


<div align="center">
    <img src="https://github.com/zjukg/MKGL/blob/main/imgs/arch.jpg" width="70%" height="auto" />
</div>

### Overview
Large language models (LLMs) have significantly advanced performance across a spectrum of natural language processing (NLP) tasks. Yet, their application to knowledge graphs (KGs), which describe facts in the form of triplets and allow minimal hallucinations, remains an underexplored frontier. In this project, we investigate the integration of LLMs with KGs by introducing a specialized KG Language (KGL), where a sentence precisely consists of an entity noun, a relation verb, and ends with another entity noun. 
    <img src="https://github.com/zjukg/MKGL/blob/main/imgs/klora.jpg" width="70%" height="auto" />
</div>


### Environment

To run this project, please first install all required packages:

```
pip install --upgrade pandas transformers peft==0.9 bitsandbytes swifter deepspeed easydict pyyaml
```

please kindly install the pyg packages via wheels, which is much faster:

```
pip install --find-links MKGL/pyg_wheels/ torch-scatter torch-sparse torchdrug
```
### Preprocessing

Then, we need to preprocess the datasets,

for standard KG completion:

```
python preprocess.py -c config/fb15k237.yaml

python preprocess.py -c config/wn18rr.yaml
```

for inductive setting:

```
python preprocess.py -c config/fb15k237_ind.yaml --version v1

python preprocess.py -c config/wn18rr_ind.yaml --version v1
```

### Run with Single GPU

If you only has one GPU (better has 80GB memory under the default setting), please run the model with the following command:

```
python main.py -c config/fb15k237.yaml
```

### Run with Multiple GPU

If you can access multiple GPUs, please run the model with the following command:

```
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/fb15k237.yaml
```

### Run with script

Please kindly use the provide scripts to run the model:

```
sh scripts/fb15k237.sh
```

### Cite

Please condiser citing our paper if it is helpful to your work!

```bigquery
@inproceedings{MKGL,
  author       = {Lingbing Guo and
                  Zhongpu Bo and 
                  Zhuo Chen and 
                  Yichi Zhang and
                  Jiaoyan Chen and
                  Lan Yarong and
                  Mengshu Sun and
                  Zhiqiang Zhang and
                  Yangyifei Luo and
                  Qian Li and
                  Qiang Zhang and
                  Wen Zhang and
                  Huajun Chen},
  title        = {MGKL: Mastery of a Three-Word Language},
  booktitle    = {{NeurIPS}},
  year         = {2024}
}
```

### Thanks

We appreciate [LLaMA](https://github.com/facebookresearch/llama), [Huggingface Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama), [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora), and many other related works for their open-source contributions.

