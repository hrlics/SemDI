# Advancing Event Causality Identification via Heuristic Semantic Consistency Inquiry Network

This repository is the official Pytorch implementation of HSemCD.




## Overview

![model](/imgs/HSemCD.png "HSemCD")

Event Causality Identification (ECI) focuses on extracting causal relations between events in texts. Existing methods primarily utilize contextual features and external knowledge bases (KB) to identify causality. However, such approaches fall short in two dimensions: (1) the causal features between events in a text often lack explicit clues, and (2) prior knowledge from external KB may introduce bias when inferencing causality in a given context. Given these issues, 
we introduce a novel **Semantic Consistency Inquiry (SemCI)** to the ECI task and propose the **H**euristic **Sem**antic **C**onsistency **D**iscriminator (HSemCD), a model that is both straightforward and effective. HSemCD utilizes a *Cloze* Analyzer to facilitate a gap-filling game, aiming to help uncover the causal chain in the context. Subsequently, it examines the semantic consistency between the fill-in token and the given sentence to detect the existence of the causal chain. Through this assessment, HSemCD reveals the causal relations between events indirectly. 



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Getting Started

### Data:
We have provided the processed data in `./dataset`

The raw data can be found at:

(1) EventStoryLine v0.9 (ESC): [https://github.com/tommasoc80/EventStoryLine](https://github.com/tommasoc80/EventStoryLine)

(2) Causal-TimeBank (CTB): [https://github.com/paramitamirza/Causal-TimeBank](https://github.com/paramitamirza/Causal-TimeBank)

&nbsp;
### Training

Go to the `code/Vision-Text/` folder and run the script vtcls_script.sh to start training:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
