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

Go to the `src` folder and run the scripts to start training: 

(1) ESC: ```sh train_ESC.sh```

(2) ESC<sup>*</sup>: `sh train_ESCstar.sh`

(3) CTB: `sh train_CTB.sh`

> If the paper is accepted, we will release the full version, i.e., the total processed data for training.

### Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

### Results

Our model achieves the following performance on :

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |
| My awesome model   |     85%         |      95%       |
| My awesome model   |     85%         |      95%       |
| My awesome model   |     85%         |      95%       |
| My awesome model   |     85%         |      95%       |
| My awesome model   |     85%         |      95%       |



