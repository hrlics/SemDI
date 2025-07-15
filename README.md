<h3 align="center"><a href="https://aclanthology.org/2024.emnlp-main.87.pdf" style="color:#9C276A">
Advancing Event Causality Identification via Heuristic Semantic Dependency Inquiry Network</a></h3>

<h5 align="center">
 
![Static Badge](https://img.shields.io/badge/License-MIT-green) 
[![Static Badge](https://img.shields.io/badge/Paper-EMNLP_2024-red)](https://aclanthology.org/2024.emnlp-main.87.pdf)
</h5>

## 📣 News

- **\[07/15/2025\]** We release all the data (ESC, ESC*, and CTB) under `src/dataset`. If you find it helpful, please consider citing and giving us a star 🌟 !
- **\[09/20/2024\]** Our paper is accepted by [EMNLP 2024 main](https://aclanthology.org/2024.emnlp-main.87/) !


## :telescope: Overview

<img src="/imgs/SemDI.png"/>

Event Causality Identification (ECI) focuses on extracting causal relations between events in texts. Existing methods for ECI primarily rely on causal features and external knowledge. However, these approaches fall short in two dimensions: (1) causal features between events in a text often lack explicit clues, and (2) external knowledge may introduce bias, while specific problems require tailored analyses. To address these issues, we propose SemDI - a simple and effective **Sem**antic **D**ependency **I**nquiry Network for ECI. SemDI captures semantic dependencies within the context using a unified encoder. Then, it utilizes a *Cloze* Analyzer to generate a fill-in token based on comprehensive context understanding. Finally, this fill-in token is used to inquire about the causal relation between two events. Extensive experiments demonstrate the effectiveness of SemDI, surpassing state-of-the-art methods on three widely used benchmarks.

## :bulb: A Quick Checkout

We have provided a jupyter notebook to run fast evaluations on ESC, ESC<sup>*</sup>, and CTB:

```eval
src/evaluate_demo.ipynb
```

## :clipboard: Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## :rocket: Getting Started

### Data:

We have provided the processed data under `src/dataset`. Each datapoint follows the format below:
```
[
  'ABC19980108.1830.0711.xml', # file path in raw data
  'The financial assistance from the World Bank and the International Monetary Fund are not helping .', # context
  'assistance', # event_1
  'helping', # event_2
  'non-causal' # label
],
```


The raw data can be found at: [EventStoryLine v0.9 (ESC)](https://github.com/tommasoc80/EventStoryLine), [Causal-TimeBank (CTB)](https://github.com/paramitamirza/Causal-TimeBank).




### Training

Under `src` directory, run the following scripts to start training: 

(1) ESC: 
```
  sh train_ESC.sh
```

(2) ESC<sup>*</sup>: 
```
  sh train_ESCstar.sh
```

(3) CTB: 
```
  sh train_CTB.sh
```


## 📚 Citation
If you find our work useful, please consider citing:
```
@inproceedings{li-etal-2024-advancing-event,
    title = "Advancing Event Causality Identification via Heuristic Semantic Dependency Inquiry Network",
    author = "Li, Haoran  and
      Gao, Qiang  and
      Wu, Hongmei  and
      Huang, Li",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.87",
    pages = "1467--1478",
}
```



