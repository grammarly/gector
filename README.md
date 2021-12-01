# GECToR – Grammatical Error Correction: Tag, Not Rewrite

This repository provides code for training and testing state-of-the-art models for grammatical error correction with the official PyTorch implementation of the following paper:
> [GECToR – Grammatical Error Correction: Tag, Not Rewrite](https://aclanthology.org/2020.bea-1.16/) <br>
> [Kostiantyn Omelianchuk](https://github.com/komelianchuk), [Vitaliy Atrasevych](https://github.com/atrasevych), [Artem Chernodub](https://github.com/achernodub), [Oleksandr Skurzhanskyi](https://github.com/skurzhanskyi) <br>
> Grammarly <br>
> [15th Workshop on Innovative Use of NLP for Building Educational Applications (co-located with ACL 2020)](https://sig-edu.org/bea/2020) <br>

It is mainly based on `AllenNLP` and `transformers`.
## Installation
The following command installs all necessary packages:
```.bash
pip install -r requirements.txt
```
The project was tested using Python 3.7.

## Datasets
All the public GEC datasets used in the paper can be downloaded from [here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data).<br>
Synthetically created datasets can be generated/downloaded [here](https://github.com/awasthiabhijeet/PIE/tree/master/errorify).<br>
To train the model data has to be preprocessed and converted to special format with the command:
```.bash
python utils/preprocess_data.py -s SOURCE -t TARGET -o OUTPUT_FILE
```
## Pretrained models
<table>
  <tr>
    <th>Pretrained encoder</th>
    <th>Confidence bias</th>
    <th>Min error prob</th>
    <th>CoNNL-2014 (test)</th>
    <th>BEA-2019 (test)</th>
  </tr>
  <tr>
    <td>BERT <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/bert_0_gectorv2.th">[link]</a></td>
    <td>0.1</td>
    <td>0.41</td>
    <td>61.0</td>
    <td>68.0</td>
  </tr>
  <tr>
    <td>RoBERTa <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th">[link]</a></td>
    <td>0.2</td>
    <td>0.5</td>
    <td>64.0</td>
    <td>71.8</td>
  </tr>
  <tr>
    <td>XLNet <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gectorv2.th">[link]</a></td>
    <td>0.2</td>
    <td>0.5</td>
    <td>63.2</td>
    <td>71.2</td>
  </tr>
</table>

**Note**: The scores in the table are different from the paper's ones, as the later version of transformers is used. To reproduce the results reported in the paper, use [this version](https://github.com/grammarly/gector/tree/fea1532608) of the repository. 

## Train model
To train the model, simply run:
```.bash
python train.py --train_set TRAIN_SET --dev_set DEV_SET \
                --model_dir MODEL_DIR
```
There are a lot of parameters to specify among them:
- `cold_steps_count` the number of epochs where we train only last linear layer
- `transformer_model {bert,distilbert,gpt2,roberta,transformerxl,xlnet,albert}` model encoder
- `tn_prob` probability of getting sentences with no errors; helps to balance precision/recall
- `pieces_per_token` maximum number of subwords per token; helps not to get CUDA out of memory

In our experiments we had 98/2 train/dev split.

## Training parameters
We described all parameters that we use for training and evaluating [here](https://github.com/grammarly/gector/blob/master/docs/training_parameters.md). 
<br>

## Model inference
To run your model on the input file use the following command:
```.bash
python predict.py --model_path MODEL_PATH [MODEL_PATH ...] \
                  --vocab_path VOCAB_PATH --input_file INPUT_FILE \
                  --output_file OUTPUT_FILE
```
Among parameters:
- `min_error_probability` - minimum error probability (as in the paper)
- `additional_confidence` - confidence bias (as in the paper)
- `special_tokens_fix` to reproduce some reported results of pretrained models

For evaluation use [M^2Scorer](https://github.com/nusnlp/m2scorer) and [ERRANT](https://github.com/chrisjbryant/errant).

## Text Simplification
This repository also implements the code of the following paper:
> [Text Simplification by Tagging](https://aclanthology.org/2021.bea-1.2/) <br>
> [Kostiantyn Omelianchuk](https://github.com/komelianchuk), [Vipul Raheja](https://github.com/vipulraheja), [Oleksandr Skurzhanskyi](https://github.com/skurzhanskyi) <br>
> Grammarly <br>
> [16th Workshop on Innovative Use of NLP for Building Educational Applications (co-located w EACL 2021)](https://sig-edu.org/bea/current) <br>

For data preprocessing, training and testing the same interface as for GEC could be used. For both training and evaluation stages `utils/filter_brackets.py` is used to remove noise. During inference, we use `--normalize` flag.

<table>
  <tr>
    <th></th>
    <th colspan="2">SARI</th>
    <th rowspan="2">FKGL</th>
  </tr>
    <th>Model</th>
    <th>TurkCorpus</th>
    <th>ASSET</th>
  </tr>
  <tr>
    <td>TST-FINAL <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_tst.th">[link]</a></td>
    <td>39.9</td>
    <td>40.3</td>
    <td>7.65</td>
  </tr>
  <tr>
    <td>TST-FINAL + tweaks</td>
    <td>41.0</td>
    <td>42.7</td>
    <td>7.61</td>
  </tr>
</table>

Inference tweaks parameters: <br>
```
iteration_count = 2
additional_keep_confidence = -0.68
additional_del_confidence = -0.84
min_error_probability = 0.04
```
For evaluation use [EASSE](https://github.com/feralvam/easse) package.

**Note**:  The scores in the table are very close to those in the paper, but not fully match them due to the 2 reasons:
- in the paper, we reported average scores of 4 models trained with different seeds;
- we merged codebases for GEC and Text Simplification tasks and updated them to the newer version of transformers lib.

## Noticeable works based on GECToR

- Improving Sequence Tagging approach for Grammatical Error Correction task [[paper](https://s3.eu-central-1.amazonaws.com/ucu.edu.ua/wp-content/uploads/sites/8/2021/04/Improving-Sequence-Tagging-Approach-for-Grammatical-Error-Correction-Task-.pdf)][[code](https://github.com/MaksTarnavskyi/gector-large)]
- LM-Critic: Language Models for Unsupervised Grammatical Error Correction [[paper](https://arxiv.org/pdf/2109.06822.pdf)][[code](https://github.com/michiyasunaga/LM-Critic)]

## Citation
If you find this work is useful for your research, please cite our papers:

#### GECToR – Grammatical Error Correction: Tag, Not Rewrite
```
@inproceedings{omelianchuk-etal-2020-gector,
    title = "{GECT}o{R} {--} Grammatical Error Correction: Tag, Not Rewrite",
    author = "Omelianchuk, Kostiantyn  and
      Atrasevych, Vitaliy  and
      Chernodub, Artem  and
      Skurzhanskyi, Oleksandr",
    booktitle = "Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications",
    month = jul,
    year = "2020",
    address = "Seattle, WA, USA â†’ Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.bea-1.16",
    pages = "163--170",
    abstract = "In this paper, we present a simple and efficient GEC sequence tagger using a Transformer encoder. Our system is pre-trained on synthetic data and then fine-tuned in two stages: first on errorful corpora, and second on a combination of errorful and error-free parallel corpora. We design custom token-level transformations to map input tokens to target corrections. Our best single-model/ensemble GEC tagger achieves an F{\_}0.5 of 65.3/66.5 on CONLL-2014 (test) and F{\_}0.5 of 72.4/73.6 on BEA-2019 (test). Its inference speed is up to 10 times as fast as a Transformer-based seq2seq GEC system.",
}
```

#### Text Simplification by Tagging
```
@inproceedings{omelianchuk-etal-2021-text,
    title = "{T}ext {S}implification by {T}agging",
    author = "Omelianchuk, Kostiantyn  and
      Raheja, Vipul  and
      Skurzhanskyi, Oleksandr",
    booktitle = "Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.bea-1.2",
    pages = "11--25",
    abstract = "Edit-based approaches have recently shown promising results on multiple monolingual sequence transduction tasks. In contrast to conventional sequence-to-sequence (Seq2Seq) models, which learn to generate text from scratch as they are trained on parallel corpora, these methods have proven to be much more effective since they are able to learn to make fast and accurate transformations while leveraging powerful pre-trained language models. Inspired by these ideas, we present TST, a simple and efficient Text Simplification system based on sequence Tagging, leveraging pre-trained Transformer-based encoders. Our system makes simplistic data augmentations and tweaks in training and inference on a pre-existing system, which makes it less reliant on large amounts of parallel training data, provides more control over the outputs and enables faster inference speeds. Our best model achieves near state-of-the-art performance on benchmark test datasets for the task. Since it is fully non-autoregressive, it achieves faster inference speeds by over 11 times than the current state-of-the-art text simplification system.",
}
```
