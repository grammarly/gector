# GECToR – Grammatical Error Correction: Tag, Not Rewrite

This repository provides code for training and testing state-of-the-art models for grammatical error correction with the official PyTorch implementation of the following paper:
> GECToR – Grammatical Error Correction: Tag, Not Rewrite <br>
> [Kostiantyn Omelianchuk](https://github.com/komelianchuk), [Vitaliy Atrasevych](https://github.com/atrasevych), [Artem Chernodub](https://github.com/achernodub), [Oleksandr Skurzhanskyi](https://github.com/skurzhanskyi) <br>
> Grammarly <br>
> LINK ON ARXIV [TBD] <br>
> [15th Workshop on Innovative Use of NLP for Building Educational Applications (co-located with ACL 2020)](https://sig-edu.org/bea/current) <br>

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
    <th>CoNNL-2014</th>
    <th>BEA-2019</th>
  </tr>
  <tr>
    <th>BERT <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/bert_0_gector.th">[link]</a></th>
    <th>0.10</th>
    <th>0.41</th>
    <th>63.0</th>
    <th>67.6</th>
  </tr>
  <tr>
    <th>RoBERTa <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gector.th">[link]</a></th>
    <th>0.20</th>
    <th>0.50</th>
    <th>64.0</th>
    <th>71.5</th>
  </tr>
  <tr>
    <th>XLNet <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gector.th">[link]</a></th>
    <th>0.35</th>
    <th>0.66</th>
    <th>65.3</th>
    <th>72.4</th>
  </tr>
  <tr>
    <th>RoBERTa + XLNet</th>
    <th>0.24</th>
    <th>0.45</th>
    <th>66.0</th>
    <th>73.7</th>
  </tr>
  <tr>
    <th>BERT + RoBERTa + XLNet</th>
    <th>0.16</th>
    <th>0.40</th>
    <th>66.5</th>
    <th>73.6</th>
  </tr>
</table>

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
## Model inference
To run your model on the input file use the following command:
```.bash
python test_model.py --model_path MODEL_PATH [MODEL_PATH ...] \
                     --vocab_path VOCAB_PATH --input_file INPUT_FILE \
                     --output_file OUTPUT_FILE
```
Among parameters:
- `min_error_probability` - minimum error probability (as in the paper)
- `additional_confidence` - confidence bias (as in the paper)
- `special_tokens_fix` to reproduce some reported results of pretrained models

For evaluation use [M^2Scorer](https://github.com/nusnlp/m2scorer) and [ERRANT](https://github.com/chrisjbryant/errant).
## Citation
If you find this work is useful for your research, please cite our paper:
```
TBD
```
