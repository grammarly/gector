# Improving Sequence Tagging for Grammatical Error Correction

This repository provides several improvemets to state-of-the-art sequence tagging model for grammatical error correction descibed in following thesis:
> [Improving Sequence Tagging for Grammatical Error Correction](https://drive.google.com/file/d/17-qXILfafHR8Uv2Y9plcB9WVRdZLazzp/view?usp=sharing) <br>


The code in this repository mainly based on the [official implementation](https://github.com/grammarly/gector) from following paper:
> [GECToR – Grammatical Error Correction: Tag, Not Rewrite](https://arxiv.org/abs/2005.12592) <br>

## Installation
The following command installs all necessary packages:
```.bash
pip install -r requirements.txt
```
The project was tested using Python 3.7.

## Datasets
All the public GEC datasets used in the thesis can be downloaded from [here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data).<br>
Knowledge distilled datasets can be downloaded [here](https://drive.google.com/drive/folders/10-rECSEFvFpDf8wXDP9l_58mXhDPWjP5?usp=sharing).<br>
Synthetically PIE created datasets can be generated/downloaded [here](https://github.com/awasthiabhijeet/PIE/tree/master/errorify).<br>


To train the model data has to be preprocessed and converted to special format with the command:
```.bash
python utils/preprocess_data.py -s SOURCE -t TARGET -o OUTPUT_FILE
```
## Pretrained models
<table>
  <tr>
    <th>Pretrained encoder</th>
    <th>BEA-2019 (test)</th>
  </tr>
  
  <tr>
    <th>RoBERTa <a href="https://drive.google.com/file/d/1WvPNrpaJ5QLaoxN0vZCwTAhGNLRHUXvd/view?usp=sharing">[link]</a></th>
    <th>73.1</th>
  </tr>
  <tr>
    <th>Large RoBERTa voc10k + DeBERTa voc10k + XLNet voc 5k <a href="https://drive.google.com/drive/folders/1p5TSJroj8zflB8wWJJR6BLWcE_y0GScM?usp=sharing">[link]</a></th>
    <th>76.05</th>
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

For evaluation we use [ERRANT](https://github.com/chrisjbryant/errant).


