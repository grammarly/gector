# Ensembling and Knowledge Distilling of Large Sequence Taggers for Grammatical Error Correction

## Installation
The following command installs all necessary packages:
```.bash
pip install -r requirements.txt

python -m spacy download en_core_web_sm
```
The project was tested using Python 3.7.

## Datasets
All the public GEC datasets can be downloaded from [here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data).<br>
Knowledge distilled datasets can be downloaded [here](https://drive.google.com/drive/folders/1O2UL8e5dUzIESPkualuzKY05G1-BpwB-?usp=sharing).<br>
Synthetically PIE created datasets can be generated/downloaded [here](https://github.com/awasthiabhijeet/PIE/tree/master/errorify).<br>

To train the model data has to be preprocessed and converted to special format with the command:
```.bash
python utils/preprocess_data.py -s SOURCE -t TARGET -o OUTPUT_FILE
```
## Pretrained models
All available pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1zehxDugS2aJMMvrm8cCnEkLorKSi9nJv?usp=sharing).<br>
<table>
  <tr>
    <th>Pretrained encoder</th>
    <th>Confidence bias</th>
    <th>Min error prob</th>
    <th>BEA-2019 (test)</th>
  </tr>

  <tr>
    <th>RoBERTa <a href="https://drive.google.com/drive/folders/1Si2hwmskb7QxqSFtPBsivl_FujkR3p6l?usp=sharing">[link]</a></th>
    <th>0.1</th>
    <th>0.65</th>
    <th>73.1</th>
  </tr>
  <tr>
    <th>Large RoBERTa voc10k + DeBERTa voc10k + XLNet voc 5k <a href="https://drive.google.com/drive/folders/1SzkzVdjP30eWpHUvP5-BXMWu3szsf9Rt?usp=sharing">[link]</a></th>
    <th>0.3</th>
    <th>0.55</th>
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
- `transformer_model {bert, roberta, deberta, xlnet, bert-large, roberta-large, deberta-large, xlnet-large}` model encoder
- `tn_prob` probability of getting sentences with no errors; helps to balance precision/recall

In our experiments we had 98/2 train/dev split on each training stage.

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

### Ensembling by averaging of output tag probabilities

For evaluating ensemble you need to name your models like "xlnet_1_SOMETHING.th", "roberta_1_SOMETHING.th" and pass them all to `model_path` parameter. You also need to set `is_ensemble` parameter.

```.bash
python predict.py --model_path MODEL_PATH MODEL_PATH [MODEL_PATH ...] \
                  --vocab_path VOCAB_PATH --input_file INPUT_FILE \
                  --output_file OUTPUT_FILE \
                  --is_ensemble 1
```

### Ensembling by majority votes on output edit spans
For this ensemble, you first need to predict output files by singel models and them combine these files by script

```.bash
python ensemble.py --source_file SOURCE_FILE \
                         --target_files TARGET_FILE TARGET_FILE [TARGET_FILE ...]
                         --output_file OUTPUT_FILE
```

## Evaluation
For evaluation we use [ERRANT](https://github.com/chrisjbryant/errant).

## Citation
If you find this work is useful for your research, please cite our paper:

**Ensembling and Knowledge Distilling of Large Sequence Taggers for Grammatical Error Correction**

```
@inproceedings{tarnavskyi-etal-2022-improved-gector,
    title = "Ensembling and Knowledge Distilling of Large Sequence Taggers for Grammatical Error Correction",
    author = "Tarnavskyi, Maksym and Chernodub, Artem and Omelianchuk, Kostiantyn",
    booktitle = "Accepted for publication at 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022)",
    month = May,
    year = "2022",
    address = "Dublin, Ireland",
    url = "https://arxiv.org/pdf/2203.13064.pdf",
    abstract = "In this paper, we investigate improvements to the GEC sequence tagging architecture with a focus on ensembling of recent cutting-edge Transformer-based encoders in Large configurations. We encourage ensembling models by majority votes on span-level edits because this approach is tolerant to the model architecture and vocabulary size. Our best ensemble achieves a new SOTA result with an F0.5 score of 76.05 on BEA-2019 (test), even without pretraining on synthetic datasets. In addition, we perform knowledge distillation with a trained ensemble to generate new synthetic training datasets, "Troy-Blogs" and "Troy-1BW". Our best single sequence tagging model that is pretrained on the generated Troy- datasets in combination with the publicly available synthetic PIE dataset achieves a near-SOTA result with an F0.5 score of 73.21 on BEA-2019 (test). The code, datasets, and trained models are publicly available.",
}
```
