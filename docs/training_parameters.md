# Data

We trained model during 3 stages. Each stage requires different data. All data should be preprocessed with preprocessing script before. We use single GPU for all stages. More details can be found in the paper.
For stage1 we used shuffled 9m sentence from PIE corpus (a1 part only)
For stage2 we used shuffled combination of NUCLE, FCE, Lang8, W&I + locness datasets. Notice that we used dump of Lang8  which contained only 947,344 sentences (in 52.5% of them source/target sentences were different). If you use newer dump which has more sentences - consider sampling.
For stage3 we used shuffled version of W&I + locness datasets.


# Output vocabulary 

We used same fixed vocabulary for all stages (vocab_path=data/output_vocabulary)


# Number of epochs and early stopping

In our experiments, we used an early stopping mechanism and a fixed number of epochs.
```
  n_epoch: 20  
  patience: 3 
```
The problem with this approach is sensitivity to random seeds, model 
initialization, data order, etc. The longer you train, the higher recall you 
get, but for the price of precision, so it's important to stop training at the 
right time. For reproducibility reasons, we are providing further
 the exact number of epochs for each model and each stage. 



# Parameters

### Same parameters for all stages:
```
  tune_bert: 1  
  skip_correct: 1  
  skip_complex: 0   
  max_len: 50  
  min_len: 3  
  batch_size: 64  
  tag_strategy: keep_one  
  cold_steps_count: 0  
  cold_lr: 1e-3  
  lr: 1e-5  
  predictor_dropout: 0.0  
  lowercase_tokens: 0  
  pieces_per_token: 5  
  vocab_path: data/output_vocabulary  
  label_smoothing: 0.0
  patience: 0  
```

### Model specific parameters

#### XLNet:
```
  transformer_model: xlnet  
  special_tokens_fix: 0  
```

#### RoBERTA:
```
  transformer_model: roberta  
  special_tokens_fix: 1  
```


### Stage1 parameters:
```
  n_epoch: 20 
  cold_steps_count: 2  
  accumulation_size: 4  
  updates_per_epoch: 10000  
  tn_prob: 0  
  tp_prob: 1  
  pretrain: '' 
``` 

### Stage2 parameters:
```
  cold_steps_count: 2  
  accumulation_size: 2  
  updates_per_epoch: 0  
  tn_prob: 0  
  tp_prob: 1  
  pretrain: BEST_MODEL_FROM_STAGE1  
```

#### XLNet:
```
  n_epoch: 9 
```

#### RoBERTA:
```
  n_epoch: 5 
```

### Stage3 parameters:
```
  cold_steps_count: 0  
  accumulation_size: 2  
  updates_per_epoch: 0  
  tn_prob: 1  
  tp_prob: 1  
  pretrain: BEST_MODEL_FROM_STAGE2  
```

#### XLNet:
```
  n_epoch: 4 
```

#### RoBERTA:
```
  n_epoch: 3 
```

### For prediction during stage1-3 we used:
```
  iteration_count: 5  
  additional_confidence: 0  
  min_error_probability: 0  
```

### For getting best results after stage3 we used:
#### XLNet:
```
  additional_confidence: 0.35  
  min_error_probability: 0.66  
```
#### RoBERTa:
```
  additional_confidence: 0.2  
  min_error_probability: 0.5  
```

Notice that these parameters might need to be calibrated for your model. 
Consider using dev set for this. 

# Ensembles
For evaluating ensemble you need to name your models like "xlnet_0_SOMETHING.th", "roberta_1_SOMETHING.th" and pass them all to `model_path` parameter. You also need to set `is_ensemble` parameter.

