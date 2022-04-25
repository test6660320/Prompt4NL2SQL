# Data Preparation:

Add the sparc dataset to data/sparc

Then, run the following commands.
```yaml
# build the tokenizer for training and add the special tokens
python build_tokenizer.py
# preprocess the sparc dataset
python preprocess.py
# add turn switch data, the turn switch data is processed from RATSQL-TC
python turn_switch_aux_process.py
# tokenize data for training
python tokenize_dataset.py
```


# Training:
```yaml
build_tokenizer.py # construct data loader
pretrain.py # pretraining (use bash scripts)
inference.py # old inference codes for batch generation
inference_new.py # old inference codes for sequential generation
post_process.py # post_process the predict file for evaluation
```