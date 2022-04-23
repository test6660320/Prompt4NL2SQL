python3 pretrain.py\
    --dataset_prefix_path ../data/text2sql/data/sparc_tokenized_cased/\
    --tokenizer_prefix_path ../data/text2sql/ \
    --model_name t5-large\
    --batch_size_per_gpu 5\
    --number_of_gpu 1\
    --gradient_accumulation_steps 2\
    --save_steps 2500\
    --save_path ./t5_large_baseline_cased/\
    --save_ckpt_name large