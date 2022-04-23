python3 pretrain.py\
    --dataset_prefix_path ../data/text2sql/data/sparc_tokenized_uncased_with_rc_tw_with_final/\
    --tokenizer_prefix_path ../data/text2sql/tokenizer_with_rc_tw_gf \
    --model_name t5-base\
    --batch_size_per_gpu 6\
    --number_of_gpu 1\
    --gradient_accumulation_steps 2\
    --save_steps 2500\
    --save_path ./t5_base_baseline_uncased/\
    --save_ckpt_name pre-trained





