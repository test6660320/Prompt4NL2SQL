if __name__ == '__main__':
    import os
    save_path = r'./'
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)
    
    # sos_eos_tokens = ['<_PAD_>', '<go_r>', '<go_b>', '<go_a>', '<eos_u>', '<eos_r>', '<eos_b>',
    #             '<eos_a>', '<go_d>','<eos_d>', '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>',
    #             '<sos_db>', '<eos_db>', '<sos_context>', '<eos_context>']

    # # complex
    # sos_eos_tokens = ['<_PAD_>', '<sos_u>', '<eos_u>', '<sos_q>', '<eos_q>', '<sos_s>', '<eos_s>', '<p_key>', '<f_key>',
    #                   '<sos_context>', '<eos_context>','<table>', '<number>', '<text>', '<boolean>', '<others>',
    #                   '<time>', '<column>', '<sos_rc>', '<eos_rc>']

    # # with ts
    # sos_eos_tokens = ['<_PAD_>', '<sos_u>', '<eos_u>', '<sos_q>', '<eos_q>', '<sos_s>', '<eos_s>', '<p_key>', '<f_key>',
    #                   '<sos_context>', '<eos_context>', '<table>', '<number>', '<text>', '<boolean>', '<others>',
    #                   '<time>', '<column>', '<sos_rc>', '<eos_rc>', '<sos_tw>', '<eos_tw>']

    # with gf
    sos_eos_tokens = ['<_PAD_>', '<sos_u>', '<eos_u>', '<sos_q>', '<eos_q>', '<sos_s>', '<eos_s>', '<p_key>', '<f_key>',
                      '<sos_context>', '<eos_context>', '<table>', '<number>', '<text>', '<boolean>', '<others>',
                      '<time>', '<column>', '<sos_rc>', '<eos_rc>', '<sos_tw>', '<eos_tw>', '<sos_gf>', '<eos_gf>']

    # initialize tokenizer
    print ('Saving Tokenizer...')
    from transformers import T5Tokenizer
    model_name = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(sos_eos_tokens)
    tokenizer_save_path = save_path + r'/tokenizer_with_rc_tw_gf'
    tokenizer.save_pretrained(tokenizer_save_path)
    print ('Tokenizer Saved.')
