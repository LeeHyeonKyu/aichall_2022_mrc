from transformers import ElectraTokenizerFast, AutoTokenizer


tokenizer_dict = {'ElectraTokenizerFast': ElectraTokenizerFast}

def get_tokenizer(tokenizer_name, pretrained):
    if tokenizer_name in tokenizer_dict:
        tokenizer = tokenizer_dict[tokenizer_name].from_pretrained(pretrained)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained)
        except:
            raise ValueError(f"No such tokenizer {tokenizer_name} defined!")

    return tokenizer