from tokenizers import Tokenizer, models, pre_tokenizers, decoders

def create_tokenizer(savepath="tokenizer.json"):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.add_special_tokens(["[PAD]"])
    tokenizer.add_tokens(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.blank_token = tokenizer.token_to_id("[PAD]")
    tokenizer.save(savepath)
    return tokenizer

if __name__ == "__main__":
    tokenizer = create_tokenizer()
    print(sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))

    test_seq = 'i want to eat a cupcake'
    test_seq = test_seq.upper()
    print(tokenizer.encode(test_seq).ids)