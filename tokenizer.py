import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def encode(self, text):
        return [self.bos_token_id] + self.sp.encode(text) + [self.eos_token_id]

    def decode(self, ids):
        # BOS ve EOS'u çıkar
        ids = [i for i in ids if i not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]]
        return self.sp.decode(ids)


# Örnek kullanım
# tokenizer = SentencePieceTokenizer('tr_tokenizer.model')
# tokens = tokenizer.encode("Merhaba dünya")
# print(tokens)  # [2, 1234, 5678, 3]
# print(tokenizer.decode(tokens))  # "Merhaba dünya"

# Model eğitimi (bir kez yapılır)
# spm.SentencePieceTrainer.train(
    #   input='turkish_corpus.txt',
    #  model_prefix='tr_tokenizer',
    # vocab_size=32000,
    # model_type='bpe',  # veya 'unigram'
    # pad_id=0,
    # unk_id=1,
    # bos_id=2,
    # eos_id=3
# )
