import torch
from tokenizers import Tokenizer

from .config import SOT, EOT, UNK

class EnTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self._check_vocab()

    def _check_vocab(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc and EOT in voc

    def text_to_tokens(self, text: str):
        ids = self.encode(text)
        return torch.IntTensor(ids).unsqueeze(0)

    def encode(self, txt: str, verbose=False):
        txt = txt.replace(' ', SPACE)
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(' ', '').replace(SPACE, ' ').replace(EOT, '').replace(UNK, '')
        return txt
