import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    def __init__(
        self,
        src: list,
        target: list,
        tokenizer: tiktoken.core.Encoding,
        src_prepend_value: str,
        target_prepend_value: str,
        endoftext: str = "<|endoftext|>",
    ):
        assert len(src) == len(
            target
        ), "Length of source and target lists should be equal"
        self.src = src
        self.target = target
        self.tokenizer = tokenizer
        self.src_prepend = src_prepend_value
        self.target_prepend = target_prepend_value
        self.endoftext = endoftext
        self.allowed_special: set = set(
            [src_prepend_value, target_prepend_value, endoftext]
        )

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = self.src_prepend + self.src[index] + self.endoftext
        target_str = self.target[index]
        target_in = self.target_prepend + target_str
        target_out = target_str + self.endoftext

        return (
            torch.tensor(
                self.tokenizer.encode(source, allowed_special=self.allowed_special)
            ),
            torch.tensor(
                self.tokenizer.encode(target_in, allowed_special=self.allowed_special)
            ),
            torch.tensor(
                self.tokenizer.encode(target_out, allowed_special=self.allowed_special)
            ),
        )
