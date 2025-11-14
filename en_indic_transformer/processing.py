import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from en_indic_transformer import Tokenizer


# this sytle follows my final capstone project chathist.
# I am coping the same format here.


class TranslationDataset(Dataset):
    """
    Custom dataset to return three values.
    An source text, target text and the slided
    target text for prediction (teacher forcing).
    """

    def __init__(
        self,
        src: list,
        target: list,
        tokenizer: Tokenizer,
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
        # ex: <|english|> ....src...<|endoftext|>
        source = self.src_prepend + self.src[index] + self.endoftext

        target_str = self.target[index]
        # ex: <|english|> ....target...
        # target_in = self.target_prepend + target_str
        target_in = self.target_prepend + target_str

        # ex: ....target...<|endoftext|>
        target_out = target_str + self.endoftext

        # return a tuple back of encoded ids back.
        return (
            self.tokenizer.encode(source, allowed_special=self.allowed_special),
            self.tokenizer.encode(target_in, allowed_special=self.allowed_special),
            self.tokenizer.encode(target_out, allowed_special=self.allowed_special),
        )


class TranslationDataLoader(DataLoader):
    """
    Custom dataloader with a custom collate method that
    fills each batch with pad_tokens to make training easier.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.custom_collate_fn,
        )

    def custom_collate_fn(self, batch):
        """
        Custom collation method, the source, target_in arrays are filled
        with 50256, i.e <|endoftext|> for now. Also the target_out array
        is fill with -100 for the loss function to ignore.
        soure: the string/ids that is fed to encoder.
        target_in: the input that enters decoder.
        target_out: the output that is used for loss calculation.
        """
        sources, target_ins, target_outs = [], [], []

        for source, target_in, target_out in batch:
            sources.append(source)
            target_ins.append(target_in)
            target_outs.append(target_out)

        source_padded = pad_sequence(sources, batch_first=True, padding_value=50256)
        target_in_padded = pad_sequence(
            target_ins, batch_first=True, padding_value=50256
        )
        target_out_padded = pad_sequence(
            target_outs, batch_first=True, padding_value=-100
        )

        return source_padded, target_in_padded, target_out_padded
