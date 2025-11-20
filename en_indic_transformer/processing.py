from functools import partial
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

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = self.src[index]
        target_str = self.target[index]

        # ex: <|english|> ....src...<|endoftext|>
        inputs = self.tokenizer.encode(
            source, prefix_str=self.src_prepend, suffix_str=self.endoftext
        )

        # ex: <|english|> ....target...
        targets_in = self.tokenizer.encode(target_str, prefix_str=self.target_prepend)

        # ex: ....target...<|endoftext|>
        targets_out = self.tokenizer.encode(target_str, suffix_str=self.endoftext)

        # return a tuple back of encoded ids back.
        return (inputs, targets_in, targets_out)


class TranslationDataLoader(DataLoader):
    """
    Custom dataloader with a custom collate method that
    fills each batch with pad_tokens to make training easier.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        pad_val: int,
        ignore_index: int,
    ):
        # prefill the pad_val and ignore_index and create
        # a pre-defined method signature.
        collate_fn = partial(
            self.custom_collate_fn, pad_val=pad_val, ignore_index=ignore_index
        )
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    def custom_collate_fn(self, batch, pad_val: int, ignore_index: int):
        """
        Custom collation method, the source, target_in arrays are filled
        with 50256, i.e <|endoftext|> for now. Also the target_out array
        is fill with -100 for the loss function to ignore.
        source: the string/ids that is fed to encoder.
        target_in: the input that enters decoder.
        target_out: the output that is used for loss calculation.

        :param pad_val: Padding value for the sequence to match the longest
        sequence in the batch.
        :type pad_val: int

        :param ignore_index: Index value added to make response toknes (target_out)
        to match the longest in the sequence. But is filled with ignore_index (-100)
        to avoid loss calculation.
        :type ignore_index: int
        """
        sources, target_ins, target_outs = [], [], []

        for source, target_in, target_out in batch:
            sources.append(source)  # take all the source tokens
            target_ins.append(target_in)  # take all the target in tokens
            target_outs.append(target_out)  # take all the target out tokens

        # pad source, target_in, target_out tokens
        source_padded = pad_sequence(sources, batch_first=True, padding_value=pad_val)
        target_in_padded = pad_sequence(
            target_ins, batch_first=True, padding_value=pad_val
        )
        target_out_padded = pad_sequence(
            target_outs, batch_first=True, padding_value=ignore_index
        )

        return source_padded, target_in_padded, target_out_padded
