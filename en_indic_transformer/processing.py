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
        src: list[str],
        target: list[str],
        tokenizer: Tokenizer,
        src_prepend_value: str,
        target_prepend_value: str,
        endoftext: str = "<|endoftext|>",
        max_length: int = 1024,
    ):
        """
        :param src: List of source sequences.
        :type src: list[str].
        :param target: List of target sequences.
        :type src: list[str].
        :param tokenizer: Tokenzier from `tokenizer.py` for
        tokenization of inputs.
        :type tokenizer: Tokenizer.
        :param src_preprend_value: Prefix string to prepend to
        input sequences.
        :type src_preprend_value: str.
        :param target_preprend_value: Prefix string to prepend to
        input sequences.
        :type target_preprend_value: str.
        :param endoftext: Suffix to append both input and target
        sequences.
        :type endoftext: str.

        :param max_length: Ususally the context length upto which the
                           sequence is allowed.
        :type max_length: int.
        """
        assert len(src) == len(
            target
        ), "Length of source and target lists should be equal"
        self.src = src
        self.target = target
        self.tokenizer = tokenizer
        self.src_prepend = src_prepend_value
        self.target_prepend = target_prepend_value
        self.endoftext = endoftext
        self.max_length = max_length

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = self.src[index]
        target_str = self.target[index]

        # ex: <|english|> ....src...<|endoftext|>
        inputs = self.tokenizer.encode(
            f"{source}", prefix_str=self.src_prepend, suffix_str=self.endoftext
        )

        # ex: <|english|> ....target...
        targets_in = self.tokenizer.encode(
            f"{target_str}", prefix_str=self.target_prepend
        )

        # ex: ....target...<|endoftext|>
        targets_out = self.tokenizer.encode(f"{target_str}", suffix_str=self.endoftext)

        if len(inputs) > self.max_length:
            inputs = torch.cat(
                [inputs[: self.max_length - 1], inputs[-1:]], dim=-1
            )  # truncate and add a endoftext token

        # targets_in doesn't have a eos token anyways.
        if len(targets_in) > self.max_length:
            targets_in = targets_in[: self.max_length]

        if len(targets_out) > self.max_length:
            targets_out = torch.cat(
                [targets_out[: self.max_length - 1], targets_out[-1:]], dim=-1
            )

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
        """
        :param dataset: Dataset to makes batches for training.
        :type dataset: torch.utils.data.Dataset.
        :param batch_size: Batch size to get the number of instaces.
        :type batch_size: int.
        :param shuffle: Boolean varaiable needed to shuffle the data
        before returning/retrieving.
        :type shuffle: bool.
         :param pad_val: Padding value for the sequence to match the longest
        sequence in the batch.
        :type pad_val: int.

        :param ignore_index: Index value added to make response toknes (target_out)
                            to match the longest in the sequence. But is filled with
                            ignore_index (-100) to avoid loss calculation.
        :type ignore_index: int.
        """
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

    def custom_collate_fn(self, batch, pad_val: int, ignore_index: int) -> tuple:
        """
        Custom collation method, the source, target_in arrays are filled
        with 50256, i.e <|endoftext|> for now. Also the target_out array
        is fill with -100 for the loss function to ignore.

        source: the string/ids that is fed to encoder.
        target_in: the input that enters decoder.
        target_out: the output that is used for loss calculation.

        :param pad_val: Padding value for the sequence to match the longest
        sequence in the batch.
        :type pad_val: int.

        :param ignore_index: Index value added to make response toknes (target_out)
                            to match the longest in the sequence. But is filled with
                            ignore_index (-100) to avoid loss calculation.
        :type ignore_index: int.

        :returns: A tuple of padded input, target_in and target out tokens along with
                  the original sequences.
        :rtype: tuple.
        """
        sources, targets_in, targets_out = [], [], []

        for source, target_in, target_out in batch:
            sources.append(source)  # take all the source tokens
            targets_in.append(target_in)  # take all the target in tokens
            targets_out.append(target_out)  # take all the target out tokens

        # pad source, target_in, target_out tokens
        source_padded = pad_sequence(sources, batch_first=True, padding_value=pad_val)
        target_in_padded = pad_sequence(
            targets_in, batch_first=True, padding_value=pad_val
        )
        target_out_padded = pad_sequence(
            targets_out, batch_first=True, padding_value=ignore_index
        )

        return (
            source_padded,
            target_in_padded,
            target_out_padded,
            sources,
            targets_in,
            targets_out,
        )
