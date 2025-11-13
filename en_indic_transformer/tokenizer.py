import torch
import tiktoken

# this sytle follows my final capstone project chathist.
# I am coping the same format here.


class Tokenizer:
    """
    A wrapper for tiktoken class, with two methods, encode and decode.
    The purpose of this class is to return tensor during encoding and
    also avoid -100, i.e. the ignore index. Have hard coded for now.
    """

    def __init__(self, encoding_type: str = "gpt2") -> None:
        self.tokenizer = tiktoken.get_encoding(
            encoding_type
        )  # hard coding it to gpt2 as it is used here.

    def encode(self, text: str, allowed_special: set) -> torch.Tensor:
        """
        Uses `tiktoken.encode()` method under the hood.
        Returns a tensor of token ids.
        """
        return torch.tensor(
            self.tokenizer.encode(text, allowed_special=allowed_special)
        )

    def decode(self, ids: torch.Tensor) -> str:
        """
        Uses `tiktoken.decode()` method under the hood.
        Purpose: Filter unwanted or ignore indeces if present.
        Returns a string.
        """
        _ids = [id for id in ids.tolist() if id != -100]
        return self.tokenizer.decode(_ids)
