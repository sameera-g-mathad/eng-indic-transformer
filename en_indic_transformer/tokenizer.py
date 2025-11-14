from typing import Optional
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

    def __init__(
        self, encoding_type: str = "gpt2", extend_base_encoder: set | None = None
    ) -> None:
        self._tokenizer = tiktoken.get_encoding(
            encoding_type
        )  # hard coding it to gpt2 as it is used here.

        # If the user wants to extend the tokenizer to have more special tokens.
        # ex: <|english|>, <|hindi|>, <|kannada|>.
        if extend_base_encoder is not None and len(extend_base_encoder) > 0:
            # use the current tokenizer as base tokenizer
            base_encoder = self._tokenizer
            # get the vocab size.
            vocab_size = base_encoder.n_vocab

            # create a dictionary for special tokens
            extend_token_set = {
                "<|endoftext|>": 50256  # add this for tiktoken to consider endoftext.
            }

            # if the tokens are not present in current tokenizer,
            # then extend the special token set.
            for token in extend_base_encoder:
                if token.encode("utf-8") not in base_encoder._mergeable_ranks:
                    extend_token_set[token] = vocab_size
                    vocab_size += 1

            # create new tokenizer.
            self._tokenizer = tiktoken.Encoding(
                name=encoding_type + "_extended",
                pat_str=base_encoder._pat_str,
                mergeable_ranks=base_encoder._mergeable_ranks,
                special_tokens=extend_token_set,
            )

    @property
    def n_vocab(self):
        """Return the length of the tokenizer"""
        return self._tokenizer.n_vocab

    def encode(self, text: str, allowed_special: Optional[set] = None) -> torch.Tensor:
        """
        Uses `tiktoken.encode()` method under the hood.
        Returns a tensor of token ids.
        """
        return torch.tensor(
            self._tokenizer.encode(
                text,
                allowed_special=(
                    allowed_special if allowed_special is not None else {"all"}
                ),
            )
        )

    def decode(self, ids: torch.Tensor) -> str:
        """
        Uses `tiktoken.decode()` method under the hood.
        Purpose: Filter unwanted or ignore indeces if present.
        Returns a string.
        """
        _ids = [id for id in ids.tolist() if id != -100]
        return self._tokenizer.decode(_ids)
