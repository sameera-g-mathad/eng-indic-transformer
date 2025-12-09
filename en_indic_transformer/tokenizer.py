import torch
import sentencepiece as sp

# import tiktoken


# this sytle follows my final capstone project chathist.
# I am coping the same format here.

# using sentencepiece to train a tokenizer to understand both
# kannada and hindi characters and also learn to tokenize them
# to subwords. Tiktoken tokenizes kannada and hindi words into byte
# level tokens.
# (ChatGPT: GPT-2’s tokenizer works byte-level, meaning
# every non-ASCII character is split into multiple byte tokens.
# For Hindi or Kannada characters, a single character like
# "क" or "ಹ" may become 2–4 tokens.)


class Tokenizer:
    """
    A wrapper for sentence piece class, with two methods, encode and decode.
    The purpose of this class is to return tensor during encoding and
    also avoid -100, i.e. the ignore index.
    """

    def __init__(self, tokenizer_path: str) -> None:
        """
        :param tokenizer_path: Path to load the tokenizer from.
        :type tokenizer_path: str.
        """
        self.sp = sp.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)

    @staticmethod
    def train(
        corpus_path: str,
        save_path: str,
        vocab_size: int,
        user_defined_symbols: set,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        split_by_whitespace: bool = False,
    ):
        """
        Static method to train the tokenizer using SentencePieceTrainer class.

        :param corpus_path: Path to the corpus file to train the tokenizer.
        :type corpus_path: str.

        """
        print("Training SentencePiece on the given data.")

        sp.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=save_path,
            vocab_size=vocab_size,
            user_defined_symbols=list(user_defined_symbols),
            character_coverage=character_coverage,
            model_type=model_type,
            train_extremely_large_corpus=True,
            split_by_whitespace=split_by_whitespace,
        )

    @property
    def n_vocab(self) -> int:
        """
        Returns the vocab size
        """
        return self.sp.GetPieceSize()

    def get_piece_id(self, word: str) -> int:
        """
        Method to return id for special allowed tokens
        that is otherwise returned as a list
        by encode method.
        """
        return self.sp.PieceToId(word)

    def encode(
        self, text: str, prefix_str: str | None = None, suffix_str: str | None = None
    ) -> torch.Tensor:
        """
        Uses `SentencePieceProcessor.Encode()` method under the hood.
        Returns a tensor of token ids.
        """

        # encode the given string to token_ids
        encoded_tensor = torch.tensor(self.sp.Encode(text))

        # if the prefix_string is present prepend to the
        # token ids
        if prefix_str is not None:
            # get the pieceId, as encoding this will return
            # a list, something sentencepiece does by adding
            # a '' or '_' to begenning of the words.
            prefix_tensor = torch.tensor([self.sp.PieceToId(prefix_str)])

            # concatenate the token_ids to form output.
            encoded_tensor = torch.cat([prefix_tensor, encoded_tensor], dim=0)

        if suffix_str is not None:
            suffix_tensor = torch.tensor([self.sp.PieceToId(suffix_str)])

            # concatenate the token_ids to form output.
            encoded_tensor = torch.cat([encoded_tensor, suffix_tensor], dim=0)

        # return the encoded tensor.
        return encoded_tensor

    def decode(self, ids: torch.Tensor, ignore_index: int = -100) -> str:
        """
        Uses `SentencePieceProcessor.Decode()` method under the hood.
        Purpose: Filter unwanted or ignore indeces if present.
        Returns a string.
        """
        _ids = [id for id in ids.tolist() if id != ignore_index]
        return self.sp.Decode(_ids)
