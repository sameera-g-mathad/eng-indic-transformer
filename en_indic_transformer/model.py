from pathlib import Path
from typing import Any, Generator, Literal, Optional
from tqdm.auto import tqdm
from nltk.translate.bleu_score import corpus_bleu
import torch
from torch import nn, optim
from en_indic_transformer import Transformer, Tokenizer

DeviceType = Literal["cpu", "cuda"]  # allowed device values.


class Trainer:
    """
    A singleton class that is used to train the model.
    This class is written specifically for transformer class
    that can be found in `components.py`. The reason is simply
    because of the existance of encoder. Two inputs are passed
    instead of single item.
    """

    _instance: Optional["Trainer"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Trainer, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model: Transformer,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        tokenizer: Tokenizer,
        save_path: Path,
    ):
        """
        :param model: The model to be trained by the trainer.
        :type model: Transformer.
        :param loss_fn: The loss entity.
        :type loss_fn: torch.nn.Module.
        :param optimizer: Optimizer for training.
        :type optimizer: torch.optim.Optimizer.
        :param tokenizer: Tokenzier from `tokenizer.py` for
        tokenization of inputs.
        :type tokenizer: Tokenizer.
        :param save_path: Path to store and load models during
        training and inference. Ideally, the home dir of the project.
        :type save_path: pathlib.Path.
        """
        # checks if there is an instance
        # already.
        if self._initialized:
            return
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self._initialized = True
        # self.path = save_path
        # this stores only the model.
        model_dir = save_path / "models"
        self.model_path = model_dir / "model.pt"

        # stores optimizer state that would be optional to
        # download.
        checkpoint_dir = save_path / "checkpoints"
        self.optimizer_path = checkpoint_dir / "optimizer.pt"

        # create directory for models and optimizer.
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # # load the model if one exists already.
        # self._load_checkpoint()

    def move(
        self,
        x: torch.Tensor,
        y_in: torch.Tensor,
        y_out: torch.Tensor | None = None,
        device: DeviceType = "cpu",
    ) -> tuple:
        """
        Moves the passed tensors into their respective device.

        :param x: The input tensor to move to the device.
        :type x: torch.Tensor.
        :param y_in: The target_in tensor to move to the device.
        :type y_in: torch.Tensor.
        :param y_out: The target_out tensor to move to the device **(optional)**.
        :type y_out: torch.Tensor | None.
        :param device: Device to move the tensors to.
        :type device: 'cpu' | 'cuda'.
        :returns: Tuple of tensors moved to the specified devices.
        :rtype: tuple
        """
        x = x.to(device)
        y_in = y_in.to(device)
        if y_out is not None:
            y_out = y_out.to(device)

        # return the newly moved input/outputs
        return x, y_in, y_out

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader | None,
        epochs: int,
        device: DeviceType,
        batch_size_to_predict: int | None,
        predict_input: str | None = None,
        target_prefix: str | None = None,
        actual_target: str | None = None,
        max_tokens: int | None = 100,
    ) -> dict:
        """
        To Train the model that is clearly of Transformer instance.

        If test_dataloader is provided, then `Tokenizer.test()` is called
        to run the model to evaluate the model on unseen data.

        If predict_input and target_prefix is provided, then `Tokenizer.predict()`
        method is called internally and the result is printed.

        :param train_dataloader: Train dataloader.
        :type train_dataloader: torch.utils.data.DataLoader.
        :param test_dataloader: Test dataloader that is optional, if
                                provided, model will be tested against
                                test data and loss is returned.
        :type test_dataloader: torch.utils.data.DataLoader | None.
        :param epochs: The number of epochs to train the model.
        :type epochs: int.
        :param device: Device to move the tensors to.
        :type device: 'cpu' | 'cuda'.
        :param batch_size_to_predict: Batch size after which the model will be tested
                                      for predictions.
        :type batch_size_to_predict: int | None
        :param predict_input: The input string of the source text to predict the
        performance of the model after every epoch which is optional.
        :type predict_input: str | None.
        :param target_prefix: The target prefix to be provided to let the
        decoder predict next token which is optional.
        :type target_prefix: str | None.
        :param acutal_target: The actual target which was supposed to be predicted that is
                              optional.
        :type actual_target: str | None.
        :param max_tokens: Max tokens to be predicted if predict_input and target_prefix
        is provided.
        :type max_tokens: int | None.

        :returns: Returns a dictionary with train loss and test loss if test dataloader
        is provided.
        :rtype: dict.
        """
        train_loss_list = []
        test_loss_list = []
        bleu_list = []

        for epoch in tqdm(range(epochs)):
            train_loss = 0
            test_loss = 0

            # loop over the dataloader
            for _idx, batch in enumerate(tqdm(train_dataloader, leave=True)):
                (x, y_in, y_out) = batch[:3]
                # put the model to train mode.
                self.model.train()

                # move the input to respective device
                x, y_in, y_out = self.move(x, y_in, y_out, device=device)

                # calculate y_logits by performing feed-forward.
                y_logits = self.model(x, y_in)

                # calculate loss.
                loss = self.loss_fn(y_logits.flatten(0, 1), y_out.view(-1))

                # collect loss
                train_loss += loss.item()

                # empty the value of optimizer
                self.optimizer.zero_grad()

                # backprop
                loss.backward()

                # update weights
                self.optimizer.step()

                # if batch_size_to_predict is provided with all the other
                # parameters, the model will be tested to check how well it predicts.
                if batch_size_to_predict and (_idx + 1) % batch_size_to_predict == 0:
                    print("\n")
                    print("=============================")
                    print(
                        f"Saving checkpoint for epoch: {epoch + 1}, batch: {_idx + 1}"
                    )
                    self._save_checkpoint()

                    self._log_prediction(
                        predict_input=predict_input,
                        target_prefix=target_prefix,
                        actual_target=actual_target,
                        max_tokens=max_tokens,
                        # device=device,
                    )

            # calculate average training loss.
            avg_train_loss = train_loss / len(train_dataloader)

            # append the calculated loss
            train_loss_list.append(avg_train_loss)

            # first save the trained model for a particular epoch.
            print(f"Saving checkpoint for epoch: {epoch + 1}")
            self._save_checkpoint()

            # if test_dataloader is provided calculate average test loss.
            if isinstance(test_dataloader, torch.utils.data.DataLoader):
                test_dict = self.test(test_dataloader, device)
                test_loss = test_dict["test_loss"]  # get the test loss
                # get the bleu score
                bleu = (
                    test_dict["bleu"] * 100
                )  # multliply by 100 to make it a percentage

                # append test and bleu loss
                test_loss_list.append(test_loss)
                bleu_list.append(bleu)

            print(
                f"Epoch {epoch + 1} -->"
                f"Training Loss: {avg_train_loss},"
                f"Test Loss: {test_loss},"
                f"Bleu (Test data): {bleu}"
            )

            self._log_prediction(
                predict_input=predict_input,
                target_prefix=target_prefix,
                actual_target=actual_target,
                max_tokens=max_tokens,
                # device=device,
            )

        # return the result back.
        return {
            "train_loss": train_loss_list,
            "test_loss": test_loss_list,
            "bleu": bleu_list,
        }

    def test(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        device: DeviceType,
    ) -> dict:
        """
        To test the model against a dedicated test_dataloader.

        :param test_dataloader: Test dataloader to tested against
                                test data and loss is returned.
        :type test_dataloader: torch.utils.data.DataLoader.
        :param device: Device to move the tensors to.
        :type device: 'cpu' | 'cuda'.

        :returns: Returns the dictionary containing bleu score and
                  test loss
        :rtype: dict.
        """

        # put the model in eval mode.
        self.model.eval()

        test_loss = 0
        references: list = []
        candidates: list = []

        with torch.inference_mode():
            # for src, target_in, target_out, raw_source, raw_target_in, raw_target_out
            #  in test_dataloader.
            for x, y_in, y_out, _, _, unpadded_y_out in tqdm(
                test_dataloader, leave=False
            ):
                batch_size = x.shape[0]  # get the batch size.
                # move inputs to respective devices.
                x, y_in, y_out = self.move(x, y_in, y_out, device=device)

                # calculate the raw logits
                y_logits = self.model(x, y_in)

                # accumulate the test_loss
                test_loss += self.loss_fn(y_logits.flatten(0, 1), y_out.view(-1)).item()

                # get the prediction(candidate) by taking argmax in last dimension
                prediction = torch.argmax(y_logits, dim=-1)

                # add y_out to the references and
                # prediction to candidates for bleu
                # calculation. Always calculated in cpu.
                for i in range(batch_size):
                    # corpus_bleu expects list(list(list(str)))
                    references.append([unpadded_y_out[i].tolist()])
                    candidates.append(prediction[i].tolist())

        print("Calculating Bleu scores over the test data.")
        bleu_score = corpus_bleu(references, candidates)
        avg_test_loss = test_loss / len(test_dataloader)
        # return the average test loss
        return {"test_loss": avg_test_loss, "bleu": bleu_score}

    def _save_checkpoint(self):
        """
        To save the model to store the model state.
        """
        # torch.save(
        #     {
        #         "model": self.model.state_dict(),
        #         "optimizer": self.optimizer.state_dict(),
        #     },
        #     self.path,
        # )
        print("Saving Model...")
        torch.save(self.model.state_dict(), self.model_path)

        print("Saving Optimizer...")
        torch.save(self.optimizer.state_dict(), self.optimizer_path)

        print("Checkpoint Saved!!!")

    def load_checkpoint(self, device: DeviceType):
        """
        To load the state back into model.
        """
        if not self.model_path.exists() or not self.optimizer_path.exists():
            print("Warning: The path do not exist yet.")
            return
        print("Loading Model...")
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=torch.device(device=device))
        )

        print("Loading Optimizer...")
        self.optimizer.load_state_dict(
            torch.load(self.optimizer_path, map_location=torch.device(device=device))
        )

        print("Checkpoint Loaded!!!")
        # checkpoint = torch.load(self.path, map_location=torch.device(device=device))
        # self.model.load_state_dict(checkpoint["model"])
        # self.optimizer.load_state_dict(checkpoint["optimizer"])

    def _log_prediction(
        self,
        predict_input: str | None,
        target_prefix: str | None,
        actual_target: str | None,
        max_tokens: int | None,
        # device: DeviceType = "cpu",
    ):
        """
        :param predict_input: The input string of the source text to predict the
        performance of the model after every epoch which is optional.
        :type predict_input: str | None.
        :param target_prefix: The target prefix to be provided to let the
        decoder predict next token which is optional.
        :type target_prefix: str | None.
        :param acutal_target: The actual target which was supposed to be predicted that is
                              optional.
        :type actual_target: str | None.
        :param max_tokens: Max tokens to be predicted if predict_input and target_prefix
        is provided.
        :type max_tokens: int | None.
        :param device: Device to move the tensors to.
        :type device: 'cpu' | 'cuda'.
        """
        # print the prediction of the inference.
        if predict_input and target_prefix and max_tokens and actual_target:
            # start with target_prefix as output.
            result = target_prefix

            # collect the yielded result.
            for token in Predictor.predict(
                self.model,
                self.tokenizer,
                predict_input,
                target_prefix,
                max_tokens,
                self.tokenizer.get_piece_id("<|endoftext|>"),
            ):
                result += self.tokenizer.decode(token)

            print("Source Input", predict_input)
            print("Actual Target", actual_target)
            print("Predicted Target", result)
        else:
            print("All the inputs for prediction not given.")


class Predictor:
    """
    A class to perform prediction on a trained model.
    """

    @staticmethod
    def predict(
        model: Transformer,
        tokenizer: Tokenizer,
        inputs: str,
        target: str,
        max_tokens: int,  # context size.
        stop_token: int,
        # device: DeviceType = "cpu",
    ) -> Generator[torch.Tensor, Any, Any]:
        """
        Method to predict tokens for the given input and
        target.

        :param inputs: The source string to be passed to encoder.
        :type inputs: str.
        :param target: The target string to be passed to decoder.
                       Should pass the start token, like <|kannda|
                       or <|hindi|>
        :type target: str.

        :param max_tokens: Maximum tokens allowed for token generation.
        :type max_tokens: int.

        :param stop_token: Token to stop the generation before maximum tokens
                           limit is met.
        :type stop_token: int.

        :param device: Device to move the tensors to.
        :type device: 'cpu' | 'cuda'.

        :returns: A generator to yield a torch tensor for every iteration.
        :rtype: Generator[torch.Tensor, Any, Any].
        """
        # automatically move the model to respective device.
        # model.to(device)
        model.cpu()
        # put the model in eval mode.
        model.eval()
        with torch.inference_mode():
            # encode both the input and target
            x = tokenizer.encode(inputs).unsqueeze(dim=0)  # to make [b, x]
            y = tokenizer.encode(target).unsqueeze(dim=0)  # to make [b, y]
            # move the inputs to respective devices.
            # x, y, _ = self.move(x, y, device=device)

            # caculate the encoder state once for inference.
            memory = model.encode(x)

            # rest the caches
            model.reset_cache()

            # set y lengths for adding positions to key value cache.
            y_start = 0
            y_end = y.shape[1]

            # Repeat for max_token times
            for _ in range(max_tokens):

                # predict the next tokens.
                y_logits = model.decode(y, memory, y_start, y_end, inference=True)

                # take the last token as it is the predicted token.
                last_token = y_logits[:, -1, :]

                # find the index with max probability
                prediction = torch.argmax(last_token, dim=-1)

                # if the prediction is <|endoftext|>, stop yielding.
                if prediction.item() == stop_token:
                    break

                # update y to prediction as the next query and
                # match the dimension.
                y = torch.unsqueeze(prediction, dim=0)  # [b, prediction]

                # update the positions
                y_start, y_end = y_end, y_end + 1

                # y = torch.cat([y, prediction.unsqueeze(0)], dim=1)

                # yield the prediction back.
                yield prediction
