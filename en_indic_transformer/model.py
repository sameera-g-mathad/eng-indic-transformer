from typing import Any, Generator, Literal, Optional
import torch
from torch import nn, optim
from en_indic_transformer import Transformer, Tokenizer

DeviceType = Literal["cpu", "cuda"]  # allowed device values.


class Trainer:
    """
    A singleton class that is used to train the model.
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
    ):
        # checks if there is an instance
        # already.
        if self._initialized:
            return
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self._initialized = True

    def move(
        self,
        x: torch.Tensor,
        y_in: torch.Tensor,
        y_out: torch.Tensor | None = None,
        device: DeviceType = "cpu",
    ) -> None:
        """
        Moves the passed tensors into their respective device
        """
        x = x.to(device)
        y_in = y_in.to(device)
        if y_out is not None:
            y_out = y_out.to(device)

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader | None,
        epochs: int,
        device: DeviceType,
        predict_input: str | None = None,
        predict_target: str | None = None,
        max_tokens: int = 100,
    ) -> dict:
        """
        To Train the model that is clearly of Transformer instance.
        Should pass both train_dataloader and test_dataloader.
        Returns a dict of train_loss and test_loss (if provided)
        over all epochs.

        Pass the predict_input and predict_target to print the prediction
        after each epoch.
        """
        train_loss_list = []
        test_loss_list = []

        for epoch in range(epochs):
            print(f"----- Epoch {epoch} -----")

            # put the model to train mode.
            self.model.train()

            train_loss = 0
            test_loss = 0

            # loop over the dataloader
            for idx, (x, y_in, y_out) in enumerate(train_dataloader):
                # move the input to respective device
                self.move(x, y_in, y_out, device=device)

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

                print(f"Batch {idx} complete")

            # calculate average training loss.
            avg_train_loss = train_loss / len(train_dataloader)

            # append the calculated loss
            train_loss_list.append(avg_train_loss)

            # if test_dataloader is provided calculate average test loss.
            if isinstance(test_dataloader, torch.utils.data.DataLoader):
                test_loss = self.test(test_dataloader, device)
                test_loss_list.append(test_loss)

            print(f"Training Loss: {avg_train_loss}, Test Loss: {test_loss}")

            # print the prediction of the inference.
            if predict_input and predict_target and max_tokens:
                # start with predict_target as output.
                result = predict_target

                # collect the yielded result.
                for token in self.predict(
                    predict_input, predict_target, max_tokens, device
                ):
                    result += self.tokenizer.decode(token)

                # print the resultant prediction.
                print(f"Predicted target: {result}")
            print("-----x-----")

        # return the result back.
        return {"train_loss": train_loss_list, "test_loss": test_loss_list}

    def test(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        device: DeviceType,
    ) -> float:
        """
        To test the model against a dedicated test_dataloader.
        """

        # put the model in eval mode.
        self.model.eval()

        test_loss = 0

        with torch.inference_mode():
            # for src, target_in, target_out in test_dataloader.
            for x, y_in, y_out in test_dataloader:

                # move inputs to respective devices.
                self.move(x, y_in, y_out, device=device)

                # calculate the raw logits
                y_logits = self.model(x, y_in)

                # accumulate the test_loss
                test_loss += self.loss_fn(y_logits.flatten(0, 1), y_out.view(-1)).item()

        # return the average test loss
        return test_loss / len(test_dataloader)

    def predict(
        self,
        inputs: str,
        target: str,
        max_tokens: int = 4096,  # context size.
        device: DeviceType = "cpu",
    ) -> Generator[torch.Tensor, Any, Any]:
        """
        Method to predict tokens for the given input and
        target.
        """
        # put the model in eval mode.
        self.model.eval()

        with torch.inference_mode():
            # encode both the input and target
            x = self.tokenizer.encode(inputs).unsqueeze(dim=0)  # to make [b, x]
            y = self.tokenizer.encode(target).unsqueeze(dim=0)  # to make [b, y]

            # caculate the encoder state once for inference.
            memory = self.model.encode(x)

            # Repeat for max_token times
            for _ in range(max_tokens):
                # move the inputs to respective devices.
                self.move(x, y, device=device)

                # predict the next tokens.
                y_logits = self.model.decode(y, memory, inference=True)

                # take the last token as it is the predicted token.
                last_token = y_logits[:, -1, :]

                # find the index with max probability
                prediction = torch.argmax(last_token, dim=-1)

                # if the prediction is <|endoftext|>, stop yielding.
                if prediction == 50256:
                    break

                # update y to prediction as the next query and
                # match the dimension.
                y = torch.unsqueeze(prediction, dim=0)  # [b, prediction]

                # yield the prediction back.
                yield prediction
