from typing import Optional, Literal
import torch
import tiktoken
from torch import nn, optim


class Trainer:
    """
    A singleton class that is used to train the model.
    """

    _instance: Optional["Trainer"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Trainer, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        tokenizer: tiktoken.core.Encoding,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.tokenizer = tokenizer

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader | None,
        epochs: int,
        device: Literal["cpu", "cuda"],
    ) -> dict:
        """Experimental"""

        self.model.train()
        train_loss_list = []
        test_loss_list = []

        for epoch in range(epochs):
            print(f"----- Epoch {epoch} -----")

            train_loss = 0
            test_loss = 0
            for idx, (x, y) in enumerate(train_dataloader):
                x = x.to(device)
                y = y.to(device)

                y_logits = self.model(x, y)
                y_pred = torch.argmax(torch.softmax(y_logits, dim=-1), dim=-1)

                loss = self.loss_fn(y, y_pred)
                train_loss += loss.item()

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                print(f"Batch {idx} complete")

            train_loss_list.append(train_loss / len(train_dataloader))

            if isinstance(test_dataloader, torch.utils.data.DataLoader):
                test_loss = self.test(test_dataloader, device)
                test_loss_list.append(test_loss)

            print(f"Training Loss: {train_loss}, Test Loss: {test_loss}")
            print("-----x-----")

        return {"train_loss": train_loss_list, "test_loss": test_loss_list}

    def test(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        device: Literal["cpu", "cuda"],
    ) -> float:
        self.model.eval()
        test_loss = 0
        with torch.inference_mode():
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)

                y_logits = self.model(x, y)
                y_pred = torch.softmax(y_logits, dim=-1).argmax(dim=-1)

                test_loss += self.loss_fn(y_pred, y).item()

        return test_loss / len(test_dataloader)

    def predict(self, inputs: str) -> str:
        pass  # need to implement
        self.model.eval()
        return inputs
