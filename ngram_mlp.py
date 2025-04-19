import torch
import torch.nn as nn


class NGramMLPModel(nn.Module):
    def __init__(self, vocab_size, batch_size, embedding_size, hidden_layer_size=100, generator=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_layer_size = hidden_layer_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.model = nn.Sequential(
            nn.Linear((batch_size * embedding_size), hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, vocab_size)
        )

        # Initialize weights
        for layer in self.model:
            if not isinstance(layer, nn.Linear):
                continue
            torch.nn.init.normal_(
                layer.weight,
                mean=-1,
                std=1,
                generator=generator
            )

        torch.nn.init.normal_(
            self.embedding.weight,
            mean=-1,
            std=1,
            generator=generator
        )

    def forward(self, X):
        emb = self.embedding(X).view(-1)
        return self.model(emb)