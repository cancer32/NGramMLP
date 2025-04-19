import os
import random
import argparse

import torch
import torch.nn as nn

from ngram_mlp import NGramMLPModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, type=str, help='Path to the dataset')
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to save the model weights')
    parser.add_argument('--batch', type=int, default=3, help='Batch size of consecutive character as input feature')
    parser.add_argument('--emb', type=int, default=2, help='Embedding size')
    parser.add_argument('--hid', type=int, default=100, help='Hidden layer perceptron size')
    parser.add_argument('--seed', type=int, default=random.randint(0, 999999), help='Random seed')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    args = parser.parse_args()

    names = open(args.dataset_path, 'r').read().splitlines()
    random.seed(args.seed)
    random.shuffle(names)

    vocab = ['.'] + sorted(set(''.join(names)))
    vocab_size = len(vocab)

    # ordinal encoder/decoder
    itos = dict(enumerate(vocab))
    stoi = dict((s, i) for i, s in itos.items())

    # Creating train, dev, test dataset
    X, Y = [], []

    for name in names:
        content = [0] * args.batch 
        for ch in name + '.':
            ch = stoi[ch]
            X.append(list(content))
            Y.append(ch)
            content.append(ch)
            content = content[1:]
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    itr = int(0.8 * X.shape[0])
    idev = int(0.9 * X.shape[0])
    x_train, x_dev, x_test = X.tensor_split((itr, idev))
    y_train, y_dev, y_test = Y.tensor_split((itr, idev))

    g = torch.Generator().manual_seed(args.seed)
    model = NGramMLPModel(
        vocab_size=vocab_size,
        batch_size=args.batch,
        embedding_size=args.emb,
        hidden_layer_size=args.hid,
        generator=g
    )
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.train()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        b = torch.randint(0, x_train.shape[0], (int(x_train.shape[0]*0.01),))
        loss = nn.functional.cross_entropy(model(x_train[b]), y_train[b])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{args.epochs}: Loss {loss}')

    print(f'Dev set loss: {nn.functional.cross_entropy(model(x_dev), y_dev)}')
    print(f'Test set loss: {nn.functional.cross_entropy(model(x_test), y_test)}')

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'batch_size': model.batch_size,
        'embedding_size': model.embedding_size,
        'hidden_layer_size': model.hidden_layer_size,
        'itos': itos
    }, args.checkpoint_path)