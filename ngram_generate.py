import random
import argparse

import torch
import torch.nn as nn

from ngram_mlp import NGramMLPModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to save the model weights')
    parser.add_argument('--seed', type=int, default=random.randint(0, 999999), help='Random seed')
    parser.add_argument('--count', type=int, default=10, help='Number of words to generate')
    parser.add_argument('--start', type=str, default=None, help='Starting string')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path, weights_only=True)
    itos = checkpoint['itos']
    stoi = dict((s, i) for i, s in itos.items())
    g = torch.Generator().manual_seed(args.seed)

    model = NGramMLPModel(
        vocab_size=checkpoint['vocab_size'],
        embedding_size=checkpoint['embedding_size'],
        batch_size=checkpoint['batch_size'],
        hidden_layer_size=checkpoint['hidden_layer_size'],
        generator=g
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        for i in range(args.count):
            out = '.' * model.batch_size
            if args.start:
                out = '.' * max((model.batch_size-len(args.start), 0)) + args.start
            out = [stoi[i] for i in out]
            idx = out[-model.batch_size:]
            while True:
                probs = model(torch.tensor(idx)).softmax(dim=0)
                pred = torch.multinomial(probs, 1, replacement=True, generator=g).item()
                idx.append(pred)
                out.append(pred)
                idx = idx[1:]
                if pred == 0:
                    break
            print(''.join(itos[i] for i in out if i))