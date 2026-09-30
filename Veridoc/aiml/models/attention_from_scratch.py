"""
Self-attention and cross-attention implemented from raw matrix operations
(no nn.MultiheadAttention, no library attention call) so you can point at
every line and explain what it is doing.

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Run this file directly to embed a sentence with a pretrained tokenizer's
embedding table (or random embeddings if none available), compute the
attention matrix by hand, and plot it as a heatmap -- this is the artifact
worth showing live in a demo.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class ScratchSelfAttention(nn.Module):
    """Single-head self-attention, written out explicitly."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (seq_len, embed_dim)
        returns: (output, attention_weights)
        """
        Q = self.W_q(x)  # (seq, d)
        K = self.W_k(x)  # (seq, d)
        V = self.W_v(x)  # (seq, d)

        d_k = Q.shape[-1]
        scores = Q @ K.T / (d_k ** 0.5)          # (seq, seq) raw compatibility scores
        weights = torch.softmax(scores, dim=-1)   # (seq, seq) attention distribution per token
        output = weights @ V                       # (seq, d) weighted sum of values

        return output, weights


class ScratchCrossAttention(nn.Module):
    """Cross-attention: queries come from one sequence, keys/values from another
    (this is exactly what a decoder does when attending over encoder/retrieved context)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query_seq: torch.Tensor, context_seq: torch.Tensor):
        """
        query_seq:   (q_len, d)  e.g. the question tokens
        context_seq: (c_len, d)  e.g. the retrieved evidence tokens
        """
        Q = self.W_q(query_seq)
        K = self.W_k(context_seq)
        V = self.W_v(context_seq)

        d_k = Q.shape[-1]
        scores = Q @ K.T / (d_k ** 0.5)   # (q_len, c_len)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V
        return output, weights


def _toy_embed(tokens: list[str], dim: int = 16, seed: int = 0) -> torch.Tensor:
    """
    Deterministic random embedding per unique token, for demo purposes only.
    Swap this for real embeddings (e.g. from retrieval/embeddings.py) in production.
    """
    rng = np.random.default_rng(seed)
    vocab = {}
    vecs = []
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = rng.normal(size=dim)
        vecs.append(vocab[tok])
    return torch.tensor(np.stack(vecs), dtype=torch.float32)


def plot_attention(tokens: list[str], weights: torch.Tensor, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(weights.detach().numpy(), cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key / Value tokens")
    ax.set_ylabel("Query tokens")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved heatmap to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", default="the critic rejected the answer because evidence was weak")
    parser.add_argument("--embed_dim", type=int, default=16)
    args = parser.parse_args()

    tokens = args.sentence.lower().split()
    x = _toy_embed(tokens, dim=args.embed_dim)

    self_attn = ScratchSelfAttention(embed_dim=args.embed_dim)
    _, weights = self_attn(x)
    plot_attention(tokens, weights, "Self-Attention", "self_attention_heatmap.png")

    # cross-attention demo: treat the first half of the sentence as "query"
    # and the second half as "retrieved context"
    mid = len(tokens) // 2
    query_tokens, ctx_tokens = tokens[:mid], tokens[mid:]
    q_x, c_x = _toy_embed(query_tokens, dim=args.embed_dim), _toy_embed(ctx_tokens, dim=args.embed_dim)
    cross_attn = ScratchCrossAttention(embed_dim=args.embed_dim)
    _, cross_weights = cross_attn(q_x, c_x)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cross_weights.detach().numpy(), cmap="magma")
    ax.set_xticks(range(len(ctx_tokens)))
    ax.set_yticks(range(len(query_tokens)))
    ax.set_xticklabels(ctx_tokens, rotation=90)
    ax.set_yticklabels(query_tokens)
    ax.set_xlabel("Context tokens (keys/values)")
    ax.set_ylabel("Query tokens")
    ax.set_title("Cross-Attention")
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig("cross_attention_heatmap.png", dpi=150)
    print("Saved heatmap to cross_attention_heatmap.png")


if __name__ == "__main__":
    main()
