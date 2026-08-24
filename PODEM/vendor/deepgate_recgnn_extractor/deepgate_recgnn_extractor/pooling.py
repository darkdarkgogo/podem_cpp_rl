import torch


def pool_graph(node_embeddings, method="mean"):
    if method == "mean":
        return node_embeddings.mean(dim=0)
    if method == "sum":
        return node_embeddings.sum(dim=0)
    if method == "max":
        return node_embeddings.max(dim=0).values
    raise ValueError(f"Unsupported graph pooling method: {method}")

