import torch


def frobenius_inner_product(a, b):
    return torch.trace(
            torch.matmul(a, torch.transpose(b))
    )
