import torch
import torch.nn.functional as F


@torch.no_grad()
def predict(model, tensor, idx_to_class=None):
    """Batched inference.

    Parameters
    ----------
    model        : nn.Module  in eval() mode on device
    tensor       : torch.Tensor  (B, 3, H, W) on the same device as model
    idx_to_class : dict[int, str] or None  maps class index -> class name

    Returns
    -------
    class_indices : LongTensor (B,)        predicted class index per sample
    class_names   : list[str] (len B)      predicted class name ('' if idx_to_class is None)
    scores        : FloatTensor (B, C)     per-class softmax confidence
    """
    was_training = model.training
    model.eval()

    logits = model(tensor)
    scores = F.softmax(logits, dim=1)
    class_indices = scores.argmax(dim=1)

    if was_training:
        model.train()

    if idx_to_class is not None:
        class_names = [idx_to_class[int(i)] for i in class_indices]
    else:
        class_names = [""] * len(class_indices)

    return class_indices, class_names, scores
