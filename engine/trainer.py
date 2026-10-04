import copy

import torch


def compute_class_weights(dataset, num_classes, device):
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    counts = torch.bincount(targets, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def binary_metrics_from_logits(logits, labels, positive_index=1):
    preds = logits.argmax(dim=1)

    tp = ((preds == positive_index) & (labels == positive_index)).sum().item()
    tn = ((preds != positive_index) & (labels != positive_index)).sum().item()
    fp = ((preds == positive_index) & (labels != positive_index)).sum().item()
    fn = ((preds != positive_index) & (labels == positive_index)).sum().item()
    return tp, tn, fp, fn


def summarize_metrics(loss_sum, samples, tp, tn, fp, fn):
    loss = loss_sum / max(samples, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_one_epoch(model, device, loader, criterion, optimizer=None, positive_index=1):
    """
    Run one training or evaluation epoch.

    Pass optimizer=None for eval mode (no gradient updates).
    Returns a dict with loss, accuracy, precision, recall, f1, tp/tn/fp/fn, cm.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    total_tp = total_tn = total_fp = total_fn = 0
    cm = torch.zeros(2, 2, dtype=torch.long)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.detach().argmax(dim=1)
        tp, tn, fp, fn = binary_metrics_from_logits(logits.detach(), labels, positive_index=positive_index)
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        # 攤平成 2x2 索引
        flat = (labels.cpu() * 2 + preds.cpu()).long()
        cm += torch.bincount(flat, minlength=4).reshape(2, 2)

    metrics = summarize_metrics(total_loss, total_samples, total_tp, total_tn, total_fp, total_fn)
    metrics["cm"] = cm.numpy()
    return metrics


def train_model(
    model,
    device,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    epochs=12,
    unfreeze_epoch=4,
    lr=1e-3,
    backbone_lr_factor=0.1,
    should_freeze_backbone=True,
    positive_index=1,
    learning_depend_metric="f1",
    epoch_callback=None,
    early_stopping=True,
    patience=6,
    min_delta=1e-4,
    checkpoint_guard=None,
):
    """
    Full training loop with optional backbone unfreeze.

    When should_freeze_backbone=True, the model's backbone should already be
    frozen and the optimizer should contain only classifier params before this
    call. At unfreeze_epoch, backbone params are unfrozen and added as a
    second param group with lr * backbone_lr_factor.

    unfreeze_epoch=None 表示全程不解凍。
    """
    best_state = None
    monitor_mode = "min" if "loss" in learning_depend_metric else "max"
    best_score = float("inf") if monitor_mode == "min" else float("-inf")
    best_cm = None
    best_epoch = None
    history = []
    patience_counter = 0

    for epoch in range(epochs):
        print(f"===== Epoch {epoch + 1}/{epochs} =====")

        if unfreeze_epoch is not None and epoch == unfreeze_epoch and should_freeze_backbone:
            print("===== Unfreezing Backbone =====")
            backbone_params = []
            for name, param in model.named_parameters():
                if "classifier" not in name and not param.requires_grad:
                    param.requires_grad = True
                    backbone_params.append(param)
            if backbone_params:
                optimizer.add_param_group({"params": backbone_params, "lr": lr * backbone_lr_factor})

        train_m = run_one_epoch(model, device, dataloaders["train"], criterion, optimizer, positive_index)
        val_m = run_one_epoch(model, device, dataloaders["val"], criterion, None, positive_index)

        scheduler.step(val_m[learning_depend_metric])

        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "train_f1": train_m["f1"],
            "val_f1": val_m["f1"],
            "train_acc": train_m["accuracy"],
            "val_acc": val_m["accuracy"],
            "train_precision": train_m["precision"],
            "val_precision": val_m["precision"],
            "train_recall": train_m["recall"],
            "val_recall": val_m["recall"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        print_metrics(train_m, val_m)

        if epoch_callback is not None:
            epoch_callback(val_m, history, epoch)

        score = val_m[learning_depend_metric]
        is_best = (score < best_score - min_delta) if monitor_mode == "min" else (score > best_score + min_delta)
        if is_best and checkpoint_guard is not None:
            is_best = checkpoint_guard(val_m, history, epoch)
        if is_best:
            best_score = score
            best_cm = val_m["cm"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    return best_state, best_score, best_cm, best_epoch, history


def print_metrics(train_m, val_m):
    header = f"{'Split':<6} | {'Loss':>6} | {'F1':>6} | {'Acc':>6} | {'Precision':>9} | {'Recall':>6}"
    print(header)
    print("-" * len(header))
    for name, m in [("Train", train_m), ("Val", val_m)]:
        print(
            f"{name:<6} | {m['loss']:6.4f} | {m['f1']:6.4f} | {m['accuracy']:6.4f} | "
            f"{m['precision']:9.4f} | {m['recall']:6.4f}"
        )
