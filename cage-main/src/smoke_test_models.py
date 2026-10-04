import argparse
import gc
import itertools
import sys
import torch
from dataloader import get_data, get_early_fusion_data, get_late_fusion_data
from main import cough_dir, grid, speech_dir, LR_BATCH_SIZE
from model_scripts import LateFusion, ResNet18, Logistic_Regression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--all-fold-pairs", action="store_true")
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--dev-fold",  type=int, default=1)
    parser.add_argument("--max-configs", type=int)
    return parser.parse_args()


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    return torch.device(requested)


def valid_grid_points(all_fold_pairs, test_fold, dev_fold):
    for values in itertools.product(*grid.values()):
        point = dict(zip(grid.keys(), values))
        if point["test_set"] == point["dev_set"]:
            continue
        if not all_fold_pairs and (
            point["test_set"] != test_fold or point["dev_set"] != dev_fold
        ):
            continue
        if point["arch"] == "lr" and point["fusion"] != "none":
            continue
        if point["arch"] == "lr" and point["use_pretrained"] is False:
            continue
        yield point


def build_data(point, batch_size):
    data_folds = f"data/{point['dataset']}/data_folds_filtered"
    loss = "cross_entropy" if point["arch"] == "lr" else point["loss_selected"]
    common = {
        "dataset": point["dataset"], "data_folds": data_folds,
        "i": point["test_set"], "j": point["dev_set"],
        "loss": loss, "batch_size": batch_size,
        "augmentation": point["augmentation"],
    }
    if point["fusion"] == "none":
        return get_data(cough_dir=cough_dir, **common)
    if point["fusion"] == "early":
        return get_early_fusion_data(cough_dir=cough_dir, speech_dir=speech_dir,
                                     arch=point["arch"], **common)
    return get_late_fusion_data(cough_dir=cough_dir, speech_dir=speech_dir, **common)


def build_model(point, device):
    if point["fusion"] == "late":
        return LateFusion(num_classes=2,
                          use_pretrained=point["use_pretrained"]).to(device)
    if point["arch"] == "lr":
        return Logistic_Regression(input_dim=128, num_classes=2).to(device)
    return ResNet18(num_classes=2,
                    use_pretrained=point["use_pretrained"]).to(device)


def move_batch(batch, device):
    if len(batch) == 4:
        cough, speech, labels, _ = batch
        return (cough.to(torch.float32).to(device),
                speech.to(torch.float32).to(device)), labels.long().to(device)
    inputs, labels = batch[0], batch[1]
    return (inputs.to(torch.float32).to(device),), labels.long().to(device)


def run_config(point, batch_size, device):
    train_loader, dev_loader, _ = build_data(point, batch_size)
    train_batch = next(iter(train_loader), None)
    dev_batch   = next(iter(dev_loader), None)

    model = build_model(point, device)
    if hasattr(model, "param_groups"):
        optimizer = torch.optim.AdamW(
            model.param_groups(point["learning_rate"], point["weight_decay"])
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=point["learning_rate"],
                                      weight_decay=point["weight_decay"])

    model.train()
    inputs, labels = move_batch(train_batch, device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(*inputs)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss), "non-finite training loss"
    loss.backward()
    optimizer.step()

    model.eval()
    dev_inputs, dev_labels = move_batch(dev_batch, device)
    with torch.inference_mode():
        dev_logits = model(*dev_inputs)
    assert torch.isfinite(dev_logits).all(), "NaN logits at eval"
    return float(loss.detach().cpu())


def main():
    args = parse_args()
    device = choose_device(args.device)
    points = list(valid_grid_points(args.all_fold_pairs,
                                    args.test_fold, args.dev_fold))
    if args.max_configs:
        points = points[:args.max_configs]

    passed = 0
    for index, point in enumerate(points, start=1):
        try:
            bs = LR_BATCH_SIZE if point["arch"] == "lr" else args.batch_size
            loss = run_config(point, bs, device)
            passed += 1
            print(f"PASS {index}/{len(points)} fusion={point['fusion']} "
                  f"arch={point['arch']} loss={loss:.4f}")
        except Exception as exc:
            print(f"FAIL {index}/{len(points)}: {exc}")
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\n{passed}/{len(points)} passed.")
    return 0 if passed == len(points) else 1


if __name__ == "__main__":
    sys.exit(main())