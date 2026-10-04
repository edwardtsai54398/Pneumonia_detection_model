"""執行環境相關的工具：Colab 偵測與「這次是在什麼環境、哪一版程式碼跑的」。

這個模組存在的理由是把環境差異集中在一處。Colab 與本機真正不同的只有
幾件事（是否掛載 Drive、Kaggle 憑證來源、資料與輸出根目錄、num_workers），
全部是設定值或這裡的判斷函式，訓練邏輯本身不因環境而異。
"""

import platform
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent


def in_colab():
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _git(*args):
    """在 repo 根目錄執行 git，失敗一律回傳 None（沒裝 git、不是 repo 都算失敗）。"""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_commit():
    """目前 HEAD 的完整 SHA；拿不到就回傳 None。

    這是把一組超參數變成「可重現」的那一格資訊：光記 batch_size / lr 不夠，
    因為同樣的參數配不同版本的程式碼會跑出不同結果。要重現某次實驗就
    `git checkout <這個 SHA>`。

    注意：這個值的正確性靠「訓練前先 commit 並 push」的紀律，沒有程式在把關。
    在 Colab 隨手改了模組而沒 commit 時，HEAD 仍指著舊 commit，記下來的 SHA
    就不是實際跑的程式碼。
    """
    return _git("rev-parse", "HEAD")


def collect_run_info():
    """回傳這次執行的環境快照，直接寫進 metadata.json。"""
    cuda_available = torch.cuda.is_available()
    return {
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": _torchvision_version(),
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        # 重跑不保證結果一致
        "deterministic": False,
        "in_colab": in_colab(),
        "platform": f"{platform.system()} {platform.release()}",
    }


def _torchvision_version():
    try:
        import torchvision

        return torchvision.__version__
    except ImportError:
        return None


if __name__ == "__main__":
    import json

    json.dump(collect_run_info(), sys.stdout, indent=2, ensure_ascii=False)
    print()
