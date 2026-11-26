from __future__ import annotations

import argparse
import os
import shutil

HARMFUL = ["Illegal_Activitiy", "HateSpeech", "Malware_Generation",
           "Physical_Harm", "EconomicHarm", "Fraud", "Sex"]
BENIGN = ["Political_Lobbying", "Legal_Opinion", "Financial_Advice",
          "Health_Consultation", "Gov_Decision"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/safety")
    ap.add_argument("--variants", nargs="*", default=["TYPO", "SD_TYPO", "Text_only"])
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    say = lambda *x: print(*x, flush=True)
    os.makedirs(a.root, exist_ok=True)

    mm = os.path.join(a.root, "mmsafetybench", "data")
    os.makedirs(mm, exist_ok=True)
    for s in HARMFUL + BENIGN:
        for v in a.variants:
            dst = os.path.join(mm, s, f"{v}.parquet")
            if os.path.exists(dst):
                continue
            try:
                p = hf_hub_download("PKU-Alignment/MM-SafetyBench", f"data/{s}/{v}.parquet",
                                    repo_type="dataset")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(p, dst)
                say(f"  mm {s}/{v} {os.path.getsize(dst) / 1e6:.0f} MB")
            except Exception as e:
                say(f"  mm {s}/{v} failed: {type(e).__name__}")

    for repo, sub in (("hkust-nlp/SafeBench", "figstep"), ("HKUST-NLP/SafeBench", "figstep")):
        d = os.path.join(a.root, sub, "data")
        if os.path.isdir(d) and os.listdir(d):
            break
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=os.path.join(a.root, sub),
                              allow_patterns=["*.parquet", "*.csv", "*.json"])
            say(f"  figstep from {repo}")
            break
        except Exception as e:
            say(f"  figstep {repo} failed: {type(e).__name__}")

    for repo in ("AIcell/MOSSBench", "MOSSBench/MOSSBench"):
        d = os.path.join(a.root, "mossbench")
        if os.path.isdir(d) and os.listdir(d):
            break
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=d)
            say(f"  mossbench from {repo}")
            break
        except Exception as e:
            say(f"  mossbench {repo} failed: {type(e).__name__}")

    tot = sum(os.path.getsize(os.path.join(r, f))
              for r, _, fs in os.walk(a.root) for f in fs)
    say(f"{a.root}: {tot / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
