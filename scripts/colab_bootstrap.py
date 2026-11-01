"""Bootstrap a fresh Colab runtime for this repo (Phase 3+ GPU sessions).

Runs INSIDE the runtime via the Colab CLI:

    colab new -s rq --gpu T4
    colab exec -s rq -f scripts/colab_bootstrap.py          # then:
    echo 'cd rq-replication && uv run python -m rq_probe extract --max-examples 200' | colab exec -s rq

Clones the repo, installs uv, syncs the locked environment, runs the fast
smoke test, and prints the next command. Repo URL: argv[1] or RQ_REPO_URL.
"""

import os
import subprocess
import sys

REPO_URL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RQ_REPO_URL", "")


def run(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


if not REPO_URL:
    sys.exit("usage: python colab_bootstrap.py <repo-url>  (or set RQ_REPO_URL)")

run(f"git clone {REPO_URL}")
os.chdir("rq-replication")
run("pip install -q uv")
run("uv sync")
run("uv run python scripts/colab_smoke_test.py")
print("\nbootstrap complete. Next: uv run python -m rq_probe extract --max-examples 200  (pilot)")
