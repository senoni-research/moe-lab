#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt


def read_rows(path: str):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except Exception:
        return x


def main(path: str = "sweep.csv"):
    rows = read_rows(path)
    if not rows:
        print("No rows in", path)
        return

    x = [to_float(r.get("dispatch_buf_mb", 0.0)) for r in rows]
    y = [to_float(r.get("drop_rate", 0.0)) for r in rows]
    ks = [str(r.get("k", "?")) for r in rows]

    plt.figure()
    for xv, yv, kv in zip(x, y, ks):
        plt.scatter(xv, yv, label=f"k={kv}")
    plt.xlabel("Dispatch buffer (MB)")
    plt.ylabel("Drop rate")
    plt.title("Drop rate vs buffer size")
    # de-duplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys())
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


