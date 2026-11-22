"""Summarize full steady CUDA rounds from timeline.cpp in a server stderr log."""
import sys
from statistics import median

calls = []
graphs = []
for line in open(sys.argv[1]):
    if line.startswith("p7api "):
        _, name, start, ns = line.split()
        calls.append((int(start), int(ns), name))
    if line.startswith("p7graph "):
        _, start, ms = line.split()
        graphs.append((int(start), float(ms)))

# A >5 ms stream fence ends the verify. Keep consecutive generation fences and all four graphs.
fences = sorted(start + ns for start, ns, name in calls if ns > 5e6 and name == "cudaStreamSynchronize")
rounds = []
for left, right in zip(fences, fences[1:]):
    if not 8e6 < right - left < 25e6:
        continue
    gpu = [ms for start, ms in graphs if left <= start < right]
    if len(gpu) != 4 or max(gpu) < 5:
        continue
    waits = [ns / 1e6 for start, ns, _ in calls if left <= start < right]
    rounds.append((gpu, waits, (right - left) / 1e6))
if len(rounds) < 10:
    sys.exit("fewer than ten complete four-graph rounds")

print("Complete rounds", len(rounds))
for label, values in [
    ("round ms", [ms for _, _, ms in rounds]),
    ("GPU graphs total ms", [sum(gpu) for gpu, _, _ in rounds]),
    ("longest GPU graph ms", [max(gpu) for gpu, _, _ in rounds]),
    ("auxiliary sync ms", [sum(waits) - max(waits) for _, waits, _ in rounds]),
    ("longest sync ms", [max(waits) for _, waits, _ in rounds]),
]:
    print(f"{label}: median {median(values):.4f}, min {min(values):.4f}, max {max(values):.4f}")
for threshold in [0, 0.05, 0.1, 1.0]:
    counts = [sum(w > threshold for w in waits) for _, waits, _ in rounds]
    print(f"API waits > {threshold:g} ms: median {median(counts):g}, range {min(counts)}..{max(counts)}")
