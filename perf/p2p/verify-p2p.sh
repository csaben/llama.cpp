#!/bin/bash
# Post-reboot check for the P2P driver (no root needed).
K=$(uname -r)
echo "== module"; modinfo nvidia | grep -E '^(filename|version|signer)'; cat /proc/driver/nvidia/version | head -1
echo "== kernel cmdline has iommu=pt?"; grep -o 'iommu=pt' /proc/cmdline || echo "MISSING iommu=pt"
echo "== nvidia-smi"; nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader
echo "== P2P matrix"; nvidia-smi topo -p2p r 2>/dev/null
cd "$(dirname "$0")" && [ -x p2p-latency ] || ${NVCC:-nvcc} -O2 -arch=sm_86 -o p2p-latency p2p-latency.cu
echo "== latency (before patch: P2P 0, ~26 us round trip)"; ./p2p-latency
