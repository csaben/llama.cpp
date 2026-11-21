# cradle P2P driver (GPU↔GPU over PCIe BAR1)

Why: the two 3090s have no peer access today (`nvidia-smi topo -m` = PHB, cudaDeviceCanAccessPeer = 0;
a 9 KB exchange costs ~26 µs through host memory). Inkling 2 Mini tensor-parallel across both GPUs needs ~96
small all-reduces per token, so it only pays off with P2P (target: a few µs each).

Patch: aikitoria/open-gpu-kernel-modules branch `615.71.09-p2p` (supports RTX 3090 via BAR1). Build it next to these
scripts: `git clone -b 615.71.09-p2p https://github.com/aikitoria/open-gpu-kernel-modules && (cd open-gpu-kernel-modules && make modules -j)`.
Must match driver userspace exactly → Ubuntu `nvidia-driver-615-open` = 615.71.09 (upgrade from 580.173.02).

Preconditions on the reference machine (checked 2026-11-21): open kernel modules ✅, Resizable BAR ✅ (BAR1 32 GB per GPU),
IOMMU on (→ script adds `amd_iommu=on iommu=pt`), Secure Boot ON (→ modules are signed with a MOK; one
enrollment at the physical console on the first reboot).

## Reboot day
1. Make sure no long GPU job is mid-run (driver userspace changes under running work).
2. `sudo bash perf/p2p/install-p2p-driver.sh` — apt may show a dialog asking for a Secure Boot password; pick one.
3. `sudo reboot` right away. On the blue **MOK management** screen: Enroll MOK → Continue → Yes → password → Reboot.
4. `bash perf/p2p/verify-p2p.sh`: module version 615.71.09 from `updates/nvidia-p2p`,
   `iommu=pt` present, P2P matrix OK, latency test, services healthy.

## Rollback
`sudo bash perf/p2p/rollback-p2p-driver.sh && sudo reboot` → stock 615 DKMS module, IOMMU flags removed.
Back to 580: `sudo apt install nvidia-driver-580-open` and reboot.

## Notes / caveats
- `iommu=pt` removes DMA isolation for passthrough devices (fine for a home server without untrusted PCIe devices).
- After a kernel upgrade the patched modules only exist for the old kernel; the new kernel gets the stock DKMS
  module (works, no P2P). Rebuild: `cd open-gpu-kernel-modules && make modules -j16`, then re-run the install
  script's step 3 (or the whole script).
- Some boards need ACS override for P2P across root ports; if `verify-p2p.sh` still shows P2P disabled, that is
  the next thing to check.
