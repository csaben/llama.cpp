#!/bin/bash
# Undo install-p2p-driver.sh: drop the patched modules and the IOMMU flags. the stock DKMS 615 module
# (built by apt) takes over. To go all the way back to 580: apt install nvidia-driver-580-open.
# Run: sudo bash perf/p2p/rollback-p2p-driver.sh && sudo reboot
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
KVER=$(uname -r)
rm -rf /lib/modules/$KVER/updates/nvidia-p2p /etc/depmod.d/00-nvidia-p2p.conf
depmod -a "$KVER"
update-initramfs -u -k "$KVER"
if [ -f /etc/default/grub.bak-p2p ]; then cp /etc/default/grub.bak-p2p /etc/default/grub; update-grub; fi
modinfo -k "$KVER" nvidia | grep -E '^(filename|version)'
echo "Rolled back. Reboot now."
