#!/bin/bash
# Install the P2P-patched NVIDIA open kernel modules (aikitoria fork, 615.71.09-p2p) on cradle.
#
# Run as:  sudo bash perf/p2p/install-p2p-driver.sh   (P2P_SRC=<path to the built fork> to override)
# Then reboot IMMEDIATELY. After apt upgrades the driver userspace, new GPU processes fail with
# "driver/library version mismatch" until the reboot loads the new modules.
#
# What it does:
#   1. apt: nvidia-driver-615-open (userspace 615.71.09; its kernel module is built by DKMS = stock fallback).
#      Ubuntu's DKMS creates a signing key (/var/lib/dkms/mok.key) and may ask you (debconf dialog) to set a
#      Secure Boot enrollment password — pick one and remember it.
#   2. uses that DKMS key (or creates /root/p2p-mok if there is none) and queues it for enrollment
#   3. signs the prebuilt patched modules and installs them to /lib/modules/$KVER/updates/nvidia-p2p
#      ("updates" comes before "ubuntu"/"kernel" in depmod's search order), then depmod and initramfs
#   4. adds "amd_iommu=on iommu=pt" (IOMMU passthrough, required for BAR1 P2P) to GRUB
#
# AT THE REBOOT (physical console needed once): the blue "Perform MOK management" screen appears.
#   Enroll MOK -> Continue -> Yes -> type the password from step 2 -> Reboot.
#   If you miss it (10 s timeout), the patched module is rejected; run rollback-p2p-driver.sh or redo
#   `sudo mokutil --import /root/p2p-mok/MOK.der` and reboot again.
#
# Rollback: sudo bash perf/p2p/rollback-p2p-driver.sh  (and reboot)
set -euo pipefail

SRC=${P2P_SRC:-$(cd "$(dirname "$0")" && pwd)/open-gpu-kernel-modules}  # git clone -b 615.71.09-p2p https://github.com/aikitoria/open-gpu-kernel-modules && make modules
KVER=$(uname -r)
WANT=615.71.09
DEST=/lib/modules/$KVER/updates/nvidia-p2p
MOKDIR=/root/p2p-mok
SIGN=/usr/src/linux-headers-$KVER/scripts/sign-file

[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
for m in nvidia nvidia-uvm nvidia-modeset nvidia-drm nvidia-peermem; do
  [ -f "$SRC/kernel-open/$m.ko" ] || { echo "missing $SRC/kernel-open/$m.ko — build first: (cd $SRC && make modules -j16)"; exit 1; }
done
grep -q "$WANT" "$SRC/version.mk" || { echo "fork checkout is not $WANT"; exit 1; }
[ -x "$SIGN" ] || { echo "missing $SIGN (install linux-headers-$KVER)"; exit 1; }

echo "== 1/4 driver userspace $WANT"
apt-get update -qq
apt-get install -y nvidia-driver-615-open
dpkg -l | grep -q "nvidia-kernel-common-615.*$WANT" || echo "WARNING: installed 615 userspace is not exactly $WANT — check: dpkg -l | grep nvidia"

echo "== 2/4 Secure Boot signing key"
if [ -f /var/lib/dkms/mok.key ] && [ -f /var/lib/dkms/mok.pub ]; then
  mkdir -p $MOKDIR && chmod 700 $MOKDIR
  cp /var/lib/dkms/mok.key $MOKDIR/MOK.priv && cp /var/lib/dkms/mok.pub $MOKDIR/MOK.der && chmod 600 $MOKDIR/MOK.priv
  echo "using the DKMS signing key (one key to enroll for both stock and patched modules)"
fi
if [ ! -f $MOKDIR/MOK.priv ]; then
  mkdir -p $MOKDIR && chmod 700 $MOKDIR
  openssl req -new -x509 -newkey rsa:2048 -nodes -days 36500 -subj "/CN=cradle P2P NVIDIA module key/" \
    -keyout $MOKDIR/MOK.priv -outform DER -out $MOKDIR/MOK.der
  chmod 600 $MOKDIR/MOK.priv
fi
if mokutil --test-key $MOKDIR/MOK.der 2>&1 | grep -q "is already enrolled"; then
  echo "key already enrolled"
elif mokutil --list-new 2>/dev/null | grep -qF "$(openssl x509 -inform DER -in $MOKDIR/MOK.der -noout -subject -nameopt multiline | awk -F' = ' '/commonName/{print $2}')"; then
  echo "key already queued for enrollment (by DKMS) — use the password you set in the apt dialog"
else
  echo "Choose a one-time password; you'll type it on the blue MOK screen at the next boot:"
  mokutil --import $MOKDIR/MOK.der
fi

echo "== 3/4 sign + install patched modules -> $DEST"
mkdir -p "$DEST"
for m in nvidia nvidia-uvm nvidia-modeset nvidia-drm nvidia-peermem; do
  install -m 644 "$SRC/kernel-open/$m.ko" "$DEST/$m.ko"
  "$SIGN" sha256 $MOKDIR/MOK.priv $MOKDIR/MOK.der "$DEST/$m.ko"
done
# pin our copies above the stock DKMS build (which also lives under updates/)
{ for m in nvidia nvidia_uvm nvidia_modeset nvidia_drm nvidia_peermem; do echo "override $m $KVER updates/nvidia-p2p"; done; } > /etc/depmod.d/00-nvidia-p2p.conf
depmod -a "$KVER"
modinfo -k "$KVER" nvidia | grep -E '^(filename|version|signer)'
update-initramfs -u -k "$KVER"

echo "== 4/4 IOMMU passthrough"
if ! grep -q 'iommu=pt' /etc/default/grub; then
  cp /etc/default/grub /etc/default/grub.bak-p2p
  sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 amd_iommu=on iommu=pt"/' /etc/default/grub
fi
grep '^GRUB_CMDLINE_LINUX_DEFAULT' /etc/default/grub
update-grub

echo
echo "DONE. Reboot now (sudo reboot). Watch for the blue MOK screen: Enroll MOK -> Continue -> Yes -> password."
echo "After boot: bash $(dirname "$0")/verify-p2p.sh"
