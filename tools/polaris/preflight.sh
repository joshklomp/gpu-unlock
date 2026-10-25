#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
# Read-only checks before installing gpu-unlock on an Ubuntu box with
# initramfs-tools (written for Polaris: Ubuntu 26.04, kernel 7.0, 2x RTX 3090).
# Changes nothing. Exit 0 = no blockers, 1 = at least one FAIL.
set -u

fails=0
pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; fails=$((fails + 1)); }
info() { printf 'INFO  %s\n' "$*"; }

kver=$(uname -r)

# 1. Cards. Supported GeForce Ampere IDs are in kernel/gpu_unlock/main.c;
#    Polaris only needs 10de:2204 (RTX 3090).
gpus=$(lspci -Dnn -d 10de: 2>/dev/null | grep -E '\[03(00|02)\]' || true)
if [ -z "$gpus" ]; then
	fail "no NVIDIA display/3D controller found by lspci"
else
	echo "$gpus" | while read -r line; do
		addr=${line%% *}
		id=$(echo "$line" | sed -n 's/.*\[10de:\([0-9a-f]*\)\].*/\1/p')
		vga=$(cat "/sys/bus/pci/devices/$addr/boot_vga" 2>/dev/null || echo "?")
		drv=$(basename "$(readlink "/sys/bus/pci/devices/$addr/driver" 2>/dev/null)" 2>/dev/null)
		case "$id" in
		2204) pass "$addr RTX 3090 (10de:$id) driver=${drv:-none} boot_vga=$vga" ;;
		2203|2205|2206|2208|2484|2482|2504) pass "$addr supported Ampere (10de:$id) driver=${drv:-none} boot_vga=$vga" ;;
		*) warn "$addr 10de:$id is not a target this check knows; see main.c supported_ids" ;;
		esac
	done
fi

# 2. Kernel headers for DKMS.
if [ -f "/lib/modules/$kver/build/Makefile" ]; then
	pass "kernel headers present for $kver"
else
	fail "no headers for $kver: sudo apt install linux-headers-$kver"
fi
if command -v dkms >/dev/null; then pass "dkms installed"; else fail "dkms missing: sudo apt install dkms"; fi

# 3. Secure Boot. Ubuntu's DKMS signs with the shim MOK key; an unsigned
#    module is refused at load and the card simply stays locked.
if command -v mokutil >/dev/null; then
	sb=$(mokutil --sb-state 2>/dev/null | head -1)
	case "$sb" in
	*enabled*)
		key=/var/lib/shim-signed/mok/MOK.der
		if [ -f "$key" ] && mokutil --test-key "$key" 2>/dev/null | grep -q "already enrolled"; then
			pass "Secure Boot on; DKMS MOK key enrolled ($key)"
		else
			fail "Secure Boot on and the DKMS MOK key is not enrolled: sudo update-secureboot-policy --enroll-key, then reboot and enroll"
		fi ;;
	*disabled*) pass "Secure Boot off" ;;
	*) warn "Secure Boot state unknown: $sb" ;;
	esac
else
	warn "mokutil missing; cannot check Secure Boot"
fi

# 4. Load order. initramfs-tools starts udev (init-top) BEFORE it loads
#    modules.d entries. If nvidia or nouveau is inside the initramfs, udev
#    coldplug binds the card first and gpu_unlock refuses it (-EBUSY).
initrd=/boot/initrd.img-$kver
if command -v lsinitramfs >/dev/null && [ -r "$initrd" ]; then
	inside=$(lsinitramfs "$initrd" 2>/dev/null | grep -E '/(nvidia[^/]*|nouveau)\.ko' || true)
	if [ -n "$inside" ]; then
		fail "GPU driver inside $initrd; udev would bind it before gpu_unlock runs:"
		echo "$inside" | sed 's/^/        /'
	else
		pass "no nvidia/nouveau module inside $initrd"
	fi
	have=$(lsinitramfs "$initrd" 2>/dev/null | grep -c 'gpu_unlock\.ko\|gpu-unlock/610' || true)
	[ "$have" -gt 0 ] && info "gpu_unlock already in initramfs ($have entries)"
else
	warn "cannot read $initrd (run with sudo to check initramfs contents)"
fi

# 5. /boot space. The firmware adds ~70 MB (compressed) to EVERY initramfs.
kernels=$(ls /boot/initrd.img-* 2>/dev/null | wc -l)
avail=$(df -Pm /boot 2>/dev/null | awk 'NR==2 {print $4}')
need=$((kernels * 75 + 100))
if [ -n "$avail" ]; then
	if [ "$avail" -ge "$need" ]; then
		pass "/boot has ${avail} MB free; ~${need} MB needed for $kernels initramfs images"
	else
		fail "/boot has ${avail} MB free; ~${need} MB needed for $kernels initramfs images (purge old kernels first)"
	fi
fi

# 6. Command line opt-in.
if grep -q 'gpu_unlock.targets=' /proc/cmdline; then
	info "already opted in: $(grep -o 'gpu_unlock\.targets=[^ ]*' /proc/cmdline)"
else
	info "not opted in yet; add gpu_unlock.targets=all to GRUB_CMDLINE_LINUX_DEFAULT, then sudo update-grub"
fi

# 7. Driver and power state, for the benchmark record.
if command -v nvidia-smi >/dev/null; then
	info "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
	nvidia-smi --query-gpu=index,pci.bus_id,power.limit,power.max_limit,clocks.max.sm \
		--format=csv,noheader | sed 's/^/INFO  gpu /'
fi

if [ "$fails" -gt 0 ]; then
	echo "$fails blocker(s). Fix them before make install."
	exit 1
fi
echo "No blockers. Next: baseline benchmarks, then make install (see tools/polaris/README.md)."
