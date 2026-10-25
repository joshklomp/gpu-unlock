#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
# Did gpu_unlock unlock every card it was asked to, this boot?
#
# Once nvidia owns a card the selectors cannot be read back from userspace,
# so the kernel log of this boot is the record. Usage:
#   status.sh           human-readable, exit 0 only if every target is unlocked
#   status.sh --json    one JSON object (for vllmctl /status, Aqua)
#   status.sh --quiet   exit code only (ExecStartPre=- in a vllm unit)
# Reading the kernel log needs the adm or systemd-journal group (or root).
set -u
mode=${1:-text}

log=$(journalctl -k -b -o cat 2>/dev/null | grep 'gpu_unlock' || true)
[ -z "$log" ] && log=$(dmesg 2>/dev/null | grep 'gpu_unlock' || true)
cmd=$(grep -o 'gpu_unlock\.targets=[^ ]*' /proc/cmdline 2>/dev/null | cut -d= -f2-)

state_of() {
	# Last verdict line for one PCI address wins (resume lines come later).
	echo "$log" | grep -i "$1" | grep -E 'gpu_unlock: (unlocked|already unlocked|still unlocked|unlock failed|unlock after resume failed|not bound)' |
		tail -1 | sed -E 's/.*gpu_unlock: //'
}

addrs=$(lspci -Dn -d 10de: 2>/dev/null | awk '$2 ~ /^03/ {print $1}')
ok=1
json_cards=""
text=""
for a in $addrs; do
	s=$(state_of "$a")
	case "$s" in
	"unlocked"*|"already unlocked"*|"still unlocked"*) v=unlocked ;;
	"") v=untouched ;;
	*) v=locked ;;
	esac
	if [ -n "$cmd" ] && { [ "$cmd" = all ] || echo ",$cmd," | grep -qi ",$a,"; }; then
		[ "$v" = unlocked ] || ok=0
		target=true
	else
		target=false
	fi
	text="$text$a  $v${s:+  ($s)}\n"
	json_cards="$json_cards{\"pci\":\"$a\",\"state\":\"$v\",\"target\":$target,\"detail\":\"$(echo "$s" | sed 's/"/\\"/g')\"},"
done
[ -z "$cmd" ] && ok=0

case "$mode" in
--json)
	printf '{"opted_in":%s,"targets":"%s","all_unlocked":%s,"cards":[%s]}\n' \
		"$([ -n "$cmd" ] && echo true || echo false)" "$cmd" \
		"$([ $ok = 1 ] && echo true || echo false)" "${json_cards%,}" ;;
--quiet) ;;
*)
	if [ -z "$cmd" ]; then
		echo "gpu_unlock.targets= is not on the kernel command line: nothing was unlocked"
	elif [ -z "$log" ]; then
		echo "no gpu_unlock lines in this boot's kernel log (module not in initramfs, or no permission to read it)"
	fi
	printf "%b" "$text" ;;
esac
[ $ok = 1 ]
