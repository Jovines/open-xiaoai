#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file=${OPEN_XIAOAI_ENV_FILE:-$HOME/.config/open-xiaoai-recorder/env}
binary="$repo_dir/device/audio_fanout.armv7"

if [[ ! -f "$env_file" ]]; then
  echo "missing environment file: $env_file" >&2
  exit 1
fi
set -a
source "$env_file"
set +a

if ! file "$binary" | grep -q 'ELF 32-bit.*ARM.*statically linked.*GNU/Linux 3.2.0'; then
  echo "refusing incompatible audio_fanout binary" >&2
  exit 1
fi

remote=${RECORDER_USER:-root}@${RECORDER_HOST:-192.168.8.242}
ssh_options=(-o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa -o StrictHostKeyChecking=no)
ssh_command=(ssh "${ssh_options[@]}")
scp_command=(scp -O "${ssh_options[@]}")
if [[ -n ${SSHPASS:-} ]]; then
  ssh_command=(sshpass -e "${ssh_command[@]}")
  scp_command=(sshpass -e "${scp_command[@]}")
else
  ssh_command+=( -o BatchMode=yes )
  scp_command+=( -o BatchMode=yes )
fi

"${ssh_command[@]}" "$remote" 'mkdir -p /data/open-xiaoai/audio-reference'
"${scp_command[@]}" "$binary" "$remote:/data/open-xiaoai/audio-reference/audio_fanout.new"
"${scp_command[@]}" "$repo_dir/device/rollback_audio_fanout.sh" "$remote:/data/open-xiaoai/audio-reference/rollback.sh.new"
"${ssh_command[@]}" "$remote" '
set -eu
root=/data/open-xiaoai/audio-reference
chmod 755 "$root/audio_fanout.new" "$root/rollback.sh.new"
mv "$root/audio_fanout.new" "$root/audio_fanout"
mv "$root/rollback.sh.new" "$root/rollback.sh"
test "$(sha256sum "$root/audio_fanout" | awk "{print \$1}")" = "'"$(sha256sum "$binary" | awk '{print $1}')"'"
cp /etc/asound.conf "$root/asound.conf.before-canary"
sed "s#safe_fifo /tmp/vis_audio.fifo /tmp/mis_audio.fifo#$root/audio_fanout /tmp/vis_audio.fifo /tmp/mis_audio.fifo /tmp/open_xiaoai_playback.fifo#g" \
  /etc/asound.conf > "$root/asound.conf.canary"
test "$(grep -c "$root/audio_fanout" "$root/asound.conf.canary")" -eq 1
if grep -q " /etc/asound.conf " /proc/mounts; then umount /etc/asound.conf; fi
mount --bind "$root/asound.conf.canary" /etc/asound.conf
/etc/init.d/mediaplayer restart >/dev/null 2>&1 || { "$root/rollback.sh"; exit 1; }
echo "audio fan-out canary staged; rollback: $root/rollback.sh"
'
