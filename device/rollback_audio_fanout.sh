#!/bin/sh
set -eu

if grep -q ' /etc/asound.conf ' /proc/mounts; then
    umount /etc/asound.conf
fi
/etc/init.d/mediaplayer restart >/dev/null 2>&1 || true
echo "audio fan-out canary rolled back"
