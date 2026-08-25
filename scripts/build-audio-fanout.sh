#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image="ubuntu@sha256:8feb4d8ca5354def3d8fce243717141ce31e2c428701f6682bd2fafe15388214"

docker run --rm \
  -e BUILD_UID="$(id -u)" \
  -e BUILD_GID="$(id -g)" \
  -v "$repo_dir:/work" \
  -w /work \
  "$image" \
  bash -lc '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends gcc-arm-linux-gnueabihf libc6-dev-armhf-cross >/dev/null
    arm-linux-gnueabihf-gcc -std=c11 -Os -static -s -Wall -Wextra -Werror \
      device/audio_fanout.c -o device/audio_fanout.armv7
    chown "$BUILD_UID:$BUILD_GID" device/audio_fanout.armv7
  '

file "$repo_dir/device/audio_fanout.armv7"
sha256sum "$repo_dir/device/audio_fanout.armv7"
