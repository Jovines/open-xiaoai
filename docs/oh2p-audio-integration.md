# OH2P 音频集成研究记录

> 设备：Xiaomi 智能音箱 Pro（OH2P）｜固件：`1.56.19`｜内核：Linux `4.9.61`｜记录日期：2026-08-25

本文保存持续录音与小爱自身播放区分的可复现实证、工程取舍和回滚边界。不得在这里加入 SSH 密码、Zeris token、真实家庭逐字稿或私人原音。

## 已验证的设备事实

- `/proc/asound/cards` 有 `AML-AXGSOUND` 和 `UAC2_Gadget` 两张声卡。
- `hw:0,3` 是 PDM capture。OH2P 的 `/etc/asound.conf` 将 `pcm.Capture` 配成 `48 kHz / S32_LE / 4 channels` 的 `dsnoop`；持续录音通过 `pcm.noop -> pcm.Capture` 并发旁路读取，不占用或替换原厂进程。
- 三个通道包含有效阵列麦克风信号；第四通道在实测中恒为零，所以长期证据保存前三路 `48 kHz / 24-bit FLAC`。不能把旧 LX06 配置里的 `mic_num=6; ref_num=1` 直接当作 OH2P 原始采集通道布局。
- 原厂 `/usr/bin/mipns-xiaomi` 同时持有 `pcmC0D3c` 与 `pcmC0D2p`，并加载 `libmdspeech.so`、`libvpm.so`。二进制包含 AEC latency 与 VPM dump 相关符号，说明原厂语音链内部有回声处理；这不等于已有可供我们归档的 AEC reference API。
- 默认播放 PCM 是 `pcm.!default -> pcm.vis -> pcm.tocopy -> pcm.Playback -> pcm.dmixer -> hw:0,2`。`pcm.vis` 的 file 插件执行 `safe_fifo /tmp/vis_audio.fifo /tmp/mis_audio.fifo`，后者被 `misound_service` 消费用于灯效。观察到的 `safe_fifo` stdin 是 ALSA file 插件创建的 pipe。
- 上述播放镜像已覆盖一次通过小爱原生 TTS 发起的受控播报；全部提示音、音乐、蓝牙等路径仍须逐项实测。部分原厂进程可能直接打开命名 PCM，不能用一次 TTS 成功推断所有播放路径都已覆盖。
- Home Assistant 中音量、静音、播放状态当前均为 `unavailable`，不能作为可靠的唯一来源归因信号。

## 为什么不注入原厂进程

- `open-xiaoai/client` 与我们的 recorder 都是独立进程。目标是最大化利用现有硬件，同时让小爱唤醒、问答、播放、蓝牙和 OTA 主链保持原样。
- 设备未暴露 BPF/uprobe tracing，且没有 `bpftool`/`perf`；4.9 内核不能使用预期的 eBPF 无重启采集方案。
- `strace -p`、ptrace/Frida 会暂停或扰动实时线程，不适合 7×24 音频。`LD_PRELOAD` 仍需要按新环境重启原进程，且会把故障带进原厂地址空间。
- ALSA 边界已经能看到 PCM，因此选择独立 fan-out，而不是 hook `mipns-xiaomi` 内存。这个边界更容易测试、审计和回滚。

## Fail-open fan-out 设计

`device/audio_fanout.c` 是 `safe_fifo` 调用位置的候选替代品：

1. stdin 始终持续排空，避免 ALSA file 插件反压播放线程。
2. 原有 `mis_audio.fifo` 仍收到未经修改的 raw PCM；它是主兼容输出。
3. `/tmp/open_xiaoai_playback.fifo` 是可选旁路。没有 reader、FIFO 满、SSH 断线或采集机停止时，只丢旁路帧，不等待、不重试阻塞。
4. reference 采用 10 ms 帧：`48 kHz / stereo / S16_LE`，每帧 1920 bytes。40-byte 小端头包含 `OXR1` magic、版本、头长度、播放进程 stream id、流内连续序号、设备 `CLOCK_REALTIME` 纳秒、payload 长度和尾帧 flag；完整包 1960 bytes，小于 Linux `PIPE_BUF`，写成功时保持原子性。stream id 防止多个播放客户端同时写 FIFO 时序号互相污染。
5. 序号用于发现丢帧，时间戳用于与三麦克风录音对齐；离线处理仍需用 reference 与麦克风回声做互相关/延迟估计，不能把 SSH 到达时间当作声学同步真值。

本机测试覆盖：旁路 reader 缺失时主输出字节完全一致；旁路存在时完整帧与尾帧的 magic、版本、stream id、序号、时间戳、长度和 flag 正确。

2026-08-25 canary 已用 `/data/open-xiaoai/audio-reference/asound.conf.canary` bind mount 到 `/etc/asound.conf`，未覆盖系统文件，只重启 `mediaplayer`，未重启 `mipns-xiaomi`。设备 `/tmp` 随机 payload 对拍：主路 3840 bytes 逐字节一致，旁路两帧 3920 bytes。随后 7 秒数字静音首次连接捕获 1.870 秒，说明 receiver 冷启动可能错过前段；连接保持后再次播放 1 秒数字静音，完整归档 1.010 秒、0 丢帧。麦克风 recorder、processor、playback recorder 三项服务全程 active。canary 尚未设为重启后持久启用。

同日 11:44 的原生 TTS 可听测试进一步验证了真实链路：reference 完整归档 2.930 秒、293 个 10 ms 包、0 丢帧，三路麦克风均捕获到一致的外放波形，处理器把同一 reference 的 NAS URI、SHA-256 和事件内偏移附到了语音证据。三路互相关峰值位置相差不超过 0.13 ms，说明阵列录音与 reference 可稳定对齐；但默认 PCM 的数字旁路约比麦克风收到的真实外放提前 1.95 秒，后续 AEC/归因必须估计并补偿播放缓冲延迟，不能只比较设备时间戳的瞬时重叠。

这次受控句也再次证明“有 reference”不等于“ASR 文本可靠”：目标句中的“原有”被 Qwen3-ASR-1.7B 识别成“仍有”。当前保留原音、模型候选、`fallible_asr` 标记和 `needs_review` 的策略是必要的。因为尚未执行播放时真人插话，处理器把麦克风与播放重叠段保守标为 `unknown + aec_not_yet_applied`，没有误报为已确认的 `overlap`。

## 构建兼容性经验

- OH2P 是 aarch64 内核上的 ARM hard-float 32-bit 用户态；产物必须是 `ELF 32-bit LSB ARM EABI5 hard-float`。
- `dockcross/linux-armv7-lts` 静态产物最低内核 4.19，`dockcross/linux-armv7` 产物最低内核 5.4；两者在设备只运行 usage 冒烟时均被 `FATAL: kernel too old` 拒绝，临时文件随后删除，未安装、未改配置。
- open-xiaoai 上游 Dockerfile 曾使用 Linaro `latest-7` URL；2026-08-25 该 URL 已重定向到普通网页，不能再作为可复现下载源。
- 当前 `scripts/build-audio-fanout.sh` 固定 Ubuntu 20.04 镜像 digest，安装 focal 的 ARM hard-float cross libc，生成最低内核 3.2 的静态 stripped 二进制。设备 usage 冒烟已通过；当前 canary 二进制 SHA-256 为 `8a700af84779b7e168a9446e04bfa768d5f8937c70d3a36481e7c26fa84888b3`。

## 角色归因与 Zeris 语义

- “转写内容”与“声音来源”是两条独立的不确定推断链。
- 场景类型：`ambient_speech | xiaoai_dialogue | device_playback | mixed | unknown`。
- turn 来源：`human | xiaoai_output | overlap | unknown`。
- 只有同步 `playback_refs[]` 已归档且原始证据可回听时，`xiaoai_output` 才有强依据。仅有播放状态、时间重叠或音色相似时必须待复核，Zeris 服务端把设备来源置信度封顶在 0.69。
- 小爱回答可用于还原人机对话上下文，但不能当作家庭成员的偏好、承诺、待办或授权。`overlap` 不能整段删除，应从 microphone array 与 playback reference 中复核近端人声。

## 后续验收与回滚门禁

生产切换前必须全部满足：

1. ARM 产物架构、最低内核和 SHA-256 有记录；usage 冒烟通过。
2. 在临时 FIFO 上做音频 payload 对拍，并验证旁路无人读取、慢读取、断开重连均不阻塞 stdin。
3. 备份当前 `/etc/asound.conf` 与文件哈希；先用 `/data` 文件 bind mount 做临时 canary，不直接覆盖系统文件。
4. 准备单命令回滚：卸载 bind mount、恢复原 `safe_fifo` 配置、仅重启为重新打开 PCM 所必需的最小音频服务。
5. canary 后实测小爱唤醒、回答、音量、媒体、蓝牙、灯效、持续三麦录音；任何一项退化立即回滚。
6. 采集一轮“人问小爱答”和一轮“播放时人插话”，检查 reference 覆盖、时钟偏移、丢帧、互相关延迟、AEC 后残余回声，再允许生成强来源标签。

## 外部依据

- [open-xiaoai Rust client：独立转发麦克风与设备事件](https://github.com/idootop/open-xiaoai/blob/main/packages/client-rust/README.md)
- [open-xiaoai stereo：ALSA 边界重定向的现有实现](https://github.com/idootop/open-xiaoai/blob/main/examples/stereo/src/utils/alsa.rs)
- [WebRTC Audio Processing：near-end `ProcessStream` 与 far-end `ProcessReverseStream`](https://webrtc.googlesource.com/src/+/refs/heads/main/api/audio/audio_processing.h)
- [SpeexDSP Echo Cancellation：同步 playback reference 与 capture 的要求](https://www.speex.org/docs/manual/speex-manual/node7.html)
