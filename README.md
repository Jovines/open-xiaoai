# Open-XiaoAI Household Audio

这个仓库现在只做一件事：持续采集 Xiaomi 智能音箱 Pro（OH2P）的三麦克风阵列，筛掉无语音片段，并向家庭管家提供“原音 + 可疑但可追溯的中文转写”。旧的小智、MiGPT、唤醒词、对话和播放逻辑已经完全舍弃。

## 数据链路

1. `recorder.py` 只建立一条长期 SSH/`arecord` 音频流，以准确的样本数切成 60 秒存储块；切段时不会重新连接音箱，因此边界处没有采集空洞。
2. 原音以 `48 kHz / 3 通道 / 24-bit FLAC` 无损保存。硬件提供四路 PCM，其中第四路恒为零，因此不归档。
3. `processor.py` 分别检查三支麦克风。FSMN-VAD/SenseVoice 判断没有语音的段先隔离 72 小时，核对哈希后才删除；分类清单永久留下。若只有一路被噪声误触发、而 Qwen 主转写为空，也按 `no_reliable_speech` 进入同样的可恢复隔离。
4. 处理器会等待下一分钟落盘，再把“当前块 + 下一块”作为连续窗口运行 FSMN-VAD。话语按开始时间唯一归属：若从当前块末尾说到下一块开头，上一事件取得完整尾巴，持久化 carry 状态让下一块不重复转写；采集停止时，最后一块等待 75 秒后自动降级为单块处理。
5. 有语音的整段原音进入 `evidence/`。只把毫秒级语音区间（两端各保留 200 ms）拼接给 CPU 上的 Qwen3-ASR-1.7B；原始 60 秒三通道音频完全不裁剪。跨界事件携带所有相关 FLAC 的 NAS URI、SHA-256 和块内偏移，能够重建并校验原话。SenseVoiceSmall 从三路麦克风独立生成候选，字符一致度只用于提示风险。单段 Qwen 默认最多运行 45 秒，避免噪声触发长幻觉并卡住队列；超时原音永久保留为 `.processing_failed.json`，不向 Agent 发布错误文本。
6. sherpa-onnx 用 Pyannote segmentation + 中文 3D-Speaker 在 CPU 上给出仅在当前录音内有效的匿名说话人片段；它不会猜姓名、性别，也不会把不同录音的 `speaker-00` 当成同一个人。
7. 事件连同模型、运行设备、候选文本、原音 SHA-256 和 NAS 地址发给 Zeris。临时上报失败时保留 `.event.pending.json`，以 15 秒至 15 分钟指数退避重试；每个事件独立处理，一个坏事件不会阻塞之后的 VAD、转写或上报。永久拒绝的事件标为 `.event.rejected.json`，原音保留供人工复核。

转写永远是 `fallible_asr`（可能听错的二手证据），不是事实。涉及时间、金额、医疗、门锁、承诺等高影响内容时，Agent 必须结合上下文、回听原音或询问家庭成员，不能仅凭转写执行。

当前采集端尚未归档音箱自身的播放 PCM，因此事件会明确携带 `acoustic_scene.scene_type=unknown`、`playback_reference.available=false`，不会凭音色把小爱回答冒充成家庭成员的话。设备已经确认存在默认播放流镜像（`pcm.vis` → `/tmp/vis_audio.fifo`）；后续接入同步 `playback_refs[]` 后，Zeris 可按 `human | xiaoai_output | overlap | unknown` 形成可回听的人机对话 turn。

## 实测基线

- 10 秒中文家庭录音：Qwen3-ASR-1.7B 纯 CPU 推理约 6.3 秒，模型常驻约 13 GB RAM；GPU 不参与。
- SenseVoiceSmall-Q8 同一录音约 0.5 秒、约 300 MB RAM，适合 VAD 和异构复核。
- 10 秒三通道无损 FLAC 约 3.3 MiB，即未经筛选约 28–30 GiB/天。NAS 临时承接全部原音，处理成功后只长期保留含语音的证据段。

数字是当前 i7-12700 主机上的一次实测，不是性能承诺。N5105 小主机内存不足以舒适常驻 1.7B float32 模型，因此转写服务部署在当前主机，Zeris 仍运行在小主机。

## 分段方案依据

没有把录音文件本身改成“遇到静音才落盘”：无限延长或异常 VAD 都会让原始文件难恢复。当前采用开源流式 ASR 常见的双层结构——固定大小的可靠存储块，加上有状态的语音端点层。实现复用现有 FSMN-VAD；设计参考 [FunASR 的流式 FSMN-VAD 与 lookback 会话](https://github.com/modelscope/FunASR/blob/main/funasr/bin/realtime_ws.py) 和 [Silero VAD 的流式 `VADIterator`](https://github.com/snakers4/silero-vad)。Silero 适合作为未来异构复核，但当前不额外引入第二套运行时，避免两个 VAD 的边界规则互相冲突。

## 依赖与模型

```bash
sudo apt install ffmpeg openssh-client sshpass
```

处理器使用：

- `~/.venvs/qwen3-asr`：CPU 版 PyTorch 和 `qwen-asr`
- `~/models/huggingface`：Qwen3-ASR-1.7B
- `~/.local/opt/sensevoice/llama-funasr-sensevoice`
- `~/models/sensevoice-small/{sensevoice-small-q8.gguf,fsmn-vad.gguf}`
- `~/.venvs/sherpa-onnx` 与 `~/models/speaker-diarization/`：匿名说话人分离

## 短时测试

```bash
cd ~/vscode/open-xiaoai
SSHPASS=open-xiaoai python3 recorder.py \
  --segment-seconds 10 \
  --max-segments 1 \
  --output-dir /tmp/open-xiaoai-test
```

录音会从音箱的 `hw:0,3`/`noop` PCM 设备读取四路 S32_LE，并修正 A113 将有效 24 位样本放在 S32 低位的布局。默认输出目录为 `~/recordings/open-xiaoai`；生产配置改到 NAS 的 `inbox/`。

主要录音变量：

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `RECORDER_HOST` | `192.168.8.242` | 音箱 IP |
| `RECORDER_OUTPUT_DIR` | `~/recordings/open-xiaoai` | 临时原音目录 |
| `RECORDER_SEGMENT_SECONDS` | `60` | 临时段秒数 |
| `RECORDER_SAMPLE_RATE` | `48000` | 采样率 |
| `RECORDER_ARCHIVE_CHANNELS` | `3` | 三路有效麦克风 |
| `RECORDER_CODEC` | `flac` | 生产使用无损 FLAC |
| `RECORDER_MIN_FREE_GB` | `5` | 低磁盘余量时暂停 |
| `SSHPASS` | 无 | 音箱 SSH 密码；推荐最终换成密钥 |

处理变量见 `.env.example`。真实 SSH 密码和 `ZERIS_AUDIO_INGEST_TOKEN` 只放在权限为 `600` 的本机环境文件，不能提交。

## systemd 用户服务

```bash
mkdir -p ~/.config/open-xiaoai-recorder ~/.config/systemd/user
cp .env.example ~/.config/open-xiaoai-recorder/env
chmod 600 ~/.config/open-xiaoai-recorder/env
# 编辑 env，填入 SSHPASS 和与 Zeris 相同的随机 token
cp deploy/open-xiaoai-{recorder,processor}.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now open-xiaoai-recorder.service open-xiaoai-processor.service
```

```bash
systemctl --user status open-xiaoai-recorder.service open-xiaoai-processor.service
journalctl --user -u open-xiaoai-recorder.service -u open-xiaoai-processor.service -f
```

两个服务都要求 `/mnt/dx4600` 是真实挂载点。NAS 掉线时录音和清理均停止，不会悄悄写进系统盘。

## 当前边界

- 三路内容高度相关，但简单求平均在实测中会降低识别效果；当前选择与其他两路结果最一致的一路给主模型，同时保留三路无损原音，后续可加入真正的阵列波束形成。
- 匿名说话人分离已经接入，但跨录音声纹聚类、声纹注册和人工姓名标注尚未接入。加入后也必须把“这一段像谁”与经本人标注的身份分开，并允许人工纠错。
- 持续录音会采集房间内所有可听声音。启用前应确保可能被录到的人知情同意，并限制 NAS 权限、备份范围和 Zeris 访问权。
