# Open-XiaoAI Household Audio

这个仓库现在只做一件事：持续采集 Xiaomi 智能音箱 Pro（OH2P）的三麦克风阵列，筛掉无语音片段，并向家庭管家提供“原音 + 可疑但可追溯的中文转写”。旧的小智、MiGPT、唤醒词、对话和播放逻辑已经完全舍弃。

## 数据链路

1. `recorder.py` 只建立一条长期 SSH/`arecord` 音频流，以准确的样本数切成 60 秒存储块；切段时不会重新连接音箱，因此边界处没有采集空洞。
2. 原音以 `48 kHz / 3 通道 / 24-bit FLAC` 无损保存。硬件提供四路 PCM，其中第四路恒为零，因此不归档。A113 低位样本使用 `×96` 映射，在受控外放中相对旧 `×256` 将削波减少约 560 倍，同时保留约 8.5 dB 头部空间。
3. `processor.py` 用 FSMN-VAD 分别检查三支麦克风，至少两路同时支持才进入语音链路。没有语音或只有一路被风扇等噪声误触发的存储块先隔离 72 小时，核对哈希后才删除；分类清单永久留下。
4. 处理器会等待下一分钟落盘，再把“当前块 + 下一块”作为连续窗口运行 FSMN-VAD。话语按开始时间唯一归属：若从当前块末尾说到下一块开头，上一事件取得完整尾巴，持久化 carry 状态让下一块不重复转写；采集停止时，最后一块等待 75 秒后自动降级为单块处理。
5. 有语音的整段原音进入 `evidence/`。三个方向麦克风使用完全相同的毫秒级语音边界，各自由 Qwen3-ASR-1.7B 转写一次；采集层不挑赢家、不投票，也不要求三路逐字一致。三份文本、各自真正送入 ASR 的 `16 kHz / mono / lossless FLAC`、通道编号和原始三通道录音一起交给 Zeris，由主脑结合家庭上下文理解。GCC-PHAT 加权融合仍生成可审计的阵列分析派生音频，并继续作为离线评测候选，但不在生产链路替主脑裁决语义。跨界事件携带所有相关 FLAC 的 NAS URI、SHA-256 和块内偏移，能够重建并校验原话。Qwen 的 CPU 看门狗采用动态预算：至少 180 秒，长语音按“30 秒 + 语音时长 × 3”放宽，最多 300 秒；某一路超时不会抹掉其他方向的有效观察，三路全部不可用时原音永久保留为 `.processing_failed.json`。
6. sherpa-onnx 用 Pyannote segmentation + 中文 3D-Speaker 在 CPU 上给出仅在当前录音内有效的匿名说话人片段；它不会猜姓名、性别，也不会把不同录音的 `speaker-00` 当成同一个人。
7. 事件连同模型、运行设备、阵列质量、原音与增强音频的 SHA-256/NAS 地址发给 Zeris。临时上报失败时保留 `.event.pending.json`，以 15 秒至 15 分钟指数退避重试；每个事件独立处理，一个坏事件不会阻塞之后的 VAD、转写或上报。永久拒绝的事件标为 `.event.rejected.json`，原音保留供人工复核。

转写永远是 `fallible_asr`（可能听错的二手证据），不是事实。涉及时间、金额、医疗、门锁、承诺等高影响内容时，Agent 必须结合上下文、回听原音或询问家庭成员，不能仅凭转写执行。

播放 reference canary 已从默认 PCM 的 `pcm.vis` 旁路无损归档到 NAS；处理器会把与麦克风事件重叠的 `playback_refs[]` 交给 Zeris，并按 `human | xiaoai_output | unknown` 形成保守 turn。受控“小爱问答 + 播放时插话”验收完成前，重叠近端声音不会冒充已确认的 `overlap`；没有 reference 的事件仍明确标为 `acoustic_scene.scene_type=unknown`、`playback_reference.available=false`。

OH2P 声卡拓扑、原厂进程边界、播放 fan-out 协议、内核兼容踩坑、验收与回滚门禁统一记录在 [`docs/oh2p-audio-integration.md`](docs/oh2p-audio-integration.md)。经验文档是后续实现的依据，不能只留在聊天记录里。

声学场景、手机/多邻国外放、长期匿名声纹、人工命名和低风险家庭认知候选的模型选型与上线门禁，见 [`docs/acoustic-scene-and-speaker-identity.md`](docs/acoustic-scene-and-speaker-identity.md)。

中文 ASR 的可复现清单、micro-CER、关键短语、稳定性、CPU RTF 与削波评测见 [`evaluation/README.md`](evaluation/README.md)。音频与机器报告保存在 NAS，Git 只保存无隐私标签、哈希和运行代码。

## 实测基线

- 10 秒中文家庭录音：Qwen3-ASR-1.7B 纯 CPU 推理约 6.3 秒，模型常驻约 13 GB RAM；GPU 不参与。
- 三麦 VAD、GCC-PHAT 对齐和加权融合均为 CPU 信号处理；生产对三个方向各调用一次 Qwen。受控阵列集的单路与增强输入都达到 CER 0，当前小样本没有证明融合优于最佳方向麦克风。
- 10 秒三通道无损 FLAC 约 3.3 MiB，即未经筛选约 28–30 GiB/天。NAS 临时承接全部原音，处理成功后只长期保留含语音的证据段。

数字是当前 i7-12700 主机上的一次实测，不是性能承诺。N5105 小主机内存不足以舒适常驻 1.7B float32 模型，因此转写服务部署在当前主机，Zeris 仍运行在小主机。

## 分段方案依据

没有把录音文件本身改成“遇到静音才落盘”：无限延长或异常 VAD 都会让原始文件难恢复。当前采用开源流式 ASR 常见的双层结构——固定大小的可靠存储块，加上有状态的语音端点层。实现复用现有 FSMN-VAD；设计参考 [FunASR 的流式 FSMN-VAD 与 lookback 会话](https://github.com/modelscope/FunASR/blob/main/funasr/bin/realtime_ws.py) 和 [Silero VAD 的流式 `VADIterator`](https://github.com/snakers4/silero-vad)。Silero 适合作为未来异构复核，但当前不额外引入第二套运行时，避免两个 VAD 的边界规则互相冲突。

## 历史重转写

模型或提示词升级后，先把历史原始 FLAC 以硬链接放进独立 replay 根目录，用 `processor.py --once` 生成新版证据；不要覆盖生产 `evidence/`。确认全部 `.event.pending.json` 生成后，再用独立 audit ID 幂等发布：

```bash
python3 scripts/publish_historical_replay.py \
  --evidence-dir /mnt/dx4600/家庭管家/录音/replay/<audit-id>/evidence \
  --audit-id <audit-id>
```

脚本会派生稳定的 `audio-replay-*` 事件 ID，并写入 `provenance.historical_replay`；原事件、原音和旧转写均不覆盖。网络中断后可重复执行，Zeris 按事件 ID 去重。发布完成的清单改名为 `.event.replay.json`。随后由 Zeris 的历史重审入口把新版语音与同一 `occurred_at` 时间窗内的设备状态组合成只读 Episode，允许重新形成认知和低风险候选，但不会重放家电动作、规则或补采。

## 依赖与模型

```bash
sudo apt install ffmpeg openssh-client sshpass
```

处理器使用：

- `~/.venvs/qwen3-asr`：CPU 版 PyTorch 和 `qwen-asr`
- `~/models/huggingface`：Qwen3-ASR-1.7B
- `~/.local/opt/sensevoice/llama-funasr-vad` 与 `~/models/sensevoice-small/fsmn-vad.gguf`：只做三麦语音区间检测，不做转写
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
| `RECORDER_PCM_GAIN` | `96` | A113 低位样本增益；相对完整映射保留约 8.5 dB 防削波余量 |
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

- 等权平均会改写短词：真实样本中曾把“小爱”变成“小雅”。当前使用全句固定参考通道、70% 锚定权重和逐语音段 GCC-PHAT 对齐；4 条人工真值阵列样本的 Qwen micro-CER 与关键短语错误均为 0。样本仍很小，新增家庭人工标注必须持续进入回归集。
- 匿名说话人分离已经接入，但跨录音声纹聚类、声纹注册和人工姓名标注尚未接入。加入后也必须把“这一段像谁”与经本人标注的身份分开，并允许人工纠错。
- 持续录音会采集房间内所有可听声音。启用前应确保可能被录到的人知情同意，并限制 NAS 权限、备份范围和 Zeris 访问权。
