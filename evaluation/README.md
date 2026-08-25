# 中文 ASR 评测

评测把模型能力与采音质量分开：同一句话应同时保留数字播放 reference 和房间麦克风回录。生产家庭语音必须人工校对后才能加入参考文本；未经校对的 ASR 输出不能反过来充当答案。

## 指标

- `micro_cer`：按整个集合汇总编辑距离后计算字符错误率，文本先做 NFKC、去标点/空格/模型标签和大写归一化。
- `critical_term_accuracy`：时间、金额、否定词、姓名、动作等任务关键短语的精确命中率；它比总体 CER 更接近家庭 Agent 的风险。
- `unstable_cases`：同一模型重复运行得到多个归一化结果的样本数。
- `rtf`：推理秒数 / 音频秒数，小于 1 才能持续追上实时输入；模型加载时间单独记录。
- `audio_quality`：每条输入的峰值、RMS、满幅削波与接近削波比例。

报告同时在 `summary.*.by_tag` 中按 `clean`、`room_microphone`、`noise`、`overlap` 等标签分层汇总。模型只能在目标场景的分层指标上晋级，不能用干净公开集的平均分掩盖房间场景退化。

## 清单格式

清单使用 JSON Lines，每行至少包含 `id`、相对 `audio` 和人工 `reference`；可选 `critical_terms`、`tags`、`channel`、`start_seconds`、`duration_seconds`。音频不提交 Git，只提交无隐私的标签和清单。当前受控样本放在 NAS 的 `录音/evaluation/controlled/`。

```bash
set -a
source ~/.config/open-xiaoai-recorder/env
set +a
CUDA_VISIBLE_DEVICES='' ~/.venvs/qwen3-asr/bin/python -m evaluation.asr_benchmark \
  --manifest evaluation/controlled.jsonl \
  --audio-root /mnt/dx4600/家庭管家/录音/evaluation \
  --engine qwen --engine sensevoice --engine funasr-nano --repeat 3 \
  --output /mnt/dx4600/家庭管家/录音/evaluation/results/controlled.json
```

FireRed 运行时峰值约 9.6 GB 内存，宜单独跑，避免与 Qwen 同一评测进程叠加：

```bash
CUDA_VISIBLE_DEVICES='' ~/.venvs/qwen3-asr/bin/python -m evaluation.asr_benchmark \
  --manifest evaluation/controlled.jsonl \
  --audio-root /mnt/dx4600/家庭管家/录音/evaluation \
  --engine firered \
  --output /mnt/dx4600/家庭管家/录音/evaluation/results/controlled-firered.json
```

公开普通话集合用于测模型基础能力，真实房间集合用于测距离、风扇、混响、多人重叠和小爱外放。二者必须分别报告，不能把公开集低 CER 当作家庭场景已经可靠。可按照 [Qwen3-ASR 官方评测](https://github.com/QwenLM/Qwen3-ASR) 使用 WenetSpeech/AISHELL/Fleurs，也可导入 [SenseVoice 官方 CPU benchmark](https://github.com/FunAudioLLM/SenseVoice/blob/main/runtime/llama.cpp/BENCHMARKS.md) 的人工标注样本；导入时转换为上述 JSONL，不修改原始音频和标注。

模型进入生产主路径前至少满足：真实家庭集关键短语准确率不低于当前版本，micro-CER 不倒退，CPU RTF 小于 1，并且任何模型分歧继续以多个候选和 `fallible_asr` 上报，不能静默选择更“通顺”的文本。

## 当前受控基线

2026-08-25 的首个基线包含同一句小爱 TTS 的数字 reference，以及 `×256 / ×128 / ×96` 三次真实房间回录。三模型同轮单次报告保存在 NAS `controlled-three-models.json`：Qwen3-ASR-1.7B CPU FP32 的整体 micro-CER 为 5.56%、关键短语准确率 50%、RTF 0.70；SenseVoiceSmall-Q8 为 8.33%、87.5%、0.07；Fun-ASR-Nano-2512 Q8 为 8.33%、87.5%、0.19。SenseVoice 和 Fun-ASR-Nano 的三个房间样本 CER 都为 0，但两者共享同系编码器且没有在本集合提供额外纠错价值，因此 Fun-ASR-Nano 暂不进入生产常驻路径。

FireRedASR2-AED FP32 CPU 单独批测四条时整体 micro-CER 为 1.39%、关键短语准确率 87.5%、推理 RTF 0.72；三个房间样本全部正确，数字 reference 把“原有”错成“人有”。它对生产同形态的 6.67 秒 VAD 单声道样本冷启动 5.9 秒、推理 4.9 秒、峰值内存约 9.6 GB。生产据此采用“强冲突时隔离进程按需仲裁”，而不是把第三个大模型常驻。GLM-ASR-Nano 等候选继续留在扩展评测池；在现有四条小样本不足以证明增益前，不再引入另一套大运行时。样本太少，以上结果只证明当前分工合理，不代表通用模型排名。

增益对照中，房间语音满幅削波比例从 `×256` 的 1.581973% 降至 `×128` 的 0.062687%，最终 `×96` 为 0.002811%；模型的房间样本 CER 未因降增益退化。生产因此采用 `×96`。初始机器报告保存在 NAS `录音/evaluation/results/controlled-gain96.json`。
