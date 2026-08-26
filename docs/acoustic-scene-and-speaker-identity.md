# 家庭声学场景与长期说话人识别方案

> 状态：2026-08-26 已以 `candidate` 模式进入生产 Evidence；不会单独触发设备动作或输出姓名事实。本文不记录真实家庭逐字稿、姓名、声纹或原音。

## 目标

持续录音不能只输出一整段文本。每段 Evidence 还应回答四个彼此独立的问题：

1. 这段声音主要是现场人声、设备外放、人与设备重叠，还是无法判断？
2. 如果有人说话，哪些时间区间属于同一个匿名说话人？
3. 这个匿名说话人是否稳定匹配到跨录音档案；该档案是否经过人工命名？
4. 转写里是否包含值得家庭管家低风险跟进的补货、快递、日程或异常候选？

任何一层不确定都必须原样传递，不能让后层模型把候选改写成事实。

## 当前能力与缺口

- 原始证据是 `48 kHz / 3 通道 / 24-bit FLAC`；小爱自身部分播放路径另有同步 PCM reference。
- 三路 FSMN-VAD 先形成至少两麦支持的语音区间；GCC-PHAT 与参考通道保护型加权融合生成一条单声道，Qwen3-ASR-1.7B 在 CPU 上只转写一次。
- sherpa-onnx 已用 Pyannote segmentation + 中文 3D-Speaker 做单文件匿名 diarization。
- `recording-speaker-00` 仍只表示单份录音内的聚类；处理器现在额外提取质量门控后的 3D-Speaker embedding，并在本机私有文件中维护跨录音匿名 profile。姓名必须由家庭所有者人工确认，且当前缺少可靠回放分数时不会向 Zeris 暴露姓名。
- 小爱播放继续以数字 PCM reference 为最高优先级。无 reference 时新增 CED-mini int8 粗标签、三麦几何无关的电平/时差特征，以及语言和重复模式融合；“手机语言学习外放”只作为概率候选，不能据此声称具体应用就是多邻国。
- Qwen3-ASR 支持 context/hotwords，但家庭原音实测表明上下文可能改变不清晰词而不一定恢复原话。生产保持无提示单次结果；热词和其他模型只允许进入离线评测，禁止静默覆盖原转写。

## 推荐流水线

```text
三麦原音 + 小爱播放 reference
  -> 三麦 VAD 共识 / 逐段 GCC-PHAT / 参考通道保护型加权融合
  -> 单条增强音频 / Qwen 单次 ASR
  -> 粗粒度 AudioSet 标签 + 阵列空间特征 + 回放检测
  -> 概率化声学场景
  -> 人声区间 diarization
  -> 质量门控后的说话人 embedding 与长期匿名档案
  -> 可追溯语义候选（补货、快递、日程、异常）
  -> 风险分级：记录候选 / 低打扰确认 / 交叉验证 / 禁止动作
```

“手机外放”和“真人说话”不能依赖单一模型。最终场景分数应融合：

- 小爱数字播放 reference 和互相关延迟；
- 通用声音标签，如 speech、conversation、television、music、notification、typing；
- 回放/反欺骗分数；
- 三麦通道间相干性、到达时间差和声源方向稳定性；
- ASR 的语言、轮次、重复模式和设备提示音；
- 将来可选的手机前台应用或媒体播放状态，但它只能作为经授权的旁证。

## 开源模型选型

| 任务 | 候选 | 选择与边界 |
|---|---|---|
| 通用声音标签 | YAMNet、EfficientAT、BEATs、CED、Zipformer Audio Tagging | 生产首选 sherpa-onnx 的 `CED-mini int8` 做 CPU 粗标签：当前运行时已经具备接口、部署最小。YAMNet 可作异构评测；BEATs 更重；标签只能说明“像电视/语音/音乐”，不能单独证明来自手机。 |
| 开放词汇场景 | LAION-CLAP / Microsoft CLAP | 适合离线探索“语言学习外放”“短视频外放”等家庭自定义标签，并生成 embedding；常驻 CPU 成本和域偏移不适合直接作为唯一生产分类器。可用于帮助构建标注集。 |
| 回放检测 | AASIST/AASIST-L、ASVspoof Physical Access 基线 | 只作真人近场与扬声器回放的一个分数。公开模型主要面向反欺骗数据，不可直接宣称能识别真实家庭中的手机、电视或小爱；必须用本房间、本麦克风和多种音量/距离重新校准。 |
| 单文件 diarization | Pyannote community、3D-Speaker、WeSpeaker、sherpa-onnx | 继续使用 sherpa-onnx Pyannote segmentation + 3D-Speaker，CPU 和现有代码最匹配。Pyannote community-1 可离线作为质量上限对照，但不先放入常驻处理链。 |
| 长期说话人识别 | 3D-Speaker CAM++/ERes2Net、WeSpeaker | 当前中文 ERes2Net 可先作为基线；补测更轻的 CAM++ ONNX。只在非回放、非重叠、足够长且信噪比合格的人声上更新档案。 |
| 阵列增强/定位 | GCC-PHAT、MVDR、SRP-PHAT、ODAS | 生产先用不依赖几何的 GCC-PHAT 时延对齐与加权融合。等权融合会改写中文短词，因此固定全句参考麦克风并至少保留 70% 权重。MVDR/ODAS 只有在拿到或标定 OH2P 三麦几何、且家庭中文 CER 明确改善后才可替换。方向是旁证，不等于设备类型。 |

主要上游：

- sherpa-onnx：<https://github.com/k2-fsa/sherpa-onnx>
- 3D-Speaker：<https://github.com/modelscope/3D-Speaker>
- WeSpeaker：<https://github.com/wenet-e2e/wespeaker>
- Pyannote audio：<https://github.com/pyannote/pyannote-audio>
- EfficientAT：<https://github.com/fschmid56/EfficientAT>
- YAMNet：<https://www.tensorflow.org/hub/tutorials/yamnet>
- LAION-CLAP：<https://github.com/LAION-AI/CLAP>
- AASIST：<https://github.com/clovaai/aasist>
- ODAS：<https://github.com/introlab/odas>
- Qwen3-ASR：<https://github.com/QwenLM/Qwen3-ASR>

## 事件数据结构

现有单一 `scene_type` 应保留为兼容投影，新的权威结果使用候选分布和信号来源：

```json
{
  "scene": {
    "primary": "live_conversation",
    "candidates": [
      {"label": "live_conversation", "probability": 0.71},
      {"label": "phone_or_computer_media_playback", "probability": 0.21},
      {"label": "unknown", "probability": 0.08}
    ],
    "signals": {
      "xiaoai_playback_reference": false,
      "audio_tags": [],
      "replay_score": null,
      "source_directions": [],
      "authorized_device_context": []
    },
    "model_versions": []
  },
  "turns": [
    {
      "start_seconds": 0.0,
      "end_seconds": 2.8,
      "origin": "human",
      "local_speaker_id": "recording-speaker-00",
      "speaker_profile_id": "household-speaker-anonymous-0001",
      "speaker_similarity": 0.82,
      "identity_label": null,
      "identity_status": "anonymous_candidate"
    }
  ]
}
```

`speaker_profile_id` 是匿名档案，不是身份事实。只有用户听过多份代表片段并明确命名后，才能增加 `identity_label`；任何一次低分匹配都必须回退匿名。系统不需要以“男女”作为身份主键，避免把音高、年龄或设备回放误当作性别和人物。

## 多邻国与手机外放

通用 AudioSet 模型没有稳定的“多邻国”类别。可落地方案是家庭域的小模型：

1. 先用 CED/EfficientAT 或 CLAP 提取 embedding。
2. 从真实房间录制并人工标注少量样本：现场中文对话、现场英文朗读、手机多邻国外放、其他手机视频、电视/电脑、小爱输出、人与设备重叠、噪声。
3. 只训练轻量分类头，按录音日期和说话人分割训练/验证集，防止同一声音泄漏造成虚高准确率。
4. 融合多邻国常见的短英语提示、重复练习轮次、反馈提示音和手机侧授权状态。
5. 输出概率，低于阈值时保持 `unknown_device_playback`，不强行猜应用名。

场景标签建议先控制在：

- `live_conversation`
- `live_monologue_or_reading`
- `xiaoai_dialogue`
- `xiaoai_playback`
- `phone_language_learning_playback`
- `phone_or_computer_media_playback`
- `television_or_remote_media`
- `mixed_live_and_playback`
- `non_speech_household_sound`
- `unknown`

## 长期说话人档案

每个合格人声 turn 提取 embedding，并保存下列内容：模型与版本、原音哈希及区间、场景分数、SNR/削波/时长质量、匿名 profile、相似度和人工标注状态。不要保存只有一个 centroid 而无法追溯代表片段。

更新门槛：

- 人声有效时长建议至少 1.5 秒，注册姓名时累计至少 20--30 秒、覆盖多天和不同位置；
- 排除小爱 reference 重叠、高回放概率、多人重叠、低 SNR 和严重削波；
- 单次高相似只允许“像某匿名档案”，不能自动命名；
- centroid 使用质量加权缓慢更新，并保留最近代表 embedding 以检测档案污染；
- 用户纠错后支持拆分、合并、撤销姓名和重建档案。

## 从转写到家庭认知

认知层不能只有“可信事实”和“全部丢弃”两个选择。需要明确的候选类型：

- `shopping_shortage_candidate`
- `delivery_eta_candidate`
- `pickup_confirmation_candidate`
- `schedule_candidate`
- `device_command_candidate`
- `household_preference_candidate`

策略：

- 可逆、低打扰的确认问题可以使用较低门槛，但措辞必须询问而非断言，并受去重和每日打扰预算限制。
- 补货候选可以问“是否需要加入购物清单”，不能直接下单。
- 快递物品较可信而日期不确定时，可以在宽松到货窗口后问“是否已经取件”，不能声称已送达或未取。
- 日程、金额、身份、承诺、门锁和设备动作继续要求更高置信或外部回执。
- 同主题 ASR 复核必须真正合并为一个 case；不得为每批录音创建一个新的“并入既有任务”任务。

## 评测与上线门禁

Git 只提交无隐私的 manifest 模板、哈希和汇总指标；音频与逐字标签留在 NAS 受限目录。

至少建立以下指标：

- 场景：macro-F1、每类 precision/recall、unknown 拒识率、概率校准误差；
- 设备回放：真人误判为回放率、回放误判为真人率，按设备/音量/距离/房间位置切片；
- diarization：DER、重叠语音 DER、说话人数量误差；
- 长期档案：已知人 top-1、未知人拒识、错误合并率、错误拆分率；
- 认知：补货/快递候选召回、错误提醒率、重复提醒率、事实越权率；
- 性能：CPU RTF、峰值 RAM、每小时耗时和对 ASR 队列的影响。

上线顺序：

1. `shadow`：只写机器报告，不进入 Zeris。
2. `candidate`：进入 Evidence，但所有新字段 `needs_review=true`，不触发动作。
3. `low-risk-confirmation`：只允许去重后的购物/取件确认。
4. `identity-assisted`：经过人工命名和未知人拒识评测后，允许使用姓名候选。
5. 高风险动作仍必须依赖设备/外部回执，不由声学模型单独授权。

当前处于第 2 阶段。第 3 阶段只对 Zeris 已去重的补货/取件候选开放了“被动确认”：它不会自行打断用户，不会自动下单，也不会把到货时间写成事实。

## 2026-08-26 家庭实测

- CED-mini int8 在 CPU 上对 60 秒三麦 FLAC 采用 10 秒窗口聚合，推理约 0.23 秒；直接把 60 秒送入单个 CED stream 会越过模型位置张量，已通过固定窗口规避。
- 一条已知语言学习外放录音得到 `phone_language_learning_playback=0.4801`；相邻的一条普通中文现场对话得到 `live_conversation=0.5606`。两者都保留完整候选分布和 `needs_review=true`，这只是烟雾测试，不是准确率声明。
- 相邻两份现场录音中，主说话人跨录音 embedding 余弦相似度为 `0.8480`，其他有效组合最高为 `0.5802`。生产匿名 profile 阈值据此保持 `0.82`；正式身份上线仍要求多天、多位置、已知/未知人评测。
- AASIST 官方模型主要针对 ASVspoof 的逻辑访问/合成语音，不直接等价于家庭里的手机或电视外放检测；因此没有把未经本房间校准的反欺骗模型硬接进姓名门禁。
