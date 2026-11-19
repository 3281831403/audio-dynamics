# audio-dynamics

## 题面

给直播加压缩和限幅后，人声忽大忽小还带爆音：包络检测把 attack 和 release 用同一个时间常数，快起慢放的压缩器变成对称；压缩曲线没有软膝，阈值附近出现折角；ratio=1 时仍加 makeup 增益，直通被改电平；限幅不做前瞻，瞬态仍会过冲；增益平滑用固定系数，与 time_ms 无关；biquad 归一化漏除 a0，低通在低频处增益不为 1；biquad 系数每次重新计算，相位在块边界跳变；apply_biquad 不保留状态，块边界出现爆音；多声道共用一份状态，左右声道互相串扰；批量处理遇 NaN 只跳过该声道，输出里留下 NaN。dsp/dynamics.py 是核心，tests/test_dynamics.py 15个红。要修的是：…

## 运行方式

在仓库根目录运行：

```powershell
cd D:\gsb-20261119\audio-dynamics
python -m pytest tests -q
```

初始失败环境必须恰好是：

```text
15 failed
```

输出尾部应为 `15 failed in ...s`，且没有 collection error、import error 或 syntax error。只需要 Python 3 和 pytest；全部离线，无外部依赖、无随机数、无 sleep。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 验收基线

`tests/test_dynamics.py` 是验收基线，不得修改、删除或弱化其中任何测试。实现只改 `dsp/dynamics.py`，保持 8 个公开函数的签名、dB/ms 契约和 `DspError` 语义；所有时间常数单位为毫秒，默认绝对容差为 `1e-6`。
