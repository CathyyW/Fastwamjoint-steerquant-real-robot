# BF16 25 Hz / W4A8 40 Hz 异步运行配置

后续更新：pack 原始 step_015000 已下载到 `/home/agilex/.cache/modelscope/hub/models/Mikuz12/joint_pack/step_015000.pt`，SHA256 与本页记录的期望值一致；配置已设置该默认路径。本页下文“缺权重/未测试”描述保留为当时验证记录，最新启动方法见 REAL_ROBOT_LAUNCH_COMMANDS.md。

后续按用户请求整理的手动真机执行启动命令见 [REAL_ROBOT_LAUNCH_COMMANDS.md](REAL_ROBOT_LAUNCH_COMMANDS.md)。本文的原无动作入口保持不变；新增命令未由助手执行。

2026-09-24：按用户新要求新增部署方案；之前 50 Hz / 双 Graph 的比较配置和结果保留。本方案的频率及 Graph 模式不同，不能将耗时比解释为纯量化收益。未执行机器人动作或 SR 实验。

| 模式 | 模型推理 | 客户端控制频率 | 31 步延迟窗口 |
|---|---|---:|---:|
| bf16 | 对应任务原始 step_015000，BF16 eager，不安装 CUDA Graph | 25 Hz | 1240 ms |
| w4a8 | packed INT4 / A8，tile64 + Graph，fuse_block=False，fused_modules=0 | 40 Hz | 775 ms |

频率由 runtime.rospy_rate 传给 AsyncJointRTCRuntimeConfig.control_hz。原运行器仍检查 `(上一轮延迟 + 当前轮延迟) × control_hz <= 32 - 1`，每轮包含 infer 与 diagnostics RPC。频率指动作消费频率，非模型请求频率；shadow 客户端按相同节拍消费动作，但不发送机器人命令。原队列、观测时间戳、过期处理、单 producer、动作 shaping、0.2 秒等待超时、0.6 秒 fresh grace 均保留。

平滑仍为 5 步，因此现在分别覆盖 200 ms / 125 ms；关节与夹爪限制仍按原每步阈值执行，没有换算或放大阈值。改变频率也改变每秒动作变化上限和每步物理时间；模型的 32 步 horizon 没有重采样。部署前需要验证这种时序变化对任务表现的影响。

## 文件与模式选择

- 新配置 `configs/robot_{pack,stack}_bf16_async25.yaml`、`configs/robot_{pack,stack}_w4a8_async40.yaml`。
- `scripts/serve_robot_profile.py` 读取 deployment_profile 选择实际模型。BF16 调用 BF16RobotPolicy(cuda_graph=False)；量化模式沿用 PackedRobotPolicy 和原 runtime_profile，直接加载 packed deployment。
- BF16 配置里的 `wam.ckpt` 是量化部署元数据/架构/资产契约的来源；**不是 BF16 模型权重来源**。BF16 模型权重只来自 deployment_profile.baseline_checkpoint 或指定环境变量，严格校验源 SHA、step=15000、dtype、原始权重。
- `robot_comparison.py` 的 Graph 新增可选开关，默认仍为 True，历史公平比较不变。
- `robot_profiles.py` 提供配置和客户端身份检查。新服务 diagnostics 返回 control_hz / latency_window_ms，客户端拒绝频率不匹配。旧 50 Hz 配置仍走原模式。
- Q 内 `deploy_real_w4a8_async50.py` 和 `mock_w4a8_async50.py` 支持上述配置，名称保留以兼容旧命令。实机入口的 parse_args / prepare_action_sink / save_results 安全逻辑保留。
- 所有修改均在 Fastwam-steerquant；异步工作区仅只读复用。

## 资产阻塞

stack 原始 baseline：
`/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/checkpoint/stack/step_015000.pt`

SHA256：`a86b2223d4b0f1fcb079532a5a2898a5ab9f529da3143994721cdaa32d29fc9f`。

pack 原始 baseline 本机未找到。需在启动 pack BF16 服务及 live-client 的终端设置 `PACK_STEP15K_CKPT` 为真实路径；期望 SHA256：
`2b5da8ac40a2701f65f1f72cd613713047271deafbd2da929b3ed8ab75b17a49`。

缺失时立即报错；不使用 step_030000 替代。两任务 W4A8 路径、哈希及 profile 延用已核实配置，checkpoint 不改写。

## 两终端启动：默认无硬件客户端

每个终端先执行以下准备；authkey 使用相同本机值，不写入配置/命令历史：

```bash
cd /home/agilex/World_Action_Model/physical_WM/src/Fastwam-steerquant
conda activate fastwam
read -rsp 'FASTWAM_AUTHKEY: ' FASTWAM_AUTHKEY; echo
export FASTWAM_AUTHKEY
```

四组选一，一次仅启动一个模型；均使用 127.0.0.1:18765，切换方案前停止上一服务。

| 任务与模式 | 终端 1：服务端 | 终端 2：HDF 模拟客户端（128 步） |
|---|---|---|
| pack BF16 25 Hz | `bash bash/run_robot_profile.sh server pack bf16` | `bash bash/run_robot_profile.sh client pack bf16` |
| stack BF16 25 Hz | `bash bash/run_robot_profile.sh server stack bf16` | `bash bash/run_robot_profile.sh client stack bf16` |
| pack W4A8 40 Hz | `bash bash/run_robot_profile.sh server pack w4a8` | `bash bash/run_robot_profile.sh client pack w4a8` |
| stack W4A8 40 Hz | `bash bash/run_robot_profile.sh server stack w4a8` | `bash bash/run_robot_profile.sh client stack w4a8` |

pack BF16 服务端前置（路径为占位，不能原样运行）：

```bash
export PACK_STEP15K_CKPT='/实际路径/pack/step_015000.pt'
```

`client` 只读本地记录 HDF，拦截硬件模块导入，不打开 CAN、不连接 ROS，不执行动作。--max-steps 是动作消费步数，不是去噪步数；可在客户端命令后附加 `--max-steps 256`。

若需要读取现有相机和机器人状态做实时 **shadow 推理**，可将第二终端的 `client` 改为 `live-client`。此入口会读取 CAN/ROS，依赖相机和 ROS 已就绪，但包装强制 `--no-execute-actions --no-move-to-initial`，不启动驱动、不发送动作。本次不运行该入口。pack BF16 live-client 同样要设置 PACK_STEP15K_CKPT。

```bash
# 仅示例：stack W4A8 实时观测 shadow 客户端，无动作
bash bash/run_robot_profile.sh live-client stack w4a8
```

本次未提供执行动作命令；正式动作及 SR 仍需用户明确确认。

## 自动无硬件验证

包装脚本导出 PYTHONDONTWRITEBYTECODE=1，并设置 Q 源码路径。服务端启动审计写入 `outputs/profile_*_startup.json`；客户端日志为独立 `outputs/profile_*/`。W4A8 在开放监听前验证 native kernel、tile64、Graph/eager 输出及两个不同观测。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/validate_robot_profile.py --task stack --mode bf16 --port 18767
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/validate_robot_profile.py --task stack --mode w4a8 --port 18767
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/validate_robot_profile.py --task pack --mode w4a8 --port 18767
```

自动验证独立生成仅存于子进程环境的 authkey，串行运行新服务进程与模拟客户端，结束后关闭自建服务。结果放在 `outputs/robot_profile_validation/`；不能拿这些 smoke 结果当正式 SR 或公平 speedup。

## 本机实际验证结果

CPU 配置/协议/安全门测试：25 passed；shell 语法及 diff 检查通过。三组 GPU 测试顺序运行，不同时驻留模型，不启动 ROS/CAN/机器人。以下为 128 步 smoke，不是正式性能或 SR 测试。

| 任务/模式 | 控制频率 | Graph | 完成步数 | 状态 | 原始日志 |
|---|---:|---|---:|---|---|
| stack bf16 | 25 Hz | False | 128 | max_steps, 无错误 | `outputs/robot_profile_validation/stack_bf16_20260924_181343_011452` |
| stack w4a8 | 40 Hz | True | 128 | max_steps, 无错误 | `outputs/robot_profile_validation/stack_w4a8_20260924_181450_810009` |
| pack w4a8 | 40 Hz | True | 128 | max_steps, 无错误 | `outputs/robot_profile_validation/pack_w4a8_20260924_181547_078438` |

BF16 eager 的 replay=0；两组 W4A8 每次 action 的 replay=10，启动审计均确认 600 个目标层、6000 次 native kernel、tile64、fused_modules=0，两个不同观测的 Graph/eager 对照通过。
pack BF16：未测试，缺对应 step_015000；缺失拒绝逻辑已做 CPU 验证。所有配置均未做真实动作或 SR 验证。
