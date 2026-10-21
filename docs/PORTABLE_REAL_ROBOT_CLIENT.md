# Portable W4A8 真机客户端交接

本次仅迁移 pack/stack 的 AgileX Cobot Magic 双臂客户端。新增公开入口为
`python -m fastwam_steerquant.real_robot.run`，默认 shadow，不 enable、不发动作、不移动初始姿态。
shadow 运行仍会订阅已有 ROS 图像并被动读取 CAN；本次验证未运行这些硬件读取入口。

## 来源与依赖裁剪

sparse 基准 commit：`836dd4796ae57653c22c48991ec5c9f6fd8b9c88`。
开始时 sparse 有 9 个已修改文件及若干未跟踪文件，全部保持原样；以下四个控制源文件相对该 commit 均未修改。

| 原模块（`src/qi/deployment/` 下） | 迁入 `src/fastwam_steerquant/real_robot/` | 保留内容 |
|---|---|---|
| `async_joint_runtime.py` | `runtime.py` | 单 producer、RTC 消费线程、ContinuousActionGovernor、等待/重试、Space 停止、trace/report |
| `async_rtc_buffer.py` | `buffer.py` | generation/revision、cursor 对齐、过期丢弃、指数融合、arrival smoothstep、内存 sink |
| `async25_adapters.py` | `passive_can.py`、`observations.py`、`actions.py` | PassivePiperState、LiveImages、live-only ObservationSource、执行/影子 sink、governor 构建 |
| `robot_runtime.py` | `controller.py` | 仅 dual Piper 的 SDK 连接、速度/加速度设置和 enable；未迁入旧 RTC runtime 或单臂模式 |
| 原部署入口 | `run.py`、`config.py` | W4A8 配置/身份验证、资源关闭、显式运动确认；旧 Python/Shell 入口转发到这里 |

迁移源码 SHA256：

```text
async_joint_runtime.py f267b8ea94c29baed8d73a0e97a83006f570368e7154128b3ad04166ad1da014
async_rtc_buffer.py    78915b0a7342381cc21c62ca42215b11ac52b3d1a7fd26a05e02a325b61c4e0e
async25_adapters.py    ae3a4c51b4ae68539532e5bf73244807aa3efe1ada61efb997fbb67b55e437de
robot_runtime.py      9e5646967693ad74e20d06cb2bdfd04f6176aa991348a857636f5d7e2e0b38eb
```

旧入口经 adapters 导入的 WAMRemotePolicyClient、旧 rtc_buffer、整个 robot_runtime 及
BF16/W4A4 profile 分支均不再位于公开客户端依赖链。复用现有
`W4A8Async50PolicyClient` 的 RPC、10 秒响应超时和 diagnostics 校验；未复制另一套协议。
其余依赖为标准库、NumPy、PyYAML，以及本包现有的 PyTorch 依赖。
不迁移训练、校准、模型实现、comparison、HDF5 reader 或数据；客户端不读 checkpoint、episode、缓存或统计文件。
所迁代码的 MIT 声明保存在 `real_robot/LICENSE`，并随包分发。

## 实验逻辑与公开化改动

- 固定 40 Hz、当前物理状态 14 维、动作 `[32,14]`、10 次去噪的 W4A8 Graph 服务。
  按各 YAML 的 task/SHA 校验服务，启动及每次推理都检查 40 Hz / 775 ms profile。
- 保留单同步 RPC producer、观测采样后的 cursor、generation 防旧结果、过期 chunk 丢弃、
  指数权重 0.2 和五步 smoothstep。连续推理预算仍为 `(L_prev+L_current)*40 <= 31`，
  等待动作 0.2 秒、重试新块 grace 0.6 秒，没有放宽消费预算或回放过期动作。
- 保留硬件边界投影、关节每步 0.04 rad、夹爪每步 0.01 m；最终 sink 仍检查
  mean jump 1.5、max jump 0.25、夹爪 jump 0.01 m、mean tracking error 0.3。
  各任务原统计中的 action global_min/max 仅提取为 YAML 数值，用于原有 OOD 诊断，不改变安全投影。
- ROS topics、CAN channels、初始姿态、速度、安全参数和新鲜度限制全部在两个 example YAML。
  相机名固定为 cam_high / cam_left_wrist / cam_right_wrist，输入 uint8 `[480,640,3]`。
  ROS 时间戳 age < 1 秒、skew <= 50 ms；被动 CAN 状态 age < 0.25 秒。
- 原 CAN 解码不变：关节反馈 0x2A5–0x2A7 为 milli-degree 转 rad；0x2A8 宽度为 μm 转 m，
  非宽度模式拒绝。shadow 只接收，不创建 SDK controller。
- 运动只由 CLI 显式开启，YAML 无运动开关。`--move-to-initial` 要求 `--execute-actions`；
  任何 SDK 连接/enable/初始姿态之前，交互终端必须输入完整的 `EXECUTE`。
  不提供跳过确认的参数；非交互终端拒绝执行。初始姿态插值仍为 50 步、每步 0.04 秒。
- 资源通过 ExitStack 关闭；运行器先等待 producer 退出再关闭观测/RPC。
  公开配置的 join timeout 从 16 秒改为 26 秒，覆盖 infer + diagnostics 两个 10 秒超时及两次 2 秒观测读取；
  此变化不改变在线动作等待、延迟预算或模型请求超时。
- 客户端只输出终端摘要和保留内存 trace，不自动生成 checkpoint、日志、动作文件或缓存。
  `robot-client` 安装 extra 提供 NumPy/PyYAML；ROS 与 SDK 在实际使用时才延迟导入。

### 图像通道核查

所定位工作树没有 `CvBridge/imgmsg_to_cv2` 调用；旧 `LiveImages` 调用的
`ros_image_to_rgb` 用 NumPy 解码，并对 `bgr8/bgra8` 交换通道。
本次按明确要求提供 native/passthrough 语义：`rgb8`、`bgr8`、`8UC3` 原字节顺序不变，
仅移除行 padding；mono/alpha 等需要改通道的格式拒绝。每个请求写入 `image_channel_order="native"`。
服务端 adapter 增加该声明的接受，不做转换，既有 RGB/as_stored 行为保持。
对于 RGB8 输入数值不变；对于 BGR8，新的 native 入口刻意不沿用旧辅助函数的交换。
当前没有硬件帧或历史 CvBridge 证据，不能仅凭源码/mock 宣称历史 BGR 实验链路已核实一致。

## 用户仍需提供的环境

- Linux SocketCAN，已正确配置并持续提供反馈的双臂 CAN；客户端不配置网口或启动 CAN。
- 运行中的 ROS1 master、三相机驱动及上述 topics，时间戳和消息满足 YAML；
  用户需在客户端 Python 环境中提供 rospy/sensor_msgs。native 解码不依赖 cv_bridge 二进制。
- 真正执行时需要兼容的 pyAgxArm（含 Piper/AGX gripper）和设备权限；shadow 不导入 SDK。
- 已部署并重启到本分支代码的 W4A8 Graph 模型服务，以及两端相同的 FASTWAM_AUTHKEY。
  客户端不需要模型权重，服务端依然需要原有 GPU、编译扩展、模型 handoff/预处理资产和预热观测。
  这些既有服务端资产及其机器配置未公开迁移，不能把本次客户端迁移当作完整模型服务的可移植化。

## 可复制的启动命令

以下命令均从本仓库根目录运行，属于交付说明，本次未启动服务或硬件。
使用已有兼容 Python/ROS 环境。两端分别执行环境准备，并输入同一个共享值：

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
read -rsp 'FASTWAM_AUTHKEY: ' FASTWAM_AUTHKEY
export FASTWAM_AUTHKEY
```

终端 A，服务端（复用本机原有模型配置和资产；两任务任选一个，不能同时占用同一端口）：

```bash
bash scripts/serve_joint_steerquant.sh pack3 w4a8
# stack 服务替换为：
bash scripts/serve_joint_steerquant.sh stack3 w4a8
```

等待 `PROFILE_READY`。这两个 server 包装继续使用既有 `robot_*_w4a8_async40.yaml`；
换机器部署服务须另外准备模型资产及对应 server 配置，不能把客户端 live example 当作 server 配置。
客户端可在另一台已具备 ROS/CAN 的机器运行，通过 `--host` 指定模型服务。

终端 B，默认 shadow（不 enable、不发送动作、不移动初始姿态）：

```bash
python -m fastwam_steerquant.real_robot.run \
  --config configs/robot_pack_w4a8_live.example.yaml --max-steps 128
# stack：
python -m fastwam_steerquant.real_robot.run \
  --config configs/robot_stack_w4a8_live.example.yaml --max-steps 128
```

终端 B，真实执行（命令会要求输入 `EXECUTE`，确认后才连接/enable/移动）：

```bash
python -m fastwam_steerquant.real_robot.run \
  --config configs/robot_pack_w4a8_live.example.yaml --max-steps 100000 \
  --execute-actions --move-to-initial
# stack：
python -m fastwam_steerquant.real_robot.run \
  --config configs/robot_stack_w4a8_live.example.yaml --max-steps 100000 \
  --execute-actions --move-to-initial
```

不需要初始移动时省略 `--move-to-initial`，仍必须输入 `EXECUTE`。
按 Space 请求停止，Ctrl+C 中断并关闭本客户端持有的资源。停止发送和断开连接不替代设备急停。
兼容包装为 `bash scripts/deploy_joint_steerquant.sh pack3 w4a8 [相同 CLI 参数]`；
旧 Python 入口同样转发到新模块，只接受 live example 配置，不再提供 HDF5/BF16/W4A4 客户端模式。

## 无硬件验证

151 passed，8 skipped（7 项需要可选模型源码包，1 项需要 CUDA）；未安装依赖、未编译扩展。
覆盖原异步 runtime/buffer 回归、默认 shadow、终端确认顺序、native 图像/请求、14 维状态、
输出形状、task/SHA/profile、CAN 单位/新鲜度、平滑/边界/逐步限幅和跟踪拒绝。
全部迁移模块在禁止导入硬件/私有模块的独立进程中导入，CLI help 正常；Shell 做 bash -n。
这不构成真机闭环、吞吐或 SR 验证；整个过程没有启动 ROS、相机、CAN、SDK 或机器人运动。
