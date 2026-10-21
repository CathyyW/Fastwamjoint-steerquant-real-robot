# 真机 W4A8 / async50 接入记录（2026-09-24）

**当前状态：直接 packed 加载、原生 kernel、离线 forward 和 RPC 接线已验证；持续 50 Hz 验证失败，尚不可据此执行真机任务。** 两任务都因连续推理延迟超出原运行时预算停止。没有降低频率、缩短去噪、改同步、关闭限幅或放宽超时。没有启动 ROS、相机、CAN、机器人或 move-to-initial。

## 1. 修改范围与职责

全部新增/修改仅位于 `/home/agilex/World_Action_Model/physical_WM/src/Fastwam-steerquant`（下文 Q）。基线 revision `2c3abecd00ad8c2a768abc50fb95df54bad78e67`，main；本次未提交。

只读依赖 `/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/.worktrees/joint-asi-async50`（下文 A），revision `836dd4796ae57653c22c48991ec5c9f6fd8b9c88`，branch `deploy/joint-asi-async50`。**A 未修改，也未移动/删除 worktree。** A 负责现有观测读取、单生产线程、chunk 队列、50 Hz 消费、安全 governor 和机器人执行组件。两个目录及适用上级未找到 AGENTS.md。

保留已有改动：A 的 `scripts/serve_joint_asi_async50.sh` 和未跟踪 assembly 配置/文档/检查器；Q 的未跟踪 `--dataset`、`--include`、`--local_dir`、旧交接报告及 `checkpoint/`。本次未覆盖这些文件。

| Q 中新增/修改 | 用途 |
| --- | --- |
| `src/fastwam_steerquant/robot_async_server.py` | 单连接串行 RPC；启动前核验部署/资产/预处理，CPU 结构构建并直接载入 packed 权重，完整原生 kernel 审计和严格 live Graph 预热 |
| `src/fastwam_steerquant/robot_async_client.py` | 原 wire contract 的 W4A8 客户端；验证任务、部署 SHA、物理动作坐标、kernel 证明及每请求去噪/Graph 次数 |
| `src/fastwam_steerquant/kernels/w4a8/loader.py` | 增加显式 `FASTWAM_W4A8_EXTENSION`：直接导入现有指定 `.so`，ABI/符号不符即失败，不编译、不尝试 W4A4 |
| `configs/robot_{pack,stack}_w4a8_async50.yaml` | 两份独立量化配置；运行时、机器人限幅、相机/topic 字段逐项与 A 原配置相等；authkey 为环境变量占位符 |
| `scripts/deploy_real_w4a8_async50.py` | 从 A 实际异步入口派生的新入口；复用 A 全部控制组件，仅切换量化协议检查并标记 recorded/live 颜色来源。解析动作开关、初始化顺序、结果记录函数 AST 与原版相同。**本次未运行这个硬件入口。** |
| `scripts/mock_w4a8_async50.py` | 仅 HDF + 原 runtime/governor + ShadowActionSink；拒绝导入 rospy/sensor_msgs/pyAgxArm/piper_sdk，无硬件选项 |
| `scripts/validate_w4a8_async50.py` | 自动启动隔离本机服务和上述 mock，随机凭证仅存在子进程环境，记录日志并清理自己启动的服务 |
| `tests/test_robot_async_bridge.py` | 身份/精度/坐标/回退拒绝，reset/错误协议、动作不二次处理、原动作开关及初始化逻辑保持的测试 |
| `local/robot_handoff/` | 从此前本机临时交接目录复制的完整小型、哈希固定源码/配置/stats/text 缓存闭包。原文件未改；此目录被 gitignore 排除，部署时必须保留 |

原 BF16 服务/配置继续由 A 管理，没有覆盖。量化服务使用交接包中的校准版 dense 模型源码；客户端使用 A 的运行时。**两个 qi namespace 位于不同进程，不能在服务进程中混合导入 A 的 qi。**

## 2. 文件身份及模型契约

两个 checkpoint 目录的 `sha256sum -c SHA256SUMS` 均成功；metadata 用 `torch.load(map_location='cpu', weights_only=True, mmap=True)` 读取。两份均为 `fastwam_steerquant_packed_deployment_v1`、W4A8、rotation=none、600 sites、10 calls。

| 项目 | pack | stack |
| --- | --- | --- |
| deployment SHA256 | `4e3b7ff923623117d394d267243e1bb62f3027f5920ad5413754129e8dcc95b8` | `833a84d13ca4d80c1c6e6cdd1e7b41ce4b53380aa01c7c8c794c73ba142e1912` |
| metadata 源 ckpt | `joint_pack/step_015000.pt` | `joint_stack/step_015000.pt` |
| metadata 源 ckpt SHA256 | `2b5da8ac40a2701f65f1f72cd613713047271deafbd2da929b3ed8ab75b17a49` | `a86b2223d4b0f1fcb079532a5a2898a5ab9f529da3143994721cdaa32d29fc9f` |
| A 当前 BF16 配置 | `checkpoint/pack_object/step_030000.pt` | `checkpoint/stack/step_015000.pt` |
| 当前 BF16 实测 SHA256 | `29ba8afef1c7eacd0c70e5f9f6184ac6a6952cc1fd0c3f9a31cd08be0d78bb09` | 与上面源 ckpt SHA 相同 |
| dataset_stats SHA256 | `30561edb2897b9784fbf13f4e42ca9897e141b4ebd60119d047d255758aaa6bf` | `68c662dba501b1e110e09896238bfd0ae1d15c226e2c05683958bba915f7e0cb` |
| text cache SHA256 | `927573ff154e4303f06a3a719167c198b0b4ed5ea1bd72b23768217fd8381960` | `a83dfdf3a215e8314206bbf5080565617c33e23622ad05a18dd3227afa5fddf6` |

**pack 的源模型冲突真实存在。** 当前 pack_object 目录未找到 step_015000；本次确认的是 deployment 来源声明及其文件哈希，不能把当前 step_030000 BF16 当作同源量化误差基线。新配置明确保存 `baseline_checkpoint` 与部署来源，未替换旧 BF16 checkpoint。

当前两个 infer YAML 的 VAE 均指向：
`/home/agilex/World_Action_Model/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`。
流式实测 SHA256 `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`，与两份 metadata 一致。部署文件已含 VAE state，量化启动不再次读取这个 VAE 文件，也不读取原 BF16 ckpt、ActionDiT 预训练文件或文本编码器权重。

`local/robot_handoff/SHA256SUMS` 的 SHA256 为 `a429d5d2c263e988d31d20181a5621210c973c14feaf5489d700a12af9ffc359`。Adapter 在导入前验证完整 manifest 和其所有文件，拒绝额外 Python 源码；启动还核验当前 A 的统计/text 哈希和架构/processor。

两任务共同参数（从 metadata 和实际 YAML 比对）：video 30 层、hidden3072/FFN14336；action 30 层、hidden1024/FFN4096；24 heads、head_dim128；action/state14；VAE38，latent channels48；text `[128,4096]`，加 proprio 后129 tokens。相机为 high、left_wrist、right_wrist。视频9帧→3潜帧，模型图像384×320→latent24×20→每潜帧120 tokens；action32→两个16-token流。10步、CFG1、`sigma_shift=None` 使用训练配置内 video/action infer_shift5；seed42，视频/动作各自 CPU generator。无 expert_cache、无 K56、无 action prefix/gt_action 条件。

## 3. 输入输出与异步语义

输入沿用 `{images: {cam_high,cam_left_wrist,cam_right_wrist}, state, prompt}`；state 是物理坐标 float32 `[14]`，左右各6绝对关节角(rad)+夹爪宽(m)。LiveImages 依据 ROS encoding 输出 RGB；新服务不交换通道。录制 HDF 测试显式标记 `as_stored`；历史 HDF RGB/BGR 仍未证实。

三图 uint8 `[480,640,3]` → 各自 float32 CHW `/255`，resize240×320 → 高图256×320，左右各128×160放下排 → `[1,3,384,320]`、`2*x-1`、BF16。纯函数及其 processor/normalizer/resize/merger 依赖与 A 当前源码逐项比对通过。

proprio：服务端调用原 processor 的全局 z-score 并 clip[-5,5]；模型输出归一化动作，服务端原 `denormalize_actions` 用训练统计恢复物理坐标，转换 CPU float32 `[32,14]`。RPC 客户端和队列不再反归一化。stats 的实际完整数值保留在 handoff 和原 stats 文件中；没有重估统计、调整 clipping 或改颜色约定。

prompt 来自各配置，必须与部署缓存匹配：pack `Put the three objects on the table into the container.`；stack `Stack the three cups into a small tower.`。只用当前帧，不追加历史窗口；录制数据按50Hz的 control index 读取并在尾帧饱和，live 使用各相机最新且满足新鲜度/偏差检查的图像与当前状态。

协议继续使用可信本机 `multiprocessing.connection`（pickle transport）：`infer/update_observation/get_action/reset/diagnostics/shutdown`，成功 `ok:true` + `action`（NumPy float32），错误 `ok:false/error/traceback`。没有引入另一个网络协议或动作格式。模型仅串行服务一个连接；QuantizedPolicy 非重入锁、Graph `_active` 检查及每步状态检查保持有效。断开连接或 reset 清除 observation；Graph 仅保留固定形状缓冲区，每次复制新输入，随机种子仍按请求初始化。

异步运行时完全复用 A：

- 一个生产线程做同步 RPC，消费线程按50Hz/20ms取动作。RPC后 `diagnostics` 时间仍计入推理延迟，10秒响应超时不变。
- 观测前记录 probe cursor，采样后重新读取当前 cursor，再关联 inference ID。`_runtime_timing.camera_timestamps_s` 从 observation 中剥离并记录在本地 trace；原协议不把它发送给模型。本地 monotonic/wall 时钟、采样起止、RPC起止、arrival_control_t 均保留。录制 HDF 不伪造 live ROS timestamp。
- chunk始终32×14；执行 `action[t-cursor]`，按原指数权重融合重叠 chunk、到达后5步 smoothstep。generation 防旧结果写入，过期结果丢弃；没有固定每次执行32步，也没有模型上次动作续采样条件。
- 生产者更新前后延迟窗口必须满足 `(L_prev+L_current)*50 <= 31`（保留1步headroom）。第一份chunk准备好前消费不启动。动作匮乏、生产异常按原逻辑停止；不重放过期动作。
- governor 与最终 sink 安全检查保持：关节每步0.04 rad、夹爪0.01 m；任务stats extrema做OOD诊断，硬件界限+margin做投影，重试/等待机制不变。
- `--max-steps` 是控制消费者最多迭代次数，非去噪步数。真正硬件入口仍有原显式动作开关和 move-to-initial 先于Enter的顺序；**无动作验证使用 mock，完全不经过该入口**。

K56 稀疏模型与 dense W4A8 的执行路径不同。量化模型按校准拓扑运行；不能把 K56 标记伪造为 true，也不能把 K56 token裁剪/缓存套入当前 gamma stream schedule。若要二者结合，需要单独确认量化校准和稀疏路径的兼容性。

## 4. 已执行验证与失败证据

环境：`/home/agilex/miniconda3/envs/fastwam/bin/python`，Python3.10、Torch2.7.1+cu128、CUDA12.8，driver580.95.05，RTX4090 SM8.9。未安装依赖、未编译 kernel。普通沙箱不可访问驱动，实际 GPU 验证在获准 GPU 访问的工具调用中执行。

使用现有扩展：
`/home/agilex/.cache/torch_extensions/py310_cu128/fastwam_steerquant_cutlass_w4a8_sm89_shape_v3/fastwam_steerquant_cutlass_w4a8_sm89_shape_v3.so`。
启动显式指定该文件，不依赖 JIT 构建。目标 Linear 保持 packed uint8、scale/gain/clipping 保持 FP32；没有全局 `.half()`/`.bfloat16()`，没有先加载 BF16 checkpoint 到 GPU。

| 验证 | pack | stack |
| --- | --- | --- |
| 目录 SHA256SUMS | 通过 | 通过 |
| 配置/预处理/资产匹配 | 通过；源ckpt与当前BF16不同 | 通过 |
| 完整离线 `[32,14]` finite | 通过 | 通过 |
| 600 sites × 10步 native调用 | 6000 | 6000 |
| profiler A8 producer / W4A8 GEMM | 6000 / 6000 | 6000 / 6000 |
| Graph每步strict验证及最终动作/eager对照 | 通过，rtol/atol1e-3 | 通过，rtol/atol1e-3 |
| 每请求Graph replay | 10 | 10 |
| 峰值 allocated / reserved | 4,993,892,864 / 6,270,484,480 bytes | 相同 |
| 服务预热后单次forward（非统计基准） | 388.8ms | 390.5ms |
| 128步无动作异步测试 | 33个shadow步后停止 | 33个shadow步后停止 |
| 两次在线推理延迟预算 | (392.9+395.1)ms，39.40步 >31 | (396+395)ms，39.55步 >31 |
| 入队chunk / starvation / stale丢弃 | 2 / 1 / 0 | 2 / 1 / 0 |
| shadow动作周期均值 | 20.128ms | 20.124ms |

不能把33步的短暂输出称为持续50Hz通过。两次mock退出码均1，原因均为原 `_validate_latency_window` 拒绝；两次测试服务退出码均0。无硬件动作，所有动作只保存在数组/trace。

完整证据：

- `outputs/robot_async50_validation/pack_20260924_160840_433172/`
- `outputs/robot_async50_validation/stack_20260924_160945_614458/`
- 每目录含 `server_audit.json`（完整metadata关联说明/kernel证明）、`server.log`、`mock.log`、`mock/report.json`、观测/chunk/动作trace及 `status.json`。
- `outputs/robot_async50_validation/summary.json`、`source_audit.json`、`source_checkpoint_hashes.json`。
- 最早的 `pack_offline.json/stack_offline.json` 是独立离线检查；其中 CUTLASS总数还包含非量化VAE/attention kernel。以随后 server_audit 中严格筛选的6000个W4A8 GEMM为最终证明。

测试批次：协议/策略/真实adapter和原runtime/buffer/shaping/ASI测试 `59 passed, 6 skipped`（CPU批次的CUDA相关跳过）；新增安全保持测试+原async25 adapter测试 `29 passed`；实际GPU执行 `test_w4a8.py test_rollout_graph.py test_policy.py` 共15项全部通过，含变更输入后的Graph输出、W4A8数值以及调用状态/重入拒绝。批次有重叠，不应直接相加作为独立测试数量。

## 5. 可复现命令

所有命令工作目录均为 Q。服务与 mock 只需现有 fastwam 环境，不需要 source ROS/camera workspace。一次只启动一个模型。独立服务默认端口18765，原BF16的8765不占用。

```bash
cd /home/agilex/World_Action_Model/physical_WM/src/Fastwam-steerquant
conda activate fastwam
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"
export FASTWAM_AUTHKEY='<本机共享值；服务和客户端一致>'

# pack 服务：启动前执行完整离线审核和预热
python -u -m fastwam_steerquant.robot_async_server \
  --config configs/robot_pack_w4a8_async50.yaml \
  --report outputs/pack_w4a8_server_audit.json

# stack 服务：与上面二选一，先停止前一个服务
python -u -m fastwam_steerquant.robot_async_server \
  --config configs/robot_stack_w4a8_async50.yaml \
  --report outputs/stack_w4a8_server_audit.json
```

服务已启动时，在另一个相同环境的终端用模拟客户端（同一个 FASTWAM_AUTHKEY；output目录必须不存在）：

```bash
python scripts/mock_w4a8_async50.py \
  --config configs/robot_pack_w4a8_async50.yaml \
  --max-steps 128 --output-dir outputs/pack_mock_new --shutdown-server

# stack 服务对应命令
python scripts/mock_w4a8_async50.py \
  --config configs/robot_stack_w4a8_async50.yaml \
  --max-steps 128 --output-dir outputs/stack_mock_new --shutdown-server
```

更便捷的无动作端到端验证（自行生成临时凭证、启动和关闭自己的服务；执行前确保18765空闲，无需手动启动上述服务）：

```bash
python scripts/validate_w4a8_async50.py --task pack --max-steps 128
python scripts/validate_w4a8_async50.py --task stack --max-steps 128
```

**当前实测这两条会在延迟保护触发后返回非零，不是已通过的上线命令。** 不应通过减少max-steps到首次超时之前来宣称通过。

只做离线forward、不监听端口：

```bash
python -m fastwam_steerquant.robot_async_server \
  --config configs/robot_pack_w4a8_async50.yaml --offline-only \
  --report outputs/pack_w4a8_offline_audit.json
```

后续硬件入口在 Q 的 `scripts/deploy_real_w4a8_async50.py`，需要 `PYTHONPATH="$PWD/src:/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/.worktrees/joint-asi-async50/src"` 才能复用 A 的控制组件。当前不提供执行动作命令；它没有被本次启动，且下面条件未满足。

## 6. 后续执行前的阻塞及缺项

1. **持续50Hz吞吐不满足。** 当前约395ms/次，原窗口要求连续两次≤620ms，稳定均值须低于约310ms并留余量。需要在不改校准语义的前提下优化并重新完成128步及更长mock测试；不能减少去噪步数、更改horizon、放宽预算或叠加未经验证的K56。
2. **pack同源基线不一致。** 若要求替代当前step030000，需校准服务器提供对应030000的W4A8部署、来源SHA和同源误差报告；否则需明确选择015000作为另一个模型实验，不能称030000量化版。
3. **误差报告缺失。** 本机checkpoint目录未提供当前两个deployment相对于各自原BF16的逐维/总体动作误差、代表性阶段及独立观测验证报告。Graph/eager验证仅证明同一量化模型两种执行方式一致，不证明W4A8与BF16任务行为一致。
4. **历史颜色及训练统计来源仍有缺口。** metadata明确这是保留raw通道的实验校准；归一化动作敏感度向量是14维等权1，不是经证实的clip后训练std；训练run绑定、train/val清单未独立取得。本次原样加载，不重估、不补造，也不把这些约定视作本次新批准的训练事实。若需严格训练分布结论，须从校准/训练侧补证，尤其stack clipping维度10/12。
5. live ROS图像编码、实际分辨率、同步和机器人状态单位虽沿用原代码，本次未连硬件复核；录制HDF模拟不能验证相机延迟、CAN新鲜度或真实闭环SR。
6. 当前不存在安装依赖缺项：所需Torch/Hydra/OmegaConf/torchvision/numpy/h5py及扩展均可用。运行资产必须保留 `local/robot_handoff/`、两份deployment、编译好的W4A8扩展以及只读A工作区。复制Git仓库时，gitignore内的handoff与部署文件不会自动随Git转移。

主要证据代码：Q `deployment.load_deployment`、`policy.QuantizedPolicy`、`adapters.real_robot.RealRobotAdapter`、`runtime.WAMQuantLinear`、`rollout_graph.LiveJointDiTGraph`；A `scripts/deploy_real_async50_asi_wam.py`、`src/qi/deployment/async_joint_runtime.py`（生产循环、延迟校验）、`async_rtc_buffer.py`（对齐/融合/generation）、`async25_adapters.py`（观测/governor/sink）、`wam_policy_client.py`（现用预处理）。源文件SHA在 `source_audit.json` 中，便于后续检测A是否改变。
