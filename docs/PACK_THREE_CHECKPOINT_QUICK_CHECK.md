# pack 三模型快速检查（2026-09-24，未全部完成）

仅做模型加载、记录观测和无硬件模拟 RPC；未启动 ROS、相机、机器人或动作。所有新增/修改文件在 Fastwam-steerquant，checkpoint 未修改。

## 资产

| 模型 | 实际路径 | SHA256 |
|---|---|---|
| 原始 BF16 step_015000 | `/home/agilex/.cache/modelscope/hub/models/Mikuz12/joint_pack/step_015000.pt` | `2b5da8ac40a2701f65f1f72cd613713047271deafbd2da929b3ed8ab75b17a49` |
| W4A8 | `checkpoint/pack_w4a8/deployment.pt` | `4e3b7ff923623117d394d267243e1bb62f3027f5920ad5413754129e8dcc95b8` |
| W4A4 + RHT | `checkpoint/pack_w4a4/deployment.pt` | `60505946b575177f6649f0c64b2fed9ff944d291791c029ef4a8e1c71a64b430` |

两量化文件 SHA256SUMS 一致；W4A4 元数据为 A4 / RHT / 600 sites / 10 calls，记录的 pack 源 checkpoint SHA 与新下载文件一致。W4A8 源 SHA 同样一致。安全 CPU/mmap 加载使用 weights_only=True。

BF16 25 Hz eager；W4A8 40 Hz tile64 + Graph、无额外融合；W4A4 暂按 40 Hz、独立 native RHT kernel、auto tile + Graph、无额外融合准备。W4A4 不读取 W4A8 runtime_profile。三个模式保持原观测处理及归一化、seed=42、32-step horizon、10-step 去噪。

## 本轮结果与阻塞

运行前 GPU 无计算进程；测试期间出现其他模型进程 PID 1629461，约占 15252 MiB。未擅自终止该进程。本次测试进程已经自然结束，尝试停止时对应进程已退出。

| 模式 | 结果 | 计时 |
|---|---|---|
| BF16 | 严格 CPU 权重加载后，搬到 GPU 时 OOM | 未测得 |
| W4A8 | native tile64/Graph 审计及 128-step 无硬件联调通过 | 完整模型调用 N=12，mean 338.665 ms，p50 338.709 ms，p95 339.658 ms |
| W4A4 | 首次 forward 为有限的 [32,14] 物理动作；6000 次 Python native dispatch 已计数，但 CUDA profiler 未匹配到预期 GEMM/RHT 事件，严格审计主动终止 | 未进入正式计时/Graph/RPC 阶段 |

W4A8 时间边界：原始记录观测进入 policy，含预处理、VAE、全部去噪与动作反归一化，到物理 action 可用；GPU 同步，排除加载/预热。固定同一 HDF 的 frame 0/35 交替六轮，seed=42。不包含 RPC、相机、机器人动作。因与另一 GPU 进程共存，数据仅供故障排查，不用于三模型公平比较或 speedup 声明。

原始日志：

- `outputs/robot_profile_validation/pack_bf16_20260924_193636_152164/`
- `outputs/robot_profile_validation/pack_w4a8_20260924_193732_026626/`（含 server_audit.json、原始 action NPZ、mock/report.json）
- `outputs/robot_profile_validation/pack_w4a4_20260924_193833_827189/`

## 已准备但仍需验证的修改

- pack BF16 默认路径及 comparison spec 指向已核验的新下载文件；PACK_STEP15K_CKPT 可显式覆盖。
- W4A4 预编译扩展显式加载路径，不触发 JIT 编译或回退；新 PackedW4A4RobotPolicy、客户端身份/Graph/原生 kernel 审计、新 pack W4A4 配置与包装参数路由。
- 原 W4A8、BF16 入口继续保留。W4A4 kernel 审计失败时服务不开放监听，不允许跳过检查来运行机器人。
- `validate_robot_profile.py --quick-benchmark` 做 12 个热态样本及 128-step 模拟；现在启动前检查 GPU 计算进程，发现占用即拒绝，不自动停止旧服务。
- CPU 回归原 25 项通过；补充 W4A4 身份拒绝测试后 profile 测试 7 项通过。

## 恢复检查

先由用户安全结束现有任务并停止旧 GPU 服务，再执行（顺序运行，不并行）：

```bash
cd /home/agilex/World_Action_Model/physical_WM/src/Fastwam-steerquant
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"
unset FASTWAM_RUNTIME_PROFILE
python scripts/validate_robot_profile.py --task pack --mode bf16 --port 18767 --quick-benchmark
python scripts/validate_robot_profile.py --task pack --mode w4a8 --port 18767 --quick-benchmark
python scripts/validate_robot_profile.py --task pack --mode w4a4 --port 18767 --quick-benchmark
```

这三个命令均为无硬件验证，不是实际执行命令。W4A4 必须先定位 profiler 审计失败并通过 Graph/异步联调，再发布“已验证可运行”的三模型真机命令。不能将有限值或速度视为动作精度/SR 已通过；正式真机执行仍未测试。
