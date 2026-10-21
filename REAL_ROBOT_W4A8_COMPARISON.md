# 真机服务器 W4A8 公平对比：2026-09-24

> 历史记录说明（2026-10-21）：本文保留当时的测试结果和资产状态。后续已取得 pack 原始 step_015000，当前 `configs/comparison_pack_15k.json` 已设置实际路径；下文“未找到”及“路径为 null”描述仅对应当时状态。后续检查结果见 [pack 三模型快速检查](docs/PACK_THREE_CHECKPOINT_QUICK_CHECK.md)，当前启动方式见 [真机启动命令](docs/REAL_ROBOT_LAUNCH_COMMANDS.md)。取得权重不代表已完成 pack 公平误差、speedup 或 SR 验证。

## 状态与范围

**stack 已完成 step_015000 BF16 Graph 对 W4A8 tile64 Graph 的无动作实测。pack 已完成量化适配和无动作测量，但缺少对应原始 step_015000，公平 action 误差和 speedup 尚未测试。两任务 SR 均未测试。**

两侧持续50Hz均不能据本次结果认证：stack BF16在第33个shadow控制步因动作队列匮乏停止；W4A8 pack/stack在第33步后触发连续推理延迟保护。没有通过缩短测试、降低频率、改同步、增加超时或启用额外融合来宣称通过。

本次没有机器人动作、move-to-initial、ROS、相机或CAN连接，没有commit/push，没有修改checkpoint。所有代码、配置及实验产物改动仅在 `/home/agilex/World_Action_Model/physical_WM/src/Fastwam-steerquant`（Q）。所有本次测试服务已结束。

## 环境与源码

- Q revision：`3925c6e33ad6e6a11c5d33d37df11c4a49f7662c`。已有脏工作区保留；未切换分支、未覆盖原BF16启动配置。
- 异步工作区 A：`/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/.worktrees/joint-asi-async50`，revision `836dd4796ae57653c22c48991ec5c9f6fd8b9c88`。只读复用 runtime、buffer、governor、录制观测reader及ShadowActionSink；原有启动脚本修改和assembly文件保留。
- 环境 `/home/agilex/miniconda3/envs/fastwam/bin/python`：Python3.10、PyTorch2.7.1+cu128、CUDA12.8、NVIDIA driver580.95.05、物理RTX4090 24GiB、SM8.9；CPU torch threads=4。
- 顺序：stack BF16 → stack W4A8 → pack W4A8，随后单独串行补测stack两侧入队时间。每模型新进程，同一GPU、同一环境，不同时驻留两个模型。
- 每次前后 `gpu_before.txt/gpu_after.txt` 保留完整nvidia-smi。没有其他GPU计算进程；桌面/Xorg/浏览器等图形进程仍存在，约0.7GiB显存，非完全独占GPU。没有锁GPU时钟，也未修改桌面/驱动环境。
- 已完整阅读并落实 `docs/REAL_ROBOT_W4A8_RUNTIME.md`、`docs/DIT_ONLY_COMPARISON.md`、`docs/PERFORMANCE_ALIGNMENT.md`。其中304.7ms/326.8ms等其他服务器结果没有用于本报告计算。

## 资产身份及对照定义

| 资产 | pack | stack |
| --- | --- | --- |
| deployment（相对Q） | `checkpoint/pack_w4a8/deployment.pt` | `checkpoint/stack_w4a8/deployment.pt` |
| deployment SHA256，目录SHA256SUMS验证成功 | `4e3b7ff923623117d394d267243e1bb62f3027f5920ad5413754129e8dcc95b8` | `833a84d13ca4d80c1c6e6cdd1e7b41ce4b53380aa01c7c8c794c73ba142e1912` |
| metadata指定的15k源ckpt SHA256 | `2b5da8ac40a2701f65f1f72cd613713047271deafbd2da929b3ed8ab75b17a49` | `a86b2223d4b0f1fcb079532a5a2898a5ab9f529da3143994721cdaa32d29fc9f` |
| 本次原始baseline | **未找到；未以其他文件替代** | `/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/checkpoint/stack/step_015000.pt`，实测SHA与上项一致 |
| 原始baseline dtype | 未加载，不能称已实测；构造/metadata配置为BF16 | ckpt中step=15000、torch_dtype=torch.bfloat16；严格加载后600个目标Linear权重实际均为BF16 |

当前旧异步pack配置指向step_030000，其SHA为 `29ba8afef1c7eacd0c70e5f9f6184ac6a6952cc1fd0c3f9a31cd08be0d78bb09`，**完全未作为本次baseline加载或测速**。量化YAML中既有 `baseline_checkpoint` 字段仅保留旧异步部署来源记录；公平比较器只读取独立 `comparison_*_15k.json` 中的明确源路径，要求内容SHA及ckpt内step=15000同时通过，不依据文件名猜测。

已检索项目、Downloads、tmp及home下的15k文件；除stack外发现的一份回收站同名文件SHA不匹配，未使用。pack源文件路径已向用户询问；当前 `configs/comparison_pack_15k.json` 中该字段为null，因此默认公平比较命令会明确拒绝，而不会退回030000。

共同VAE：`/home/agilex/World_Action_Model/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`，SHA `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`。BF16读取并验证此文件；W4A8直接读取deployment内的VAE state，不重载原checkpoint到GPU。stack两侧每个观测的首帧VAE latent哈希也完全一致。

| 小型资产SHA256 | pack | stack |
| --- | --- | --- |
| stats | `30561edb2897b9784fbf13f4e42ca9897e141b4ebd60119d047d255758aaa6bf` | `68c662dba501b1e110e09896238bfd0ae1d15c226e2c05683958bba915f7e0cb` |
| text embedding cache | `927573ff154e4303f06a3a719167c198b0b4ed5ea1bd72b23768217fd8381960` | `a83dfdf3a215e8314206bbf5080565617c33e23622ad05a18dd3227afa5fddf6` |

完整哈希固定源码/资产闭包仍为 `local/robot_handoff/`；manifest自身SHA `a429d5d2c263e988d31d20181a5621210c973c14feaf5489d700a12af9ffc359`。源码、配置、stats/cache及选中原始观测字节的SHA见 `outputs/real_robot_comparison_20260924/source_assets_observations.json`。本次没有重新估计stats或修改归一化。

## 实际运行配置与公平性

W4A8由新版 `QuantizedPolicy(runtime_profile=...)` 明确读取 `configs/real_robot_w4a8_runtime.json`，在导入扩展前应用profile。实测日志和每次RPC诊断均为 **tile64、cuda_graph=True、fuse_block=False、fused_modules=0**。不是仅设置环境变量后推断Graph开启。

使用原已编译 `~/.cache/torch_extensions/py310_cu128/fastwam_steerquant_cutlass_w4a8_sm89_shape_v3/fastwam_steerquant_cutlass_w4a8_sm89_shape_v3.so`，没有重新编译或载入W4A4扩展。每任务完整前向profile确认600个目标层×10次=6000次native调用、6000个A8 producer、6000个实际tile64 W4A8 GEMM。内部producer/epilogue融合保留，额外AdaLN/gate-residual融合关闭。scale/gain/clipping维持FP32、权重为packed uint8，无BF16目标层静默回退。

共同模型：相同dense FastWAMJoint结构，video/action各30层，hidden3072/1024，14维action/proprio；9视频帧、3潜帧，每潜帧120 tokens；action horizon32，10次联合去噪，CFG1，video/action scheduler infer_shift5、sigma_shift=None。没有K56 token剪枝、expert cache或动作prefix条件。**比较的是相同dense模型的量化收益，不是原K56稀疏服务与dense W4A8的混合比较。** 原ASI客户端50Hz调度保持。

同seed42，视频和action分别用相同seed初始化独立CPU generator。stack 12个对照观测的初始action噪声、未来视频噪声、首帧VAE latent和含proprio的context哈希逐项相同。每侧先用两个不同观测对比Graph输出与该侧eager输出（不用于性能倍率），严格校验通过；后续12观测逐个确认Graph静态缓冲区复制了当前输入，不复用旧观测。每模型10张图、每请求10次replay，捕获校验全部通过。

相机按high/left_wrist/right_wrist处理，原始uint8 `[480,640,3]`；resize/拼接/裁剪与A当前纯函数及依赖逐项核对相同，模型输入 `[1,3,384,320]`、BF16、[-1,1]。两个任务均保留raw HDF通道（as_stored），没有猜测RGB/BGR或交换通道。live ROS路径仍按原encoding转换RGB，本次没有连接相机，历史HDF颜色未证实。

state14是左右各6关节角(rad)+夹爪宽(m)。服务端使用原processor全局z-score及clip[-5,5]；模型最终归一化action反归一化**一次**后输出float32 `[32,14]`，队列不重复处理。prompt/text cache、stats、预处理、scheduler、输入形状、seed及异步参数两侧完全一致。

## 观测和测试规模

- pack：dashboard session `pack_3_objects_original_20260911_170411_3f7b45/raw/pending/episode_0.hdf5`，908帧，session.json的runtime_task_alias=pack_3_objects_plus，录像采集时model_id=k4_disjoint_pack_3_objects。
- stack：`stack_cups_original_20260814_193507_3f1a87/raw/episode_001/episode.hdf5`，409帧，runtime_task_alias=stack_3_cups_gen，采集时model_id=groot_n1d7_grootstack3cups_checkpoint-12000。
- 完整路径/12个frame index见 `configs/comparison_{pack,stack}_15k.json`。从整条录像均匀采样，覆盖时间全程；没有阶段人工标签，不能声称已独立核验抓取/放置等各阶段覆盖。
- 这是独立评测来源的录制观测，非本次模型的真实闭环运行。与所声明校准perfect/episode_0、35、70来源不同；未取得校准逐帧内容manifest，**未完成严格内容去重/训练划分证明**，不把文件名不同当作完整held-out证明。没有使用三条轨迹重估训练统计。
- 先两观测smoke/eager-Graph对照，然后12观测完整action留样。性能选其中6观测，各预热2次后计时4次，共24样本/模式/口径。连续两次RPC另做12对，双方均排除首次加载/捕获/预热。

## 计时边界

1. **DiT去噪循环（含scheduler）**：CUDA Events从第1次 `_predict_joint_noise` 入口到第10次action scheduler.step返回，覆盖10次joint DiT、两路scheduler步进及中间latent更新，不含建schedule、VAE、前处理、RPC、动作执行。CUDA同步后读取事件。首步timestep reshape位于起点之前。
2. **core + live Graph I/O（补充口径）**：累加10次 `_joint_denoise_core` 的GPU事件，含每步Graph输入拷贝/输出clone，不含scheduler。它不是历史固定输入、无拷贝的“纯core”计时，不能与304.7ms直接相除。
3. **完整模型调用**：已有原始观测进入policy到可用物理action结束，客户端/传输之外的服务端墙钟。包含图像预处理、state归一化、context搬运、VAE、噪声、完整去噪、CPU输出和反归一化。计时前后同步CUDA。
4. **RPC端到端**：客户端发起send前到recv获得可入队action的墙钟，包含实际本机RPC、pickle序列化、服务处理/等待及返回；不含HDF磁盘读取、相机、队列入队和机器人动作。另报追加diagnostics后的总时间，该诊断RPC属于现有异步每轮路径。
5. **入队完成**：另外用原buffer.enqueue实例包装记录调用起止，参数/返回值及内部实现未改。完成时间表示原调用返回/已发布，精确内部插入点位于记录区间内。没有把首次消费时间冒充入队时间。详见 `enqueue_measurements.json`。

CUDA事件只在独立denoise阶段启用；完整模型/RPC阶段不记录这些GPU事件。RPC表是受控串行生产请求的稳态测量；真实原异步runtime另试跑128步，结果失败，不能据稳态RPC表宣称持续50Hz已通过。两次连续请求合计包含两轮diagnostics及中间主机调度。

## 已实测：完整action误差与范围

stack：12个观测，每个完整32×14，合计5376个归一化动作标量。总体 RMSE **0.03356101**，MAE **0.02058010**，最大绝对误差 **0.32617188**。

| 物理单位分组 | RMSE | MAE | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| 12个关节维(rad) | 0.00545904 | 0.00332708 | 0.03656971 |
| 2个夹爪维(mm) | 1.06958504 | 0.56121894 | 9.04148445 |

未把rad和m混成一个总物理RMSE。逐维结果如下，夹爪为便于阅读转换成mm，原始CSV保留m。

| 维度/含义 | 归一化RMSE | 归一化MAE | 归一化max | 物理RMSE | 物理MAE | 物理max | 单位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 left_joint_1 | 0.022292 | 0.018234 | 0.062500 | 0.002880 | 0.002356 | 0.008074 | rad |
| 1 left_joint_2 | 0.014867 | 0.011010 | 0.046875 | 0.006767 | 0.005011 | 0.021334 | rad |
| 2 left_joint_3 | 0.020078 | 0.015612 | 0.054688 | 0.008337 | 0.006482 | 0.022707 | rad |
| 3 left_joint_4 | 0.028002 | 0.020733 | 0.093750 | 0.002660 | 0.001970 | 0.008907 | rad |
| 4 left_joint_5 | 0.039313 | 0.027396 | 0.125000 | 0.005702 | 0.003974 | 0.018131 | rad |
| 5 left_joint_6 | 0.013319 | 0.010417 | 0.042969 | 0.002144 | 0.001676 | 0.006916 | rad |
| 6 left_gripper | 0.054032 | 0.034735 | 0.326172 | 1.497757 | 0.962847 | 9.041484 | mm |
| 7 right_joint_1 | 0.054347 | 0.030378 | 0.320312 | 0.005949 | 0.003325 | 0.035062 | rad |
| 8 right_joint_2 | 0.032513 | 0.019566 | 0.136719 | 0.008697 | 0.005233 | 0.036570 | rad |
| 9 right_joint_3 | 0.029331 | 0.016820 | 0.105469 | 0.006928 | 0.003973 | 0.024912 | rad |
| 10 right_joint_4 | 0.039862 | 0.030544 | 0.203125 | 0.003036 | 0.002326 | 0.015468 | rad |
| 11 right_joint_5 | 0.027166 | 0.018672 | 0.162109 | 0.002157 | 0.001482 | 0.012869 | rad |
| 12 right_joint_6 | 0.041807 | 0.021488 | 0.265625 | 0.004117 | 0.002116 | 0.026160 | rad |
| 13 right_gripper | 0.016592 | 0.012518 | 0.062500 | 0.211541 | 0.159591 | 0.796832 | mm |

三组已测动作（stack BF16/W4A8、pack W4A8）均为 `[12,32,14]`，NaN/Inf为0，归一化绝对值>5计数为0。归一化总体范围分别为：stack BF16 [-2.53125,2.453125]、stack W4A8 [-2.484375,2.40625]、pack W4A8 [-2.703125,2.0625]。逐维归一化/物理范围见 `action_ranges.csv`。

**pack异常**：左夹爪dim6原始预测范围[-0.0094160,0.0672215] m；384个该维输出中50个低于训练global_min=-0.0058 m。它们同时为负夹爪宽，不能直接发送机器人；原安全投影/限幅保留。缺少同源BF16，不能断言异常由量化造成。stack两侧的这些观测没有超出训练action extrema。

**pack最终action误差：未测试**，不能拿stack误差或030000结果补上。

补充实际消费的chunk索引：按相同录制观测及seed对齐两侧输出，仅比较在任一shadow队列中有非零贡献的候选索引并集；不把融合后/限幅后命令当作原始模型action，也不代替完整12观测误差。此处延迟会跳过chunk头部，第二块实际是尾段，不伪称前缀。

| 观测 | 实际候选索引并集 | 归一化RMSE |
| --- | --- | ---: |
| async:0 | 0–31（32步） | 0.07276212 |
| async:1 | 17–31（15步） | 0.07354567 |

逐维/分单位的消费索引误差也保存在summary.json。动作误差不能推出SR保持不变。

## 已实测：性能（毫秒）

| stack口径 | 模式 | N | mean | p50 | p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| DiT去噪+ scheduler GPU | bf16 Graph | 24 | 444.197 | 444.200 | 445.870 |
| DiT去噪+ scheduler GPU | w4a8 Graph | 24 | 321.436 | 321.539 | 322.340 |
| core + live Graph I/O GPU | bf16 Graph | 24 | 443.953 | 443.955 | 445.628 |
| core + live Graph I/O GPU | w4a8 Graph | 24 | 321.172 | 321.286 | 322.094 |
| 完整模型墙钟 | bf16 Graph | 24 | 460.311 | 460.236 | 462.676 |
| 完整模型墙钟 | w4a8 Graph | 24 | 339.040 | 339.079 | 341.176 |
| RPC收到action | bf16 Graph | 24 | 463.128 | 462.795 | 465.522 |
| RPC收到action | w4a8 Graph | 24 | 342.024 | 341.973 | 344.254 |
| RPC含diagnostics | bf16 Graph | 24 | 463.316 | 462.961 | 465.767 |
| RPC含diagnostics | w4a8 Graph | 24 | 342.269 | 342.262 | 344.509 |
| 连续两次RPC含diagnostics | bf16 Graph | 12 | 929.981 | 929.571 | 932.973 |
| 连续两次RPC含diagnostics | w4a8 Graph | 12 | 688.983 | 687.711 | 694.827 |

speedup = 相同口径BF16统计值 / W4A8统计值，**Graph对Graph**：

| stack口径 | mean speedup | p50 speedup | p95 speedup |
| --- | ---: | ---: | ---: |
| DiT去噪+ scheduler GPU | 1.3819× | 1.3815× | 1.3832× |
| core + live Graph I/O GPU | 1.3823× | 1.3818× | 1.3835× |
| 完整模型墙钟 | 1.3577× | 1.3573× | 1.3561× |
| RPC收到action | 1.3541× | 1.3533× | 1.3523× |
| RPC含diagnostics | 1.3537× | 1.3526× | 1.3520× |
| 连续两次RPC含diagnostics | 1.3498× | 1.3517× | 1.3427× |

pack仅W4A8一侧实测，baseline与speedup均未测试：

| pack W4A8口径 | N | mean | p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| DiT去噪+ scheduler GPU | 24 | 320.960 | 320.960 | 322.048 |
| core + live Graph I/O GPU | 24 | 320.713 | 320.715 | 321.803 |
| 完整模型墙钟 | 24 | 339.572 | 339.593 | 341.188 |
| RPC收到action | 24 | 342.519 | 342.368 | 344.243 |
| RPC含diagnostics | 24 | 342.745 | 342.593 | 344.406 |
| 连续两次RPC含diagnostics | 12 | 687.783 | 687.844 | 689.883 |

加载和启动审核成本（不纳入以上统计），显存是本进程max allocated：

| 模式 | 加载秒（含身份检查/构造） | 预热/捕获/审核秒 | 峰值allocated GiB |
| --- | ---: | ---: | ---: |
| stack bf16 | 53.902 | 7.555 | 12.812 |
| stack w4a8 | 31.681 | 14.932 | 4.651 |
| pack w4a8 | 33.812 | 14.705 | 4.651 |

独立入队探针（每侧只有2个成功入队样本；与主稳态表分开）：

| 模式 | 请求→入队完成两次实测(ms) | 收到action→入队完成(ms) |
| --- | --- | --- |
| pack_w4a8 | 342.397, 344.355 | 0.274, 0.295 |
| stack_bf16 | 471.644, 458.989 | 0.175, 0.257 |
| stack_w4a8 | 357.569, 336.828 | 0.562, 0.118 |

RPC主表不含入队；此表额外包含diagnostics、生产者检查及buffer.enqueue。精确单调时钟时间戳和enqueue调用区间保留在JSON。没有测相机采集→入队口径。


## 异步语义与预算

仍是单生产线程同步RPC + 独立50Hz消费者；chunk32、exp_weight_factor0.2、arrival_smoothstep_steps5、fresh_chunk_grace0.6s、RPC timeout10s、空队列等待0.2s、1步headroom。观测probe cursor、采样后cursor、monotonic时间、过期/generation检查、按t-cursor选择、重叠chunk融合、governor与sink安全约束均使用A原代码。没有把推理改成控制线程同步执行。

`--max-steps=128`仅是本次shadow消费者上限，不是去噪步数。三个主试跑都只得到33个shadow动作；BF16 stack发生空队列超时，量化两任务发生连续延迟预算拒绝，均记录error及全部trace。没有将这些停止当作SR试验或成功运行。

12对连续请求全部超过620ms：stack BF16最小927.91ms、W4A8最小686.21ms；pack W4A8最小685.56ms。W4A8较之前auto tile变快，但当前配置仍不能持续满足原50Hz/horizon32预算。需要后续优化并复测，不能修改预算来通过。

## 修改文件及原因

- `robot_async_server.py`：抽出共享资产/adapter准备；移除旧的显式Graph=False；先由runtime_profile设置tile，再载入扩展；启动审计实际tile64 native GEMM。eager kernel审计仅在监听前临时移除Graph，完成后恢复并严格捕获；只在Graph已预热后接受客户端。保留一次反归一化和全部RPC envelope。
- `robot_async_client.py`：诊断必须证明tile64/Graph/no-extra-fusion及实际kernel计数，否则拒绝，不只检查环境变量。
- `configs/robot_{pack,stack}_w4a8_async50.yaml`：加入明确profile路径；原BF16配置没有修改。
- 新增 `robot_comparison.py`、`scripts/serve_robot_comparison.py`：同源BF16 Graph加载器、初始噪声/条件哈希、不同输入Graph验证、独立GPU事件计时和动作留样。checkpoint加载只用CPU weights_only/mmap；BF16源要求SHA+step，不允许030000回退。
- 新增 `scripts/compare_robot_w4a8.py`：新进程串行切换、客户端墙钟、连续两次计时、原异步无动作试跑及入队探针。认证值只在子进程环境中，不写入日志/配置。
- 新增 `scripts/summarize_robot_comparison.py`、两份`comparison_*_15k.json`：固定观测、逐维/分单位误差、各口径统计、raw CSV及speedup。
- 更新协议测试并新增 `tests/test_robot_comparison.py`，验证完整chunk末端误差不会漏计、物理单位分离、非有限值拒绝等；本次最后一组CPU测试27项通过。完整GPU forward/native/Graph验证由实际两任务执行证明。
- 没有改额外融合、kernel数值实现、checkpoint、A工作区或原机器人安全限幅。此前已有的prebuilt extension loader改动保留。

## 可复现无动作命令

以下工作目录统一为Q，已有fastwam环境，无需安装依赖或source ROS。

```bash
cd /home/agilex/World_Action_Model/physical_WM/src/Fastwam-steerquant
conda activate fastwam
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"

# 自动串行启动BF16 Graph、W4A8 Graph及模拟客户端；输出目录应未使用
python scripts/compare_robot_w4a8.py --task stack --output-dir outputs/comparison_new

# 当前pack仅能做量化一侧，不能产生同源误差/speedup
python scripts/compare_robot_w4a8.py --task pack --modes w4a8 --output-dir outputs/comparison_new

python scripts/summarize_robot_comparison.py --input-dir outputs/comparison_new

# 单独补测入队时间，不重复完整性能/误差扫描
python scripts/compare_robot_w4a8.py --task stack --only-async --output-dir outputs/enqueue_new
```

pack路径补齐并通过SHA/step验证后，才可运行不带 `--modes w4a8` 的pack公平对比命令；当前这样运行会报缺少baseline。

手动启动公平BF16服务器（只读录制观测预热，监听18766，不连接硬件）：

```bash
export FASTWAM_AUTHKEY='<本机共享值>'
python -u scripts/serve_robot_comparison.py \
  --config configs/robot_stack_w4a8_async50.yaml \
  --spec configs/comparison_stack_15k.json --mode bf16 \
  --port 18766 --output-dir outputs/stack_bf16_manual
```

量化服务启动（pack/stack二选一，默认18765；不得与自动测试同时占用GPU）：

```bash
export FASTWAM_AUTHKEY='<本机共享值>'
bash bash/with_real_w4a8_runtime.sh python -u -m fastwam_steerquant.robot_async_server \
  --config configs/robot_pack_w4a8_async50.yaml --report outputs/pack_profile_audit.json

bash bash/with_real_w4a8_runtime.sh python -u -m fastwam_steerquant.robot_async_server \
  --config configs/robot_stack_w4a8_async50.yaml --report outputs/stack_profile_audit.json
```

手动量化服务可用 `scripts/mock_w4a8_async50.py --config <对应量化YAML> --max-steps 128 --output-dir <新目录> --shutdown-server` 联调。**不要使用硬件执行入口做smoke。** 自动比较脚本返回0只表示测量产物成功收集；部署是否满足50Hz必须看async.error、完成步数和连续两次预算。本次均未满足。

## 原始产物

主目录：`/home/agilex/World_Action_Model/physical_WM/src/Fastwam-steerquant/outputs/real_robot_comparison_20260924/`。

- `summary.json`、`performance.csv`、`raw_timings.csv`、`action_error.csv`：结果及逐次原始计时；逐维范围/异常计数也在summary。
- `{stack/bf16,stack/w4a8,pack/w4a8}/server.json`、`client.json`、`startup.json`、`server.log`：每请求ID/seed/阶段、噪声哈希、Graph、加载成本、profile和kernel证据。
- 对应`actions.npz`：归一化和物理坐标全部action；顺序与server.json.records严格一致，完整 `[N,32,14]`。可用numpy.load(..., allow_pickle=False)只读。
- `async.json`：完整chunk、时间戳、每步候选索引/权重/融合/限幅结果及异常，所有实际输出仅在ShadowActionSink中记录。
- `enqueue_measurements.json`：入队时间明细。stack补测原始记录另在 `outputs/real_robot_enqueue_probe_20260924/stack/{bf16,w4a8}/`，每侧只有2个成功入队样本，不能当作稳定分位数基准。
- `source_assets_observations.json`、`gpu_before.txt/gpu_after.txt`：身份和负载证据。
- `SR_TRIAL_PLAN_PENDING.csv`：80行**计划**，status=PLANNED_NOT_EXECUTED，绝不是80次已执行SR试验。

## SR方案：需要用户确认，尚未执行

先决条件：补齐pack15k来源、审阅action误差/异常、两侧通过持续异步预算和安全验证，然后用户明确授权。当前仅凭用户确认仍不能把延迟失败当作已经解决。

建议每任务20对配对试验，每模型20次，两任务共80次。5种有标记的初始摆放模板×每模板4对；每对在同一模板复现，BF16/W4A8先后交替AB/BA，每侧seed42。保存开始RGB、state、物体/杯位置记录；条件偏差在启动前重置，不在看到结果后删试验。可以由不知模型标签的人判定录像结果。

- pack：拟最长90秒（50Hz最多4500控制步）；3个指定物体均完全放入目标容器，双夹爪释放、目标状态稳定3秒。物体/容器具体清单、容器边缘是否算入需确认。
- stack：拟最长60秒（3000步）；3杯形成规定目标塔、夹爪释放并稳定3秒。**目标是两底一顶还是其他堆叠拓扑必须以用户目标图/判据确认，不能仅凭prompt猜测。**
- 频率、队列、32步horizon、10去噪步、Graph设置、限幅/重试/通信检查两侧一致。沿用原空间键/急停、相机新鲜度/偏差、CAN状态新鲜度、有限值/维度检查、关节/夹爪界限、最大步长、空队列及连续推理预算。失败立即停止，不为了收集SR放宽保护。
- 重置由现场操作员按标记恢复物体/杯和机器人起始状态，拍照核对后再开始下一次。任何自动回位或move-to-initial需另行明确授权，本次没有执行或自动安排。
- 从操作员按“开始本次试验”起记入主要分母，包括模型启动失败、任务失败、超时、人工接管、安全停止、推理/通信错误，不能只统计成功完成的有效运行。逐次记录一个主结果分类及全部附加标记、耗时、原因和录像。计划未启动的行不进分母。
- 主要SR不排除已启动失败；若存在外部停电、放错物体等独立程序违规，仍在原始表和主要全尝试SR中保留，另给预先约定的敏感性分析，并逐条公开排除理由，不按模型结果事后排除。
- 分任务、分模型给成功数/已启动总数、SR和95% Wilson区间；差值按 `100*(SR_W4A8-SR_BF16)` 报百分点，配对数据另做按pair重采样的95% bootstrap差值区间。20次/模型只是初步实验，区间会较宽，不能凭点估计宣称等效。

**当前 pack BF16/W4A8、stack BF16/W4A8 的SR、置信区间和SR差值均为“未测试/不适用”，不是0%，也不能由离线误差推断。**

## 遗留项

1. pack原始15k缺失，阻塞其公平动作误差与speedup；需要用户/校准服务器提供与已列SHA匹配的文件路径或文件，不能用030000或从量化权重重建伪baseline。
2. 当前相同dense Graph基线与量化都不满足原50Hz窗口，阻塞持续真机执行；量化加速不等于实时预算通过。
3. pack出现负夹爪原始预测；缺少BF16无法归因于量化。保留原governor，不通过改stats/clip隐藏异常。
4. 历史RGB/BGR、训练run与stats绑定、裁剪后训练std及精确校准帧manifest仍缺证据。当前deployment沿用其metadata中的实验性归一化动作等权敏感度，不冒充训练std，本次没有重新批准或重算这些统计。它们不妨碍在同输入下记录输出差异，但限制训练分布/SR结论。
5. 相机到入队、真实机器人控制延迟与SR均未测试；本机RPC结果不能代替机器人网络/相机全链路结果。下一阶段需要用户确认SR方案和动作授权，并先解决上述技术阻塞。
