# SteerQuant 真机服务与客户端启动

更新：新下载的 pack step_015000 已找到并通过源 SHA256 校验，当前配置默认使用 `/home/agilex/.cache/modelscope/hub/models/Mikuz12/joint_pack/step_015000.pt`，无需设置 PACK_STEP15K_CKPT。该变量仅作为显式路径覆盖。原有 step_030000 不参与比较。

这里只整理供用户手动执行的命令。本次未启动服务、ROS、相机、机器人或 move-to-initial，未做正式 SR 实验。新增文件均在 Fastwam-steerquant；不改 sparse 工作区。

`serve_joint_steerquant.sh` 仿照 sparse 的任务选择方式，支持 `pack3` / `stack3`；第二参数默认为 `w4a8`，也可写 `bf16`。服务只负责模型加载和 RPC。

`deploy_joint_steerquant.sh` 设置源码路径并调用已有真实客户端，复用此前核实的 **joint-asi-async50** 运行器；没有改为引用示例中的 joint-async25。脚本不自动启动 ROS/相机、不自动附加动作标志。不传动作参数时是实时 shadow，但仍读取机器人状态及相机。

| 模式 | 推理设置 | 控制频率 | 31 步窗口 |
|---|---|---:|---:|
| w4a8 | native W4A8 tile64 + CUDA Graph，无额外 block fusion | 40 Hz | 775 ms |
| bf16 | 对应任务原始 step_015000，BF16 eager，无 Graph | 25 Hz | 1240 ms |

服务与客户端使用同任务、同模式、同一 FASTWAM_AUTHKEY。端口默认 127.0.0.1:18765；一次只开一个模型服务。等待服务输出 `PROFILE_READY` 后再运行客户端。这里延用当前配置的初始姿态和安全限制，未替换成示例工作区的参数。

## pack：W4A8 实际执行命令

以下客户端命令会真实运动：`--move-to-initial` 在等待 Enter 开始策略执行**之前**就会移动到初始姿态。Enter 提示不是初始运动的确认门。只有准备执行真机任务时才运行终端 B。

终端 A（模型服务，fastwam）：

```bash
cd ~/World_Action_Model/physical_WM/src/Fastwam-steerquant
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
read -rsp 'FASTWAM_AUTHKEY: ' FASTWAM_AUTHKEY; echo
export FASTWAM_AUTHKEY
bash scripts/serve_joint_steerquant.sh pack3 w4a8
```

终端 B（真实相机/状态与动作执行，fastwam；ROS 和相机需已就绪，source 不会启动它们）：

```bash
source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd ~/World_Action_Model/physical_WM/src/Fastwam-steerquant
read -rsp 'FASTWAM_AUTHKEY: ' FASTWAM_AUTHKEY; echo
export FASTWAM_AUTHKEY
bash scripts/deploy_joint_steerquant.sh pack3 w4a8 \
  --max-steps 100000 \
  --execute-actions \
  --move-to-initial
```

两个终端输入相同本机 authkey，不在文件中硬编码。若已经 export FASTWAM_AUTHKEY，可省略 read/export 两行。

## stack 及 BF16 的对应命令

沿用上面各终端的环境准备，只替换最后一条命令：

| 任务/模式 | 终端 A | 终端 B：真实执行 |
|---|---|---|
| stack W4A8 | `bash scripts/serve_joint_steerquant.sh stack3 w4a8` | `bash scripts/deploy_joint_steerquant.sh stack3 w4a8 --max-steps 100000 --execute-actions --move-to-initial` |
| stack BF16 | `bash scripts/serve_joint_steerquant.sh stack3 bf16` | `bash scripts/deploy_joint_steerquant.sh stack3 bf16 --max-steps 100000 --execute-actions --move-to-initial` |
| pack BF16 | `bash scripts/serve_joint_steerquant.sh pack3 bf16` | `bash scripts/deploy_joint_steerquant.sh pack3 bf16 --max-steps 100000 --execute-actions --move-to-initial` |

pack BF16 服务严格核验 SHA256 `2b5da8ac40a2701f65f1f72cd613713047271deafbd2da929b3ed8ab75b17a49`。若另行设置 PACK_STEP15K_CKPT，应在服务与真机客户端终端指定相同文件。

`--max-steps 100000` 是动作消费步数上限，不是模型去噪次数或一个 SR 试验的成功判据。每次模型调用仍是 10 步去噪、32 步动作 horizon。频率不同，100000 步对应时长也不同。

## 无动作入口保持不变

纯记录观测联调，不连接 ROS/CAN：

```bash
bash bash/run_robot_profile.sh client pack w4a8
```

实时观测 shadow，无动作、无初始移动（仍读取 ROS/CAN）：

```bash
bash scripts/deploy_joint_steerquant.sh pack3 w4a8 \
  --max-steps 128 --no-execute-actions --no-move-to-initial
```

原 `bash/run_robot_profile.sh live-client ...` 仍强制 shadow；不要给它追加 execute-actions 期待执行。需要实际执行时使用本文新的 deploy 包装，并显式传入动作标志。

已有验证：stack BF16 25 Hz、pack/stack W4A8 40 Hz 均通过 128 步无硬件联调；并非真机执行验证。本次仅新增启动包装和文档，做 shell 语法、无硬件参数路由检查，保留原运动门控与控制循环。模型/动作误差和历史遗留问题仍见 `REAL_ROBOT_W4A8_COMPARISON.md`，配置验证见 `docs/ROBOT_BF16_25_W4A8_40.md`。
