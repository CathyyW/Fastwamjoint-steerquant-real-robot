# FastWAMJoint 真机模型接入 SteerQuant W4A8 / W4A4 校准与部署交接

审计日期：2026-09-23（Asia/Shanghai）。本报告描述静态代码、只读配置和本地资产检查；**未实施适配、未校准、未运行模型推理或 GPU 加载、未启动任何服务/客户端/ROS/相机/机器人，未执行动作**。唯一新增文件为本报告。未安装依赖或下载资产；没有访问远端网站，远端下载状态来自委托说明。

本地 checkpoint 检查只用现有 fastwam 环境 PyTorch 2.7.1+cu128 的 `torch.load(..., map_location="cpu", weights_only=True, mmap=True)`，检查键、shape、dtype；未实例化完整模型。大 checkpoint SHA256 用 8 MiB 缓冲流式读取。下载缓存 `.msc` 只用 `pickletools.genops` 读取字面字符串，**没有 unpickle**。另用一个 2×2 Linear 验证 PyTorch 重复模块注册的 state_dict 路径行为，不涉及实际模型推理。

## 1. 接手结论与证据坐标

可以保留现有 RPC、相机和 RTC 控制链，将服务内 `WAMPolicyClient` 的模型加载与推理分支接到 SteerQuant。当前迁移包不能直接用于这两个模型：至少缺真实 bridge、纯结构构建、可微 joint core、重复模块路径的 packed 导出/加载处理。训练来源与归一化匹配也尚未闭合，尤其 stack。

全文证据路径使用如下可迁移前缀，接收方不必能访问原服务器绝对路径。附录含配置、关键源码、统计和文件哈希；未完整内嵌的大模块须按第 10 节完整传输。

|前缀|本机位置|revision / 状态|
|---|---|---|
|S|`/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse`|`88fed984c03ea963ee6cc24f7753c59c20908c2d`，分支 `version/asi-k56`，有工作区改动|
|Q|`/home/agilex/World_Action_Model/physical_WM/src/Fastwam-steerquant`|`b20dbb147c1f0965771d8cc0d148484b2a19649d`，分支 `main`，审计开始时干净|
|P|`/home/agilex/World_Action_Model/physical_WM`|`0e7d5e5284d702294d3129fc0ddc62dee3a4d1b6`，分支 `cosmos`；只作本地基线/转换器参考|
|W|`/home/agilex/World_Action_Model`|工作区位置；数据文件不是源码 revision|
|SDK|`/home/agilex/miniconda3/envs/fastwam/lib/python3.10/site-packages/pyAgxArm`|已安装文件，只读源码证据，无 Git revision 保证|

`S:文件:行号` 默认对应 S revision 的**本次工作区内容**；不能只 checkout commit 而忽略改动。关键本地改动：

- `S:src/qi/deployment/wam_policy_client.py:48-50,89-92,159-173` 增加可选 `num_video_frames` 及转发，否则 Joint 必需参数丢失。
- `S:src/qi/deployment/robot_runtime.py:71-92,163-166` 用 NumPy 解码 ROS image，正确处理 rgb8/bgr8/rgba8/bgra8/mono8、stride，输出 RGB；不再依赖 cv_bridge 的 passthrough。
- 两个 `real_deploy_joint_*3.yaml`、两个 `train_config_joint_*3_infer.yaml` 都未跟踪；离线 smoke、overlay 脚本也未跟踪。已有 tracked 修改还包括 `docs/deploy_real_bot.md`，另有 assets/checkpoint/output_actions 等未跟踪目录。模型源码本次未显示未提交改动；其 sparse 改造已在 S revision 内。
- P 的 `src/fastwam/models/wan22` 与 HDF5 转换器在限定路径 status 检查中没有本地改动。P 只是可用的本地“普通 FastWAMJoint”对照，不能称为经权重作者确认的训练 revision。

## 2. 两任务最终有效参数

“最终”指按题述命令和当前磁盘代码静态解析的值，不代表已观测运行进程。服务端 CLI 只有 config/host/port/authkey，**没有 checkpoint 或精度 CLI 覆盖**。

`serve_policy.main` 将部署 YAML 路径 expanduser+resolve，用 OmegaConf.load 读取，然后 `WAMPolicyConfig.from_mapping(..., base_dir=config_path.parent)` 只消费 `wam`。相对 ckpt/stats/train_config/cache/config_dir 均以 YAML 所在目录为基准（字段存在时）；本次四个模型相关路径都是绝对路径。`train_config` 非空时 `_load_config` 直接 OmegaConf.load，**没有 Hydra defaults 继承/compose**；两个推理配置也没有 defaults。未设置的 task/config_dir 默认值在此路径不生效。`use_text_encoder=false` 不主动强制覆盖 train config，但本次 train config 本身也为 false。[S:scripts/serve_policy.py:15-31；S:src/qi/deployment/wam_policy_client.py:30-93,182-211]

|参数|代码默认 / 来源|pack 最终|stack 最终|
|---|---|---|---|
|ckpt|必填 YAML，无 CLI 覆盖|`S/checkpoint/pack_object/step_015000.pt`|`S/checkpoint/stack/step_015000.pt`|
|stats|必填 YAML|`P/checkpoints/k4_disjoint_pack_3_objecst/dataset_stats.json`|`P/checkpoints/fastwam_stack_3_cups/dataset_stats.json`|
|train_config|默认 None；YAML 指定|`S/configs/train_config_joint_pack3_infer.yaml`|`S/configs/train_config_joint_stack3_infer.yaml`|
|模型工厂 / 类|推理配置|`qi.runtime.create_fastwam_joint` / `FastWAMJoint(FastWAM)`|相同|
|device / dtype|policy 默认 cuda:0 / bf16；YAML 同值|cuda:0 / BF16|相同|
|load_text_encoder|工厂默认 true；两个推理配置 false|false，tokenizer 也不加载|相同|
|prompt|默认 clean the table；YAML 覆盖|`Put the three objects on the table into the container.`|`Stack the three cups into a small tower.`|
|context cache|YAML|`P/data/text_embeds_cache/pack_3_objects_plus`|`P/data/text_embeds_cache/stack_3_cups_gen`|
|context_len|data.train / 默认128|128 text + 1 proprio token|相同|
|action_horizon|policy 默认32；YAML32|32×14|相同|
|num_inference_steps|Joint 方法默认20；policy 默认10；YAML10|10 个 joint 去噪调用|相同|
|num_video_frames|policy 默认 None；Joint 必填；YAML9|9，VAE latent T=3|相同|
|seed / rand_device|policy默认及YAML|42 / cpu|相同|
|CFG / negative_prompt|Joint 方法默认1 / None；未传|1 / None|相同|
|sigma_shift|Joint 默认 None；未传|None → video/action infer_shift=5|相同|
|scheduler|推理配置|WanContinuousFlowMatchScheduler，train/infer shift=5，1000训练时间刻度|相同|
|tiled / expert_cache|Joint 默认false；未传|false / false|相同|
|sparse/K56/pruning|无安装控制器调用|未启用，dense 路径|相同|
|torch.compile / CUDA Graph|Joint入口无此开关，未安装|当前链路未启用|相同|
|attention|源码函数名 flash_attention|实际调用 PyTorch SDPA；具体 GPU 内核未确认|相同|
|图像输入|服务预处理|`[1,3,384,320]` BF16，[-1,1]|相同|
|action_video_freq_ratio|推理配置不含此字段|推理几何 32/(9−1)=4；参考数据配置为4；实际训练待确认|相同|
|量化流（建议）|从当前几何推导，并非模型已量化|video 3×120 tokens；action 2×16；context 128+1|相同|
|server host / port|CLI 默认127.0.0.1/8765，题述显式覆盖|127.0.0.1 / 8765|相同|
|server auth|题述 CLI 覆盖；不读取 policy_server.authkey|`<LOCAL_AUTHKEY>`|相同，需与客户端 YAML 保持一致|
|client host / port|CLI默认None → YAML|127.0.0.1 / 8765|相同|
|runtime arm / chunk|YAML|dual / 32|相同|
|RTC exp factor|runtime默认0.5；YAML覆盖|0.2|0.2|
|控制率|YAML50|名义50 Hz，实际未测|相同|
|execute_actions|CLI默认None；YAML false；题述CLI覆盖|false（--no-execute-actions）|true（--execute-actions）|
|move_to_initial|CLI默认None；YAML false；题述CLI覆盖|true|true|
|max_steps|YAML null；题述CLI覆盖|128 个选中动作控制步|100000 个选中动作控制步|

**pack 题述客户端仍会连接并使能机器人，且先运动到初始位。** `--no-execute-actions` 仅禁止控制循环中的 `robot.move(action)`；不约束 connect/enable 或 move_initial。此报告没有启动任何客户端。[S:scripts/deploy_real_rtc_wam.py:31-112；S:src/qi/deployment/robot_runtime.py:213-257,360-386,440-484]

## 3. checkpoint 身份、结构和 sparse 差异

|任务|本机真实路径（均非指向另一个路径的解析结果）|字节数|SHA256（本次流式计算）|校准服务器预期路径|
|---|---|---:|---|---|
|pack|`/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/checkpoint/pack_object/step_015000.pt`|12041813397|`2b5da8ac40a2701f65f1f72cd613713047271deafbd2da929b3ed8ab75b17a49`|`/root/autodl-tmp/fastwam/joint_pack/step_015000.pt`|
|stack|`/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/checkpoint/stack/step_015000.pt`|12041813397|`a86b2223d4b0f1fcb079532a5a2898a5ab9f529da3143994721cdaa32d29fc9f`|`/root/autodl-tmp/fastwam/joint_stack/step_015000.pt`|

本地 README 分别包含 ModelScope `Mikuz12/joint_pack` 和 `Mikuz12/joint_stack` 下载标识；configuration.json 仅 `{"task":"other"}`。`.mv` 分别为 `Revision:master,CreatedAt:1790045404`、`Revision:master,CreatedAt:1790062830`；`.msc` 的 step 文件 Revision 字符串分别为 `846eae5506c0e93283e97908c2e926116a3b5610`、`22089af262e6e4fb016afbfcb5bc23e036ac570c`。这支持本地文件的下载来源，但不是远端当前内容的 SHA256 证明。**同名、同大小不代表相同文件；接收方下载后必须核对上表 SHA256，当前跨服务器一致性未确认。**

两个安全加载结果：顶层键完全为 `mot, step, torch_dtype, proprio_encoder`；step=15000，torch_dtype 字符串为 torch.bfloat16；mot 各有1649个 tensor、6,020,688,078 个元素；proprio_encoder 有2个 tensor、61,440个元素（weight [4096,14]、bias [4096]）。两文件所有 tensor 键/shape/dtype 完全匹配，均 BF16，但权重内容不相同。没有 config、stats、optimizer 或 spectral/plan/mask 名称的 mot tensor。原始 checkpoint **不含 VAE/text 权重**。不能从没有 spectral keys 推断其训练数据/监督方式。

加载链：`WAMPolicyClient.__init__` → instantiate(cfg.model, model_dtype=BF16, device=cuda:0) → `create_fastwam_joint` → `FastWAMJoint.from_wan22_pretrained` → video DiT+ActionDiT+VAE 预训练构建 → `FastWAM.__init__` → `model.load_checkpoint`。后者加载 mot 使用 strict=False、proprio 使用 strict=True；当前服务不检查 mot missing/unexpected，后续 bridge 应核对完整键集合，不能以“不报错”证明完全匹配。[S:src/qi/deployment/wam_policy_client.py:108-119；S:src/qi/runtime.py:160-256；S:src/qi/models/wan22/fastwam.py:507-710,2185-2204]

架构：30层 video +30层 action，均24 heads、head_dim128；video hidden3072/FFN14336，action hidden1024/FFN4096；text4096，proprio Linear14→4096，action14；video patch(1,2,2)、in/out48、VAE38空间降采样16、时间4。模型 video tokens=3×(24/2)×(20/2)=360，action tokens=32，joint attention mask [392,392]。第一潜帧固定，first_frame_causal 视频 mask只禁止首帧query看未来帧，未来帧之间仍可双向注意，并非全时序三角因果；action 看所有 video 和全部 action，video 不看 action。[S:src/qi/models/wan22/fastwam_joint.py:12-49；wan_video_vae.py:1355-1384；wan_video_dit.py:493-641]

|每层目标 Linear|video weight [out,in]|action weight [out,in]|bias|
|---|---|---|---|
|self_attn q/k/v|[3072,3072]|[3072,1024]|有|
|self_attn o|[3072,3072]|[1024,3072]|有|
|cross_attn q/k/v|[3072,3072]|[3072,1024]|有|
|cross_attn o|[3072,3072]|[1024,3072]|有|
|ffn.0|[14336,3072]|[4096,1024]|有|
|ffn.2|[3072,14336]|[1024,4096]|有|

已对checkpoint所有30层×2专家逐项核对上述10类weight和bias均存在，共600 sites，并逐项验证K%256=0、N%8=0。每专家30×10，总600 sites，符合 Q topology；所有输入宽度满足 W4A4 256/W4A8 32对齐，输出满足8对齐。cross k/v 消费各专家 text_embedding 后的 hidden，并非原始4096 context。checkpoint 名如 `mot` 内的 `mixtures.video.blocks.0.self_attn.q.weight`；Q 目标名为 `video_expert.blocks.0.self_attn.q`。不要直接把 checkpoint 顶层映射误认成完整模型 state_dict。[Q:src/fastwam_steerquant/topology.py:12-76]

与本地 P 普通 Joint 对照：S 把 package fastwam 改为 qi，Joint 增加 expert-cache 参数、version1 sparse controller/persistent packed-video 分支；base FastWAM/MoT 加入 action/video cache、Graph/compile 管理器、稀疏 QKV/FFN 及旧 qi.quantization 特判。这些入口存在但本次服务路径没有安装 sparse controller、observer 或 quantization，没有调用 base FastWAM 的另一套 accelerated infer_action。当前 Joint 每次仍执行完整10次 joint `_predict_joint_noise`，并固定首潜帧。不能因为目录名 fastwam-sparse/分支名 k56 就宣称启用了稀疏推理；也不能声称与远端“标准版”所有源码完全等价。[S:fastwam_joint.py:153-343；fastwam.py:1261-1378；mot.py:968起，均位于 src/qi/models/wan22]

SDPA 的数学/flash/高效 attention 内核由 PyTorch/设备实际选择，本次未运行 GPU，未确认具体内核。use_gradient_checkpointing/mot_checkpoint_mixed_attn 是训练侧设置，不等于 torch.compile 或 CUDA Graph。

## 4. 客户端→服务端→模型→动作的完整链路

|阶段|实际函数、证据|输入 → 输出与行为|
|---|---|---|
|相机|S:robot_runtime.py:71-166（src/qi/deployment 下）`DualArmRosOperator`|三路 ROS Image → RGB uint8 [H,W,3]；预期480×640，运行实测未做|
|状态|同文件:260-290 `get_status_and_state`|两臂关节与夹爪状态 → np.float32 [14]；读取失败可能保留零值|
|observation|同文件:398-433 `_producer_loop`|`{"images":{三相机名:array},"state":array[14],"prompt":任务短句}`|
|RPC|S:wam_remote_policy_client.py:24-46|`{"op":"infer","obs":...}`，multiprocessing.connection，Python pickle/numpy 数组，不是 HTTP/JSON|
|服务器|S:wam_policy_server.py:12-70|单 policy、单连接串行处理；infer = update_observation + get_action|
|预处理|S:wam_policy_client.py:124-176,198-335|RGB拼接/[-1,1]，proprio训练normalizer，缓存文本；均在服务器|
|模型|S:fastwam_joint.py:153-343|infer_action(**kwargs) → `{"action": CPU torch.float32[32,14]}`，归一化动作|
|后处理|S:wam_policy_client.py:175-176,316-335|一次 normalizer.backward → CPU np.float32[32,14] 物理坐标动作|
|返回|S:wam_policy_server.py:58-60|`{"ok":True,"action":np.ndarray}`|
|RTC|S:rtc_buffer.py:51-101；robot_runtime.py:424-484|先截取[:32]入队，按控制时间融合重叠预测，选出np.float32[14]|
|执行|S:robot_runtime.py:315-332,462-484|安全检查后每次一步；move_js 6关节+move_gripper_m，双臂顺序发出|

### 4.1 相机、图像与时间对齐

相机顺序固定为 `cam_high, cam_left_wrist, cam_right_wrist`；topics 分别 `/camera_h/color/image_raw`、`/camera_l/color/image_raw`、`/camera_r/color/image_raw`。depth YAML 有路径但本入口不订阅/不使用。每路 deque maxlen2000，ROS订阅 queue_size1000。get_frame 取各路最新时间的最小值，再各路丢弃更早帧，取首个不早于该时间的帧；没有精确相等时间戳或最大误差约束。状态随后独立读取，没有与相机时间戳做硬同步；跨臂状态也非原子快照。一次请求只提供一帧/相机和一组当前 proprio，**没有历史窗口**。33训练观测步和9视频帧不是客户端要发送的历史图像数量。

RGB处理：先每路转 CHW float32 /255、BILINEAR antialias resize到240×320；top再resize到256×320，两个wrist各128×160；wrist横拼、再接在top下方，得到[3,384,320]；ResizeSmallestSideAspectPreserving+CenterCrop目标384×320（当前拼接尺寸已相同）；2*x−1；加batch后转模型device/BF16。没有额外动态裁剪ROI，不能误用横排3相机拼接。ROS bgr8会翻转为RGB；HDF5没有编码元数据，不能仅凭数组shape认定原始通道顺序已确认。

### 4.2 proprio、prompt、完整 infer kwargs

服务只归一化state，不在客户端归一化。维度顺序和动作相同，见第5节。`normalize_proprio` 返回 CPU float32 [14]，Joint接受[14]并转[1,14] BF16；Q helper目前要求[1,14]，bridge需显式unsqueeze一次。

实际格式化文本模板：

```text
A video recorded from a robot's point of view executing the following instruction: {task}
```

`task` 为第2节短句；客户端每次传短句，`obs.prompt` 优先于服务配置默认。cache文件名是完整格式化文本UTF-8 SHA256 + `.t5_len128.wan22ti2v5b.pt`。cache context [128,4096] BF16、mask [128] bool；加载后先将padding context置零，再把所有mask置True（必须复现，不要改成保留padding mask）。模型收到 `prompt=None` 与cached context/mask，扩成[1,128,4096]/[1,128]并追加proprio token → [1,129,4096]/[1,129]。

实际调用 kwargs（这是现有语义示意，不是新增实现）：

```python
model.infer_action(
    prompt=None, input_image=image, proprio=proprio,
    context=context, context_mask=context_mask,
    action_horizon=32, num_inference_steps=10,
    seed=42, rand_device="cpu", num_video_frames=9,
)
# 未传参数使用 Joint 默认：negative_prompt=None, text_cfg_scale=1.0,
# sigma_shift=None, tiled=False, expert_cache=False,
# expert_cache_reuse_steps=1, warmup_steps=1, cooldown_steps=1。
```

Joint此实现虽然签名含text_cfg_scale/negative_prompt，infer_action循环只计算正条件预测，没有CFG第二次forward；适配仍强制1以保持协议，不能开放任意CFG值。每请求分别创建两个 CPU generator，以同一个42初始化；video/action都先采float32，再转BF16；不能改成共享一个连续推进的generator。首帧VAE编码后覆盖video latent首帧，每步scheduler更新后重新固定首帧，action每步更新。噪声/去噪没有跨请求续采样状态。

### 4.3 RPC、状态和 RTC 的准确含义

服务构造一次模型，Listener接受一个连接后处理到EOF/shutdown才接受下一个。协议还支持update_observation/get_action/reset/shutdown；非法request/op或推理异常返回ok=false/error，异常附traceback，客户端抛RuntimeError，无自动重试/超时逻辑。断线不会自动清掉policy.observation；reset只清observation，文本cache保留。因此必须保持单会话串行隔离，不可让多个连接交错共享get_action状态。

客户端一个producer线程同步RPC，主线程消费动作；RTC buffer有lock和generation计数。producer记录请求发起前的control cursor，返回块挂在这个cursor下；控制继续前进时，已经过去的前缀会按 `chunk[current_time-cursor]` 跳过，没有显式测量后重新规划。所有覆盖当前t的chunks按cursor升序，以 `w_i=exp(0.2*i)/sum(exp(0.2*j))` 融合，较新预测更重。没有向服务器传上一次动作、动作前缀mask、delay或约束；没有条件续采样/inpainting。因此校准使用每条obs独立、完整10步去噪，RTC本身不增加模型输入，但闭环速度变化会改变观测和融合分布。

每返回最多32步，**不是每次同步执行完32步才发下一次请求**。控制循环每拿到一个可用动作、通过检查才t+=1；等待动作不推进t。max_steps限制这些已选控制步（dry-run也计数），不是请求数、不是去噪步数，也不是严格墙钟超时。名义50Hz；robot.move内部另sleep(0.02)，实际周期受RPC等待、日志、SDK与Rate影响；32/50=0.64s仅producer的预算比较值，不能当已达到的实时性能。

控制安全检查：mean(abs(action-last_action)) >1.5停止；mean(abs(current_state-last_action)) >0.3停止；不同单位直接混合求均值，未发现这里有显式isfinite拒绝。夹爪执行时clip到[0,0.1]m；动作反归一化没有[-5,5]截断。move_initial独立于execute_actions，先于按Enter确认，50次插值，每次双臂move_js+夹爪命令+20ms睡眠。connect始终连接两臂并设置/使能。初次离线验证不要用这个机器人入口。

## 5. 训练配置、归一化与 action_scale

### 5.1 来源界限

现用两个推理配置不是经过checkpoint内元数据验证的完整训练配置。

- pack 推理配置注释称由 `P/checkpoints/10step_joint/config.yaml` 导出：fastwam→qi，集群路径改成本地，移除spectral_plan。但该参考配置本身是GRSpectralTargetDataset、spectral_plan.enabled=true、max_steps=10的smoke配置，**不能据此认定它就是15000步 Joint checkpoint 的实际训练配置**。它指向perfect_lerobot、episode_split=all、val比例0、ratio4、33帧。参考 stats 与当前 pack stats 的字节SHA256相同，这仅证明文件相同，不证明属于当前ckpt训练。
- stack 当前stats旁的 `P/checkpoints/fastwam_stack_3_cups/config.yaml` 是 `fastwam.runtime.create_fastwam`（非Joint）的旧配置，数据目录 `./data/stack_3_cups/perfect_lerobot`，val比例0.01，引用另一旧run统计。不能等同于 `m1ku2/stack_3_cups_gen` 的新Joint训练配置；YAML原注释也要求核对。
- 本报告附录完整提供现用推理配置和训练参考配置；后者明确标为参考，不能用来填“已证实训练划分”。两个ckpt的实际训练run config、训练代码revision、split manifest、权重与stats绑定关系均未确认。

### 5.2 动作与状态坐标

|0-based维度|含义|当前生产执行解释|
|---|---|---|
|0,1,2,3,4,5|左 waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate|绝对关节目标角，rad|
|6|左夹爪|绝对开口目标，m，执行clip[0,0.1]|
|7,8,9,10,11,12|右对应6关节|绝对关节目标角，rad|
|13|右夹爪|绝对开口目标，m，执行clip[0,0.1]|

state读取同序关节反馈+夹爪值，不用qvel/effort/base_action，无末端位姿或增量变换。推理配置 `action_state_transforms=null`、`delta_action_dim_mask`未提供。HDF5转换器同帧复制action和qpos，不做单位/增量换算。SDK move_js文档明确rad；move_gripper_m明确m；get_gripper_status通用文档可为width(m)/angle(deg)，runtime直接取value未检查mode，因此运行反馈模式须另行核对。训练文件本身无每维单位属性，训练动作的来源语义仍需采集/训练run确认。[P:scripts/data_process/convert_real_hdf5_to_lerobot.py:12-33,151-200；S:robot_runtime.py:260-332；SDK:protocols/can_protocol/drivers/piper/default/driver.py:290-305,1020-1045；SDK:protocols/can_protocol/drivers/effector/agx_gripper/default/driver.py:188-216,368-390]

### 5.3 当前实际公式

两个任务都 `norm_default_mode=z-score, norm_exception_mode=null, use_stepwise_action_norm=false`，只读 `action.default.global_*` 与 `state.default.global_*`；任务通过服务实例对应stats文件选择，没有按obs.task动态选择多任务统计。JSON数组转CPU float32。

设训练统计mu/std为μ_d,s_d，epsilon=1e−8：

```text
scale_d = 1/(s_d + 1e-8)
offset_d = -μ_d/(s_d + 1e-8)
forward(x)_d = clip(x_d*scale_d + offset_d, -5, 5)
backward(y)_d = (y_d-offset_d)/scale_d
             = y_d*(s_d+1e-8)+μ_d
```

action训练target和state都经过forward clipping；服务state只forward一次，action只backward一次。服务没有调用完整 `FastWAMProcessor.postprocess`；该方法末尾按num_obs_steps−1裁动作（本配置为32），直接把生产[32,14]输出交给它会错误地切空，必须复用 `denormalize_actions` 而非照搬训练postprocess。后者不再clip，模型可能输出超出训练范围的值；机器人夹爪clip是独立的执行阶段非线性，不能用于模型坐标action_scale。

零std维度：代码分母+1e−8，没有自动设为1；精确常量在训练分布归一化std=0，而Q validate_action_scale要求严格>0。若以后遇到常量维度，需要明确约定敏感度权重忽略或正floor及其含义，不可静默除零；本次两套14维std均正。min/max/q01/q99不是当前模式的scale输入，但保留在附录用于范围核验。其他mode的range<1e−4时scale=1、offset=−input_min；所有mode最终也clamp[-5,5]。[S:src/qi/datasets/lerobot/utils/normalizer.py:19-153]

### 5.4 SteerQuant action_scale 的结论

要求的是 **std(训练分布中的归一化动作target)**，不是物理坐标raw_std，不是反归一化倍率，也不是预测动作std。

无clipping影响时：`std_norm_d = raw_std_d/(normalizer_std_d+1e-8)`。若两者为同一训练统计口径，则接近1。minmax情况下才是 `2*raw_std/(max-min)`，本次不是minmax。

**pack：条件性可计算。** 当前stats中全部action端点经z-score最大绝对值4.480604<5，state最大4.714143<5；若该文件确为此ckpt完整训练分布统计，clipping在记录范围内不改变分布，14维std_norm可由s/(s+eps)计算（附录给数值）。但ckpt-stats绑定未确认，不能将条件结果说成已认证训练尺度。

**stack：当前统计不足以精确计算完整向量。** action维10（右forearm_roll）的最大绝对z=5.043140，维12（右wrist_rotate）=5.231760，确有超出clipping范围的统计端点；state维12=5.026520、13=5.235354也超界。mean/std/min/max/quantile不能恢复尾部概率与二阶矩，必须提供训练时clip后动作std，或在**已确认完整训练划分**上按同一权重口径离线计算：`z=clip((a-μ)/(s+eps),-5,5)` 后求std。其余action维可条件性用线性公式，但整个向量目前仍未确认。附录线性候选值不能当最终action_scale。

统计计算代码 `BaseLerobotDataset.get_dataset_stats` 对每episode/块收集mean/var，再假设每个minibatch样本数相等做mean聚合；不是保证按全数据帧数加权的pooled std，quantile也有min/max聚合语义。因此重算时要明确训练使用的统计口径和校准loss希望的训练分布权重，记录ddof/episode权重/padding；不要把3条轨迹的估计替代训练统计。[S:base_lerobot_dataset.py:280-402]

## 6. HDF5与25 observation记录

### 6.1 已查到的文件与契约

本地仅候选 pack `W/cobot_magic/collect_data/data/pack_3_objects_plus/perfect/episode_0.hdf5` 存在；episode35/70未找到，两个常见stack_3_cups_gen目录不存在。pack文件attrs compress=False/sim=False；action/qpos/qvel/effort均[894,14] float32，base_action[894,2] float32；三路image均[894,480,640,3] uint8，非压缩字节。没有timestamp/task/instruction字段。旁边episode_0.source.json记录HF repo `m1ku2/pack_3_objects_plus`、perfect路径、size2472130624、SHA256 `c0c1dad8280a0853b2dcb4d4b31b0e9fc2ba0574781266a2a74e3f49881e744b`；这个HDF哈希是已有来源记录，**本次未重新计算**。

另只读检查了本地stack历史**评测**HDF5：`P/src/dashboard/data/robot_eval_incremental/sessions/stack_cups_original_20260814_193507_3f1a87/raw/episode_001/episode.hdf5`，409帧，三图像uint8[409,480,640,3]，action/qpos等float64[409,14]，有isSuccess和metadata/json。它不属于题述训练候选，不能混入校准，更不能用来证明stack训练HDF字段完全相同。

真实训练 loader消费LeRobot格式，不直接消费HDF5。可用转换器 `P/scripts/data_process/convert_real_hdf5_to_lerobot.py` 严格要求上述未压缩像素shape；不支持直接把JPEG字节当图像。它按episode文件数字排序、转换后连续重新编号，记录raw_file_name；同一frame_idx取action、qpos、3图像，生成timestamp=frame_idx/fps，fps CLI默认10，未确认真实训练转换命令；图片经PIL JPEG(默认quality95)再编码MP4(默认h264)，没有RGB/BGR翻转。HDF5原始颜色编码未存属性；当前采集代码使用cv_bridge passthrough，也不能证明旧训练数据就是RGB。校准服务器须用训练转换记录/可视颜色核对通道，不要按经验盲目cvtColor。[P转换器:94-205,208-240；W:cobot_magic/collect_data/camera_collect_data.py:199-213]

### 6.2 索引与输入构造

`BaseLerobotDataset` 强制past_obs_size=past_action_size=0：不是向过去取33帧，而是从t开始取未来窗口。参考num_frames33、global_sample_stride1、ratio4时：

```text
images/state原窗口: t..t+32，共33
训练video抽样: t,t+4,...,t+32，共9 → [3,9,384,320]
action窗口: t..t+31，共32 → [32,14]
proprio输出: state窗口去最后1帧 → [32,14]
model.build_inputs实际条件proprio: proprio[:,0,:]，即t时刻
生产infer_action实际条件: 仅t时刻3图像 + qpos[t] + 任务文本
未来video/action: 不作为生产推理条件，去噪从噪声开始
```

训练loader末端越界按episode边界复制并标is_pad；skip_padding_as_possible可重采，且RobotVideoDataset.__getitem__遇异常可能改取随机样本。因此固定校准manifest不能让异常静默替换ID，应直接读取指定帧并校验，失败即报错。仅构造生产obs边界是0≤t<N，且三图像/qpos同帧均有效；若要同时做无padding训练窗口/GT对照则0≤t≤N−33（stride1）。pack N894则完整窗口t≤861。无需历史帧或未来帧才能做生产obs，但阶段标注/GT评估可能需要周围帧。[S:base_lerobot_dataset.py:20-110,207-279；S:lerobot/lerobot_dataset.py:659-673；S:robot_video_dataset.py:46-63,115-234,270-279；S:fastwam.py:811-905，均在前述目录]

instructions来自转换时--instruction、LeRobot任务表和可选override_instruction；生产固定为任务YAML短句。统计来自任务对应训练stats，不从所选HDF5现场拟合。完整训练processor与现用推理processor不能互换：现用推理配置train_transforms/val_transforms=null，直接调用完整preprocess可能遍历None失败；当前服务只用normalizer和独立图像函数。

### 6.3 perfect/retry与train/val证据

参考pack配置仅perfect_lerobot、all、val0；参考旧stack配置仅perfect_lerobot、val0.01。没有可绑定新Joint checkpoint的证据说明retry是否参加训练、过滤/采样权重、成功标签规则。Base loader默认val0.05，按每dataset总episodes，用np.random.default_rng(seed=42) shuffle，再前floor(N*(1-val))训练、其余验证；val<1e−6取全部。**这只是代码算法，不是当前checkpoint已确认split。** 不能从episode编号0/35/70断定train；HDF编号可能与LeRobot编号不同，统计num_episodes(pack71/stack70)也不等于split manifest。

所需证明：实际resolved训练config、数据revision、转换清单raw_file_name→LeRobot episode_index、meta/info.json/episodes.jsonl/tasks.jsonl、训练过滤和split manifest。若作者另用GRSpectralTargetDataset/holdout规则，必须取其实际实现，不能只复算Base随机划分。

### 6.4 每任务9+8+8的建议采样（尚未执行）

先确认3条episode属于训练集，再逐条离线浏览关键帧/夹爪和关节变化，标注三个物体/杯子的操作阶段；时间等分仅作提议，人工确认事件覆盖。pack覆盖初始、每物体接近/抓取/搬运/入箱释放、末端撤离；stack覆盖底杯定位、下一杯抓取/对齐/落杯释放、第三杯同类阶段和最终撤离。不要预设物体处理顺序或“阶段发生在固定百分比”。

9条轨迹配额可为初始1 + 三个操作段各2 + 最终/撤离2；8条为初始1 + 三段各2 + 最终1。对stack若实际只有两次叠放操作，将“底杯/初始定位、第二杯、第三杯”作为三个操作段。3条episode的采样位置交错覆盖接触前、夹爪闭合、运输、精对齐、释放；同一段2帧不要都取静止或相邻重复画面。某episode缺某阶段则调整帧位或替换候选episode，记录原因，不为凑25复制样本。只挑成功静止终态会错过动作敏感阶段；也不要未经训练证明加入retry。

建议record自包含：

```python
{
  "observation_id": "pack:<dataset_revision>:perfect/episode_0.hdf5:frame_123",
  "sample_index": 0,  # 每任务文件内唯一，0..24
  "task": "pack", "episode": "perfect/episode_0.hdf5", "frame_index": 123,
  "stage": "<人工确认阶段>", "dataset_revision": "<确认值>",
  "source_sha256": "<该episode实际哈希>", "split_evidence": "<manifest标识>",
  "images": {"cam_high": uint8_tensor_HWC, "cam_left_wrist": uint8_tensor_HWC,
             "cam_right_wrist": uint8_tensor_HWC},
  "state": float32_tensor_14, "prompt": "<任务原始短句>",
  "image_encoding": "RGB",  # 只有核对来源后才能填
}
```

用Q的save_observation_records保存CPU Tensor/primitive；NumPy先转Tensor。可额外保存stats/config/cache哈希，明确raw_record尚未归一化。pack/stack各25，分别复用同一任务的25条对比W4A8/W4A4，不混任务统计；RHT与未旋转的校准cache不可共用。

## 7. SteerQuant接口的真实映射与阻塞

已完整阅读 Q:MIGRATION_WORKFLOW.md 第3节、adapters/base.py、adapters/fastwam.py，并追踪state/policy/deployment/topology/streams/rollout_graph。下表为**建议实施**，目前没有真实robot_bridge。

|接口|推荐复用|需要改造/约束|
|---|---|---|
|`load_model(*, device)`|S工厂架构 + checkpoint mot/proprio加载，eval|尊重cpu/cuda；CPU校准导出不得默认cuda；只安全weights_only加载；核对missing/unexpected；使用同一VAE|
|`build_model(model_config, *, device)`|WanVideoDiT(config)、ActionDiT(config)、MoT、WanVideoVAE38、FastWAMJoint底层构造|新增纯结构路径，禁止调用from_pretrained/下载器/原始source_checkpoint；处理别名及meta障碍|
|`infer_kwargs(model, record, *, seed)`|preprocess_real_images、normalize_proprio、format_prompt/load_cached_context|raw RGB/HWC和物理state仅处理一次；cached context，prompt=None；proprio[1,14]；显式steps10、sigma_shift=None（effective5）、CFG1、horizon32、frames9、cpu RNG|
|`prepare_inputs(model, record, *, seed)`|Joint infer_action:197-261；复用VAE编码与_append_proprio_to_context|Q helper不能原样使用：它强制encode_prompt，而本机text_encoder/tokenizer=None；实现cached context分支，保持两个独立同seed generator|
|`action_scale()`|本报告公式、完整训练统计|pack仅条件性可得；stack clipping后二阶矩缺失；输出FP32[14]正有限|
|`dataset_records()`|已核对HDF5契约/训练manifest；Q observations.py|确定split、颜色、stage和实际frame index；不得依赖读取时变化的路径/随机替换样本|
|`finalize_model(model, *, device)`|_sync_vae_scale_device及缓存初始化语义|修复model.device/torch_dtype、RoPE、VAE普通tensor和runtime cache；不得重新载BF16 target权重|

### 7.1 prepare_inputs的五个tensor

按生产输入图像[1,3,384,320]、当前proprio和缓存文本生成，所有浮点tensor为模型BF16/device（mask为bool）：

|返回键|shape|来源|
|---|---|---|
|latents_video|[1,48,3,24,20]|CPU float32 randn(seed)→BF16；首潜帧覆盖为first|
|latents_action|[1,32,14]|另一个CPU generator以同seed randn→BF16|
|first_frame_latents|[1,48,1,24,20]|真实拼接图经WanVideoVAE38.encode|
|context|[1,129,4096]|缓存128token padding置零/mask全true + proprio_encoder一token|
|context_mask|[1,129]|全true bool（按现有缓存语义）|

VAE空间因子是16，**不要套用WanVideoVAE旧类的8**；每潜帧120 tokens，不是480。图片编码与proprio投影不求梯度用于输入准备，但之后可微去噪应在正常grad context中接收普通可参与autograd的tensor；`_encode_input_image_latents_tensor` 是inference_mode修饰，进入可微阶段前在inference_mode外clone输出，避免inference tensor保存反传限制。不能把torch.inference_mode装饰的_predict_joint_noise直接包成可微core。

### 7.2 去噪API差异：必须补core

S `_predict_joint_noise` 存在，但有 `@torch.inference_mode()`；S及本地P没有 `_joint_denoise_core`。Q `state.differentiable_joint_denoise`直接调用这个缺失方法；Q LiveJointDiTGraph也安装到该方法。因此当前不是“只写dataset adapter就能校准”。

推荐后续在S兼容模型实现中抽取无decorator的：

```python
_joint_denoise_core(
    *, latents_video, latents_action, timestep_video, timestep_action,
    context, context_mask, attention_mask,
    fuse_vae_embedding_in_latents, action_condition=None,
) -> (pred_video, pred_action)
```

按当前dense `_predict_joint_noise`的video.pre_dit/action.pre_dit→mot→post_dit原样实现，不开启sparse/expert_cache；生产 `_predict_joint_noise` 负责构mask/转调core，以便Q tracker每步记录一次。Q differentiable loop首帧固定用cat，避免破坏梯度；生产仍按原语义更新。两边scheduler的build_inference_schedule/step签名一致：sigma_i=5*u_i/(1+4*u_i)，u=linspace(1,0,11)，t=1000*sigma[:-1]，delta=sigma[1:]−sigma[:-1]，sample←sample+prediction*delta，最后转BF16调度tensor。该兼容性是代码核对；可微/生产输出一致性及VJP尚未运行验证。

验收时逐步比对同seed初噪、context、mask、每步latents和最终action；验证非零且有限梯度，eval状态保持与生产一致，不能为梯度随意model.train引入其他行为。Q tracker包装_predict_joint_noise，要求实际调用数正好10；直接绕过predict跑core会绕过tracker，开启sparse/cache跳步也会破坏每call校准。

### 7.3 构造器、meta、外部资产

当前from_pretrained无纯结构开关：`skip_dit_load_from_pretrain=true`只跳过video/action的预训练DiT，不跳过VAE load/download；`load_text_encoder=false`才不读T5/tokenizer。真正builder应直接用下层模块类，包含VAE完整结构，text_encoder/tokenizer设None。部署export应包含VAE与非目标FP权重，所以直接INT4部署不必重读原始step、Wan视频DiT分片或ActionDiT backbone；前提是完成严格加载覆盖验证。[S:helpers/loader.py:151-228；action_dit.py:112-167；fastwam.py:591-684]

meta障碍：scheduler构造在 `_precompute_training_weight_stats` 中对tensor做`.item()`，若全局torch.device(meta)会失败；RoPE预计算、VAE mean/std/scale为普通tensor属性而非buffer；FastWAM.__init__末尾会self.to(device)并sync VAE scale。Q load_deployment在调用finalize之前先model.to(device)，若普通属性已变成meta，不能依赖之后finalize才恢复它。builder必须在移动前保证这些常量可物化（CPU显式构建scheduler/常量，或调整构造/加载时序）。建议第一阶段采用**纯结构CPU构建**，在搬GPU前替换target Linear；不得用CPU构建为借口重载原始全模型checkpoint。CPU临时内存需求仍需评估，未作meta/CPU整模构建试验。

必须处理非state资产：video_expert.freqs(3个complex RoPE表)、action_expert.freqs、VAE mean/std/scale、scheduler对象/常数、device/torch_dtype、model_paths，_graph_mgr/_vae_encode_graph_mgr/_prefill_graph_mgr/_action_body_fn/_module_*和_video_predit_cache/文本cache。builder将runtime cache置空，finalize设置设备并重建有值常量；注册nonpersistent buffers Q可导出，但普通Python属性中的tensor不会自动导出。

直接INT4后仍需：模型Python源码与结构配置、任务normalization JSON、两任务缓存text embedding（若不内嵌deployment）、图像预处理和运行依赖、Q原生kernel对应硬件构建。当前缓存文本路径无需tokenizer/T5外部文件；只有未来允许新prompt并启用text_encoder时才需要T5/tokenizer资产。VAE权重必须存在于deployment，若未打入则仍须单独带VAE，不能假定原始Joint ckpt含VAE。

当前服务器没有加载完后全局.half()/.bfloat16()；构造器内部有.to(dtype=BF16)，发生在BF16加载阶段。量化加载后只做device移动，严禁全局dtype转换使Q weight_scales/input_scale/gamma/clipping/RHT signs从FP32变成BF16。

### 7.4 重复模块注册：当前packed导出/严格加载的实际障碍

FastWAM同时注册 `video_expert`/`action_expert`、`mot.mixtures.video/action`，又设 `dit=mot`。同一Linear出现在三组state_dict路径。已用同样注册方式的2×2小module验证state_dict确实包含三组路径；不需要加载6B模型即可确认PyTorch行为。[S:fastwam.py:530-535；mot.py:48]

Q deployment.py:47-48只排除 `video_expert...`/`action_expert...` 的weight/bias，保留其他别名下的BF16 target；:164再strict=True assign加载。替换共享block中的Linear后别名路径同样指向WAMQuantLinear，但payload别名仍是旧shape和dtype、缺新scale buffer，会造成strict加载missing/shape冲突，且导出不满足无完整BF16 target的承诺。不是简单模型参数名改前缀可解决。

建议修复Q exporter/loader的alias处理：枚举named_modules(remove_duplicate=False)建立所有目标alias→canonical映射，移除所有alias的原FP target，用一致packed buffers填充严格加载所需路径或实现canonical加载+显式alias恢复；保证所有引用是同一替换模块、pack数据没有隐含BF16副本、strict检查完整且alias冲突时报错。也可采用只注册一次的兼容包装模型，但需证明mot/checkpoint映射等价。不能用strict=False掩盖。为真实alias结构补小型CPU导出/加载回归，随后检查完整deployment中所有target alias都是uint8[N,K/2]且scales FP32。

## 8. 最小真机部署接入方案（仅建议，未实施）

后续修改的最小逻辑集合：

1. **Q/local/robot_bridge.py + 每任务JSON**（待新增）：七个callback、cached context输入、真实records、统计/provenance与纯结构builder；local在gitignore中，交接必须显式传输或决定移入正式模块。
2. **S/src/qi/models/wan22/fastwam.py**（或等价随bridge交付的兼容子类）：抽取可微core并让生产predict也走它；纯结构工厂可留bridge，不必修改所有构造器。若实现为子类须明确新的_target_并保证权重名/别名一致。
3. **Q/src/fastwam_steerquant/deployment.py**：修复上述alias导出/加载；为meta路径处理时序可按需加hook，CPU优先方案避免扩大改动。
4. **S/src/qi/deployment/wam_policy_client.py**：WAMPolicyConfig新增backend/deployment/adapter_config/construct_device/cuda_graph等明确字段；在__init__:112之前选择BF16或Q QuantizedPolicy，量化模式跳过instantiate+load_checkpoint整模GPU加载。在get_action统一返回归一化action后只反归一化一次。
5. **服务端量化配置YAML**（建议新增四份或两份任务配置可切模式），保持原基线配置供对照。所有新字段目前from_mapping不会消费；现在直接加字段不会自动生效。

`serve_policy.py`与`wam_policy_server.py`可保留现有签名并仍构造WAMPolicyClient；不必修改deploy_real_rtc_wam.py、robot_runtime.py、rtc_buffer.py、相机topic或SDK控制。客户端使用原任务配置和原命令可以不变，前提服务端继续相同地址/凭证/请求响应、同32×14物理动作和prompt，且server配置路径变化不影响client配置。不要为INT4改action horizon、归一化或夹爪逻辑。

量化分支建议 `QuantizedPolicy.infer(raw_record, seed=42)`（adapter独自预处理）→output["action"]→原denormalize_actions；不要先用WAMPolicyClient处理成normalized再让adapter重复处理。另一路可集中共用infer_kwargs，但必须明确输入类型边界。

QuantizedPolicy有非阻塞lock，只允许同实例串行，当前server正好串行。client producer/consumer并发不意味着模型要并发；reset与正在进行的RPC在线程退出join(timeout=2)后可能共用连接，已有生命周期竞争不可因量化另起推理线程而放大。保持tracker每请求reset、num_calls10、sigma_shift元数据一致（None+两个scheduler5，或全链统一显式5，不可混用），先cuda_graph=false。Graph只有core适配完成、两个不同obs的eager/graph结果核对后才启用。RTC不同推理延迟带来的prefix跳过/融合变化属于端到端评估因素。

**待实施后使用，当前不支持量化配置，下面未执行：**

```bash
# 在真机 S 根目录；新YAML及bridge/packed loader完成后才可用。
conda activate fastwam
PYTHONPATH=src:<STEERQUANT_SRC>:<BRIDGE_DIR> python -u scripts/serve_policy.py \
  --config <PACK_OR_STACK_W4A8_OR_W4A4_DEPLOY_YAML> \
  --host 127.0.0.1 --port 8765 --authkey '<LOCAL_AUTHKEY>'
```

凭证经现有本机安全方式供给，不记录到交接Git/报告；server CLI与client policy_server YAML一致性需本机核对。原客户端的--move-to-initial仍有运动语义，不将它列为smoke命令。

## 9. 后续校准、离线验收与对比协议

在校准服务器 `/root/autodl-tmp/Fastwam-steerquant` 完成上面阻塞项后，pack/stack分别配置、分别W4A8与W4A4+RHT，共4份deployment.pt。P=4指每obs的敏感度probes，不是episode数、batch size或去噪步数；num_calls=10。W4A4 RHT必须在敏感度/激活收集之前安装，导出从原始未旋转CPU模型和该校准结果合成，部署不再次旋转权重。[Q:MIGRATION_WORKFLOW.md:第3、5节]

下列是**待适配并补齐数据后使用**的流程示意，本次没有执行；现有Q脚本真实存在，但local配置/bridge尚不存在：

```bash
# 每task对应独立local JSON和25 obs；每种模式独立输出，先preflight和小规模smoke。
bash bash/calibrate_w4a8.sh --config local/pack.json \
  --observations outputs/pack/observations25.pt --output-dir outputs/pack/w4a8 --num-probes 4
bash bash/calibrate_w4a4_rht.sh --config local/pack.json \
  --observations outputs/pack/observations25.pt --output-dir outputs/pack/w4a4_rht --num-probes 4
# stack换成自己的config/observations/output；不能复用pack统计或cache。
```

导出同样是**待实施并完成校准后使用**；当前Q脚本在CPU加载原始未旋转模型，并核对calibration.json中的config identity，搬迁后绝对路径/mtime改变可能触发校验失败，应先在校准服务器原配置下导出，再传部署文件；不要直接篡改identity绕过：

```bash
python scripts/export_deployment.py --config local/pack.json \
  --calibration outputs/pack/w4a8/calibration.pt \
  --output outputs/pack/w4a8/deployment.pt
# 分别使用对应任务config和对应模式calibration，再生成另外三份：
# outputs/pack/w4a4_rht/deployment.pt
# outputs/stack/w4a8/deployment.pt
# outputs/stack/w4a4_rht/deployment.pt
```

依据：Q:scripts/export_deployment.py:12-31，Q:adapters/base.py:41-53。后续adapter不应为了部署factory初始化就尝试访问原source checkpoint；packed推理的外部资产清单与校准source资产清单应分清。

初次验证入口应完全不import/构造机器人runtime：

- CPU：record字段/颜色/shape、训练split、归一化数值、图像拼接与prompt hash校验；Q已有 `examples/cpu_smoke.py` 可作合成量化测试，不证明FastWAM接入完成。
- GPU离线（后续）：已有 `S/scripts/smoke_joint_pack3_local.py` 读取HDF5→WAMPolicyClient，不连接ROS/CAN/机器人/网络服务；支持--config换stack。它会加载模型、推理并写output_actions，**本次没运行**；需要增强数值断言，因为当前只是打印finite并最终打印PASS。量化接入WAMPolicyClient后同入口可用于三模式。
- 再写/扩展纯离线Q replay入口：取两个不同真实record，生产BF16与可微去噪逐步一致；同一原始obs/seed对比BF16、W4A8、W4A4 normalized与物理动作RMSE/max/逐维误差，所有shape与finite严格检查；直接packed load须在原checkpoint不可访问、无预训练下载条件下仍成功，且所有target alias无BF16残留。
- `smoke_joint_pack3_remote.py` 不加--live-cameras时不连接机器人，但仍连接正在运行的服务；不属于第一阶段“完全离线”。加--live-cameras会订阅ROS，不能用于本次检查。overlay和W/inference_launch.sh也不属于离线入口。

速度/显存协议：固定同一任务stats、seed、obs集合、10步/32horizon/9frames、device、dtype、Graph开关；记录GPU型号/软件/kernel revision与预热次数。分别测加载时间/峰值显存、预热后纯infer_action与端到端RPC（包含预处理/反归一化/传输），CUDA计时要同步或events，报告p50/p95、allocated/reserved峰值、吞吐及32步块延迟；不要把不同工作范围的数字直接做speedup。BF16和INT4先同eager，再独立比较Graph；kernel编译/初次Graph捕获不混入稳态延迟，另外报告启动成本。

SR仅来自后续实际闭环trial，不可用离线动作相似度或合成smoke生成SR。事先约定pack/stack成功判定、固定/随机化初态、最大墙钟时间与控制步数、人工干预/安全停止计失败规则；三模式相同RTC/频率/chunk/安全阈值，交错运行多次，记录成功数/总数、置信区间、控制实际频率/等待/过期动作比例。当前成功标准、trial数、基线SR、目标GPU可用显存和实测速率均未确认。

## 10. 必须单独传输的文件和资产

报告足够定义接口，但不能替代完整模型代码。建议小型源码包使用相对根目录布局并附SHA256 manifest；不要包含凭证。以下均为后续传输清单，本次没复制或发送：

|文件集合|原因|
|---|---|
|S完整 `src/qi/models/wan22/`、`src/qi/runtime.py`、`src/qi/utils/`、`src/qi/dit_cache/`、`src/qi/quantization/`及包__init__|模型、VAE、MoT、scheduler、helpers和旧quantization的import依赖；少量snippet不足以准确构建。若只复制dense文件仍须满足import依赖，不能只带Joint一个文件|
|S完整 `src/qi/deployment/`、serve_policy.py、deploy_real_rtc_wam.py、smoke_joint_pack3_local.py|保留本地RGB与frames改动、复现RPC/后处理/RTC；审阅硬件代码不等于在校准服务器执行|
|S完整 `src/qi/datasets/`（至少processor/normalizer/merger/image/dataset_utils及LeRobot loader依赖）|正确图像和归一化契约、split/采样行为；避免从记忆重写|
|S两个部署YAML（authkey占位）+两个推理YAML，pyproject.toml/依赖版本说明|均需脱敏重映射绝对资产路径；推理配置未跟踪，单纯git clone会遗漏|
|P三个参考config：10step_joint、k4_disjoint_pack_3_objecst、fastwam_stack_3_cups；实际训练resolved config和manifest（待取得）|参考与真实训练明确区分；真实训练来源不可由推理YAML代替|
|pack/stack现用完整dataset_stats.json|包含global/stepwise，不能只复制round后的表；pack与10step stats相同，可用一份并记录映射|
|P HDF5转换器完整文件、真实转换命令/数据revision/LeRobot meta文件/采样manifest|确认颜色、fps、编号映射、split、perfect/retry、stage|
|两个约1MiB context cache .pt（附录哈希）|当前无text encoder的必要外部资产；固定prompt文件不能遗漏|
|Q上述revision完整源码、后续bridge/local JSON、alias修复和core适配补丁、kernel依赖lock|两服务器采用同一量化与模型代码版本；local ignored须显式打包|

大文件另行传输/下载：两原始checkpoint（校准所需，哈希必须一致）、同版VAE `P/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`（原Joint不含），本机VAE文件存在、大小2,818,839,170字节（未做内容哈希）；BF16原构造路径需要Wan视频DiT safetensors和 `P/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`（本机存在、2,042,025,883字节，未做内容哈希）。若安全严格核对证明Joint mot全覆盖，校准builder也可跳过video/action预训练读取，但本次未实施/验证；VAE仍必需。最后回传4个deployment.pt+各自SHA256、model_config、来源ckpt/stats/cache/obs manifest和校准日志。部署pt必须含VAE与非目标权重，且源BF16target完全移除；不要宣称所有模型参数都INT4（这里只量化600个指定Linear）。

SDK/ROS/camera workspace仅真机部署需要，校准server离线适配无需连接它们。`/opt/ros/noetic/setup.bash`、相机workspace和roslaunch命令仅定位背景，本次未source/运行。

## 11. 状态与阻塞清单

|类别|项目|下一步证据/验收|
|---|---|---|
|已确认|两任务实际服务配置、checkpoint大小/本地SHA256、键shape/dtype、dense10步Joint、缓存文本、32×14动作、RGB预处理、RTC行为|本报告与附录|
|已确认|本地pack episode0实际schema；现用两套完整normalizer数值；stack clipping超界；缺core与alias导出问题|只读文件/CPU检查|
|未确认|另一服务器ModelScope下载文件与本机是否同一内容|下载完成后核对SHA256，不仅文件名|
|阻塞校准来源认证|真正训练config/revision、stats绑定、训练split、perfect/retry过滤/权重、候选0/35/70归属|权重作者训练记录及数据manifest|
|阻塞精确stack action_scale|真实训练clip后标准差、数据加权口径|提供归一化后二阶矩或按完整训练split流式计算|
|阻塞可运行校准|真实bridge、cached context prepare、可微core、梯度/生产等价验证|按第7节实现并离线验收|
|阻塞packed导出/部署|纯结构builder、alias处理、VAE/普通tensor物化|CPU结构与完整packed严格加载，无BF16target GPU加载|
|未确认/校准资源约束|校准GPU/kernel支持和峰值内存|目标机真实预检与小规模smoke；本次不安装不编译|
|未确认|训练HDF颜色编码、实际fps及转码分布、其余5条候选schema和stage覆盖|下载后只读检查/人工阶段标注|
|仅阻塞真机部署/有效SR|服务加载分支与新配置、真实相机encoding/状态同步、夹爪反馈mode、目标kernel性能、试验成功标准与基线SR|先纯离线三模式对比，再独立授权的真机trial|

当前“已确认”均为文件/静态调用证据，不含新模型的成功抓取/堆叠、延迟、显存或SR验证。

---

## 附录A：脱敏部署YAML与关键推理配置（完整内容）

下面保留原绝对路径作为来源证据，接收方按第1节前缀和assets清单重映射；路径不能直接拿到另一台服务器执行。行号是源文件行号，authkey行统一占位。

### S:configs/real_deploy_joint_pack3.yaml（源行 1–73）

```yaml
   1 | # RTC-style FastWAMJoint real-robot deployment config for the
   2 | # pack-3-objects-into-container task (baseline dense inference, no K56).
   3 | #
   4 | # The top-level `wam` section is what WAMPolicyConfig.from_mapping reads;
   5 | # the remaining sections follow the same schema as configs/real_bot.yaml.
   6 |
   7 | wam:
   8 |   ckpt: "/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/checkpoint/pack_object/step_015000.pt"
   9 |   dataset_stats: "/home/agilex/World_Action_Model/physical_WM/checkpoints/k4_disjoint_pack_3_objecst/dataset_stats.json"
  10 |   train_config: "/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/configs/train_config_joint_pack3_infer.yaml"
  11 |   device: "cuda:0"
  12 |   mixed_precision: "bf16"
  13 |   prompt: "Put the three objects on the table into the container."
  14 |   # sha256(formatted prompt) == ad085ca9... — cache file already present in this dir
  15 |   context_cache_dir: "/home/agilex/World_Action_Model/physical_WM/data/text_embeds_cache/pack_3_objects_plus"
  16 |   use_text_encoder: false
  17 |   action_horizon: 32
  18 |   num_inference_steps: 10
  19 |   seed: 42
  20 |   rand_device: "cpu"
  21 |   # FastWAMJoint.infer_action requires this (T % 4 == 1; 3 video latents => 9);
  22 |   # plain FastWAM models must leave it unset (null).
  23 |   num_video_frames: 9
  24 |
  25 | policy_server:
  26 |   host: "127.0.0.1"
  27 |   port: 8765
  28 |   authkey: "<LOCAL_AUTHKEY>"
  29 |
  30 | runtime:
  31 |   arm_mode: "dual"
  32 |   action_chunk_size: 32
  33 |   exp_weight_factor: 0.2            # pack-3 precedent (real_deploy_k4_pack3.yaml)
  34 |   rtc_debug: false
  35 |   rospy_rate: 50
  36 |   execute_actions: False            # dry-run default; opt in via --execute-actions
  37 |   move_to_initial: False            # opt in via --move-to-initial
  38 |   max_steps: null
  39 |   save_rollout: false
  40 |   output_dir: "./output_actions/joint_pack3"
  41 |
  42 | robot:
  43 |   left_channel: "can_left"
  44 |   right_channel: "can_right"
  45 |   bitrate: 1000000
  46 |   speed_pct: 15
  47 |   max_linear_vel: 0.5
  48 |   max_angular_vel: 0.1
  49 |   max_linear_acc: 0.1
  50 |   max_angular_acc: 0.05
  51 |   action_safety_threshold: 1.5
  52 |   state_safety_threshold: 0.3
  53 |   # pack-3 task initial poses (from physical_WM configs/real_deploy_k4_pack3.yaml)
  54 |   left_init_position: [-0.5185, 1.2758, -0.8406, 0.0315, 1.0872, 0.1147, 0.0593]
  55 |   right_init_position: [0.4801, 1.1595, -0.7810, 0.0920, 1.1126, -0.0512, 0.0515]
  56 |
  57 | cameras:
  58 |   use_front: true
  59 |   use_wrist: true
  60 |   use_left_wrist: true
  61 |   use_right_wrist: true
  62 |   use_depth_image: false
  63 |   front_camera_name: "cam_high"
  64 |   left_wrist_camera_name: "cam_left_wrist"
  65 |   right_wrist_camera_name: "cam_right_wrist"
  66 |
  67 | ros_topics_dual:
  68 |   img_front_topic: "/camera_h/color/image_raw"
  69 |   img_left_topic: "/camera_l/color/image_raw"
  70 |   img_right_topic: "/camera_r/color/image_raw"
  71 |   img_front_depth_topic: "/camera_h/depth/image_raw"
  72 |   img_left_depth_topic: "/camera_l/depth/image_raw"
  73 |   img_right_depth_topic: "/camera_r/depth/image_raw"
```

### S:configs/real_deploy_joint_stack3.yaml（源行 1–75）

```yaml
   1 | # RTC-style FastWAMJoint real-robot deployment config for the
   2 | # stack-three-cups task (baseline dense inference, no K56).
   3 | #
   4 | # The top-level `wam` section is what WAMPolicyConfig.from_mapping reads;
   5 | # the remaining sections follow the same schema as configs/real_bot.yaml.
   6 |
   7 | wam:
   8 |   ckpt: "/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/checkpoint/stack/step_015000.pt"
   9 |   # Existing stack task statistics; the new checkpoint includes no normalization metadata.
  10 |   # Confirm these match its training run before evaluating task performance.
  11 |   dataset_stats: "/home/agilex/World_Action_Model/physical_WM/checkpoints/fastwam_stack_3_cups/dataset_stats.json"
  12 |   train_config: "/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/configs/train_config_joint_stack3_infer.yaml"
  13 |   device: "cuda:0"
  14 |   mixed_precision: "bf16"
  15 |   prompt: "Stack the three cups into a small tower."
  16 |   # Cached prompt verified by SHA-256; use the existing stack task embeddings.
  17 |   context_cache_dir: "/home/agilex/World_Action_Model/physical_WM/data/text_embeds_cache/stack_3_cups_gen"
  18 |   use_text_encoder: false
  19 |   action_horizon: 32
  20 |   num_inference_steps: 10
  21 |   seed: 42
  22 |   rand_device: "cpu"
  23 |   # FastWAMJoint.infer_action requires this (T % 4 == 1; 3 video latents => 9);
  24 |   # plain FastWAM models must leave it unset (null).
  25 |   num_video_frames: 9
  26 |
  27 | policy_server:
  28 |   host: "127.0.0.1"
  29 |   port: 8765
  30 |   authkey: "<LOCAL_AUTHKEY>"
  31 |
  32 | runtime:
  33 |   arm_mode: "dual"
  34 |   action_chunk_size: 32
  35 |   exp_weight_factor: 0.2            # initial RTC setting; not task-tuned
  36 |   rtc_debug: false
  37 |   rospy_rate: 50
  38 |   execute_actions: False            # dry-run default; opt in via --execute-actions
  39 |   move_to_initial: False            # opt in via --move-to-initial
  40 |   max_steps: null
  41 |   save_rollout: false
  42 |   output_dir: "./output_actions/joint_stack3"
  43 |
  44 | robot:
  45 |   left_channel: "can_left"
  46 |   right_channel: "can_right"
  47 |   bitrate: 1000000
  48 |   speed_pct: 15
  49 |   max_linear_vel: 0.5
  50 |   max_angular_vel: 0.1
  51 |   max_linear_acc: 0.1
  52 |   max_angular_acc: 0.05
  53 |   action_safety_threshold: 1.5
  54 |   state_safety_threshold: 0.3
  55 |   # Stack task poses from dashboard/robot_eval_pipeline/task_profiles.py.
  56 |   left_init_position: [-0.75764, 1.03441, -0.74778, 0.03059, 0.93736, -0.07892, 0.06808]
  57 |   right_init_position: [0.63639, 1.00319, -0.71547, 0.01034, 1.02991, 0.19950, 0.06644]
  58 |
  59 | cameras:
  60 |   use_front: true
  61 |   use_wrist: true
  62 |   use_left_wrist: true
  63 |   use_right_wrist: true
  64 |   use_depth_image: false
  65 |   front_camera_name: "cam_high"
  66 |   left_wrist_camera_name: "cam_left_wrist"
  67 |   right_wrist_camera_name: "cam_right_wrist"
  68 |
  69 | ros_topics_dual:
  70 |   img_front_topic: "/camera_h/color/image_raw"
  71 |   img_left_topic: "/camera_l/color/image_raw"
  72 |   img_right_topic: "/camera_r/color/image_raw"
  73 |   img_front_depth_topic: "/camera_h/depth/image_raw"
  74 |   img_left_depth_topic: "/camera_l/depth/image_raw"
  75 |   img_right_depth_topic: "/camera_r/depth/image_raw"
```

### S:configs/train_config_joint_pack3_infer.yaml（源行 1–104）

```yaml
   1 | # Minimal inference-only train config for the FastWAMJoint pack-3 checkpoint
   2 | # (checkpoint/pack_object/step_015000.pt).
   3 | #
   4 | # WAMPolicyClient only consumes `model` and `data.train.processor` plus
   5 | # `data.train.context_len`; dataset/scheduler training sections are
   6 | # intentionally omitted. Derived from the cluster-trained
   7 | # checkpoints/10step_joint/config.yaml with:
   8 | #   - fastwam.* targets renamed to qi.* (this repo's package namespace)
   9 | #   - /project/peilab/... cluster paths replaced by local checkpoint paths
  10 | #   - spectral_plan dropped (checkpoint payload carries no spectral keys;
  11 | #     qi.runtime.create_fastwam_joint does not accept that argument)
  12 |
  13 | data:
  14 |   train:
  15 |     context_len: 128
  16 |     processor:
  17 |       _target_: qi.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
  18 |       shape_meta:
  19 |         images:
  20 |           - key: cam_high
  21 |             raw_shape: [3, 480, 640]
  22 |             shape: [3, 240, 320]
  23 |           - key: cam_left_wrist
  24 |             raw_shape: [3, 480, 640]
  25 |             shape: [3, 240, 320]
  26 |           - key: cam_right_wrist
  27 |             raw_shape: [3, 480, 640]
  28 |             shape: [3, 240, 320]
  29 |         action:
  30 |           - key: default
  31 |             raw_shape: 14
  32 |             shape: 14
  33 |         state:
  34 |           - key: default
  35 |             raw_shape: 14
  36 |             shape: 14
  37 |       num_obs_steps: 33
  38 |       num_output_cameras: 3
  39 |       action_output_dim: 14
  40 |       proprio_output_dim: 14
  41 |       action_state_transforms: null
  42 |       use_stepwise_action_norm: false
  43 |       norm_default_mode: z-score
  44 |       norm_exception_mode: null
  45 |       action_state_merger:
  46 |         _target_: qi.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
  47 |       train_transforms: null
  48 |       val_transforms: null
  49 |
  50 | model:
  51 |   _target_: qi.runtime.create_fastwam_joint
  52 |   model_id: /home/agilex/World_Action_Model/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B
  53 |   tokenizer_model_id: /home/agilex/World_Action_Model/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B
  54 |   tokenizer_max_len: 128
  55 |   load_text_encoder: false
  56 |   proprio_dim: 14
  57 |   redirect_common_files: false
  58 |   mot_checkpoint_mixed_attn: true
  59 |   action_dit_pretrained_path: /home/agilex/World_Action_Model/physical_WM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
  60 |   skip_dit_load_from_pretrain: false
  61 |   video_dit_config:
  62 |     has_image_input: false
  63 |     patch_size: [1, 2, 2]
  64 |     in_dim: 48
  65 |     hidden_dim: 3072
  66 |     ffn_dim: 14336
  67 |     freq_dim: 256
  68 |     text_dim: 4096
  69 |     out_dim: 48
  70 |     num_heads: 24
  71 |     attn_head_dim: 128
  72 |     num_layers: 30
  73 |     eps: 1.0e-06
  74 |     seperated_timestep: true
  75 |     require_clip_embedding: false
  76 |     require_vae_embedding: false
  77 |     fuse_vae_embedding_in_latents: true
  78 |     use_gradient_checkpointing: true
  79 |     video_attention_mask_mode: first_frame_causal
  80 |     action_conditioned: false
  81 |     action_dim: 14
  82 |     action_group_causal_mask_mode: group_diagonal
  83 |   action_dit_config:
  84 |     action_dim: 14
  85 |     hidden_dim: 1024
  86 |     ffn_dim: 4096
  87 |     num_heads: 24
  88 |     attn_head_dim: 128
  89 |     num_layers: 30
  90 |     text_dim: 4096
  91 |     freq_dim: 256
  92 |     eps: 1.0e-06
  93 |     use_gradient_checkpointing: true
  94 |   video_scheduler:
  95 |     train_shift: 5.0
  96 |     infer_shift: 5.0
  97 |     num_train_timesteps: 1000
  98 |   action_scheduler:
  99 |     train_shift: 5.0
 100 |     infer_shift: 5.0
 101 |     num_train_timesteps: 1000
 102 |   loss:
 103 |     lambda_action: 1.0
 104 |     lambda_video: 1.0
```

### S:configs/train_config_joint_stack3_infer.yaml（源行 1–97）

```yaml
   1 | # FastWAMJoint stack-3 inference architecture.
   2 | # The stack step_015000.pt tensor names/shapes/dtypes match the validated
   3 | # pack-3 Joint checkpoint. Reuse its inference architecture and processor;
   4 | # task-specific normalization and text conditioning live in the deploy YAML.
   5 |
   6 | data:
   7 |   train:
   8 |     context_len: 128
   9 |     processor:
  10 |       _target_: qi.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
  11 |       shape_meta:
  12 |         images:
  13 |           - key: cam_high
  14 |             raw_shape: [3, 480, 640]
  15 |             shape: [3, 240, 320]
  16 |           - key: cam_left_wrist
  17 |             raw_shape: [3, 480, 640]
  18 |             shape: [3, 240, 320]
  19 |           - key: cam_right_wrist
  20 |             raw_shape: [3, 480, 640]
  21 |             shape: [3, 240, 320]
  22 |         action:
  23 |           - key: default
  24 |             raw_shape: 14
  25 |             shape: 14
  26 |         state:
  27 |           - key: default
  28 |             raw_shape: 14
  29 |             shape: 14
  30 |       num_obs_steps: 33
  31 |       num_output_cameras: 3
  32 |       action_output_dim: 14
  33 |       proprio_output_dim: 14
  34 |       action_state_transforms: null
  35 |       use_stepwise_action_norm: false
  36 |       norm_default_mode: z-score
  37 |       norm_exception_mode: null
  38 |       action_state_merger:
  39 |         _target_: qi.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
  40 |       train_transforms: null
  41 |       val_transforms: null
  42 |
  43 | model:
  44 |   _target_: qi.runtime.create_fastwam_joint
  45 |   model_id: /home/agilex/World_Action_Model/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B
  46 |   tokenizer_model_id: /home/agilex/World_Action_Model/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B
  47 |   tokenizer_max_len: 128
  48 |   load_text_encoder: false
  49 |   proprio_dim: 14
  50 |   redirect_common_files: false
  51 |   mot_checkpoint_mixed_attn: true
  52 |   action_dit_pretrained_path: /home/agilex/World_Action_Model/physical_WM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
  53 |   skip_dit_load_from_pretrain: false
  54 |   video_dit_config:
  55 |     has_image_input: false
  56 |     patch_size: [1, 2, 2]
  57 |     in_dim: 48
  58 |     hidden_dim: 3072
  59 |     ffn_dim: 14336
  60 |     freq_dim: 256
  61 |     text_dim: 4096
  62 |     out_dim: 48
  63 |     num_heads: 24
  64 |     attn_head_dim: 128
  65 |     num_layers: 30
  66 |     eps: 1.0e-06
  67 |     seperated_timestep: true
  68 |     require_clip_embedding: false
  69 |     require_vae_embedding: false
  70 |     fuse_vae_embedding_in_latents: true
  71 |     use_gradient_checkpointing: true
  72 |     video_attention_mask_mode: first_frame_causal
  73 |     action_conditioned: false
  74 |     action_dim: 14
  75 |     action_group_causal_mask_mode: group_diagonal
  76 |   action_dit_config:
  77 |     action_dim: 14
  78 |     hidden_dim: 1024
  79 |     ffn_dim: 4096
  80 |     num_heads: 24
  81 |     attn_head_dim: 128
  82 |     num_layers: 30
  83 |     text_dim: 4096
  84 |     freq_dim: 256
  85 |     eps: 1.0e-06
  86 |     use_gradient_checkpointing: true
  87 |   video_scheduler:
  88 |     train_shift: 5.0
  89 |     infer_shift: 5.0
  90 |     num_train_timesteps: 1000
  91 |   action_scheduler:
  92 |     train_shift: 5.0
  93 |     infer_shift: 5.0
  94 |     num_train_timesteps: 1000
  95 |   loss:
  96 |     lambda_action: 1.0
  97 |     lambda_video: 1.0
```

## 附录B：现用统计的精确global数值与尺度候选

完整文件应单独传输。下列JSON为原文件global_*精度值（不是重新估计），全部顺序按第5节14维。global_mean/global_std实际参与当前z-score；min/max与quantile用于范围核查。stepwise_*未用于当前归一化，未在报告重复展开。

### pack：P/checkpoints/k4_disjoint_pack_3_objecst/dataset_stats.json

SHA256 `30561edb2897b9784fbf13f4e42ca9897e141b4ebd60119d047d255758aaa6bf`；源行1起，精确key定位如下。

```json
{
  "action": {
    "default": {
      "global_min": [
        -0.7464985251426697,
        0.5993409752845764,
        -1.998977780342102,
        -0.3060724139213562,
        0.66883784532547,
        -0.3932400941848755,
        -0.005799999926239252,
        0.09676186740398407,
        0.854965329170227,
        -2.2580385208129883,
        -0.20426924526691437,
        0.6320310235023499,
        -0.32138824462890625,
        -0.0035000001080334187
      ],
      "global_max": [
        -0.12095669656991959,
        2.1204752922058105,
        -0.6353628039360046,
        0.4025202989578247,
        1.2315987348556519,
        0.42500561475753784,
        0.08070000261068344,
        0.6975506544113159,
        2.4567954540252686,
        -0.588595449924469,
        0.2802029848098755,
        1.2326802015304565,
        0.41314369440078735,
        0.0737999975681305
      ],
      "global_q01": [
        -0.7464985251426697,
        0.5993409752845764,
        -1.998977780342102,
        -0.29364916682243347,
        0.6701217293739319,
        -0.33328506350517273,
        -0.004999999888241291,
        0.09676186740398407,
        0.854965329170227,
        -2.2580385208129883,
        -0.20426924526691437,
        0.6320310235023499,
        -0.3184707462787628,
        -0.003100000089034438
      ],
      "global_q99": [
        -0.12095669656991959,
        2.1204752922058105,
        -0.6353628039360046,
        0.3861112892627716,
        1.2315987348556519,
        0.42500561475753784,
        0.08070000261068344,
        0.6975506544113159,
        2.452434539794922,
        -0.588595449924469,
        0.2802029848098755,
        1.2320871353149414,
        0.41314369440078735,
        0.07349999994039536
      ],
      "global_mean": [
        -0.4663515090942383,
        1.4284223318099976,
        -1.0373151302337646,
        0.023459576070308685,
        1.0967848300933838,
        0.07584848999977112,
        0.05168206989765167,
        0.43530532717704773,
        1.5468578338623047,
        -1.3134510517120361,
        0.06507649272680283,
        1.1303578615188599,
        0.014401243068277836,
        0.03794793039560318
      ],
      "global_std": [
        0.13589313626289368,
        0.3008541166782379,
        0.33417728543281555,
        0.08989857137203217,
        0.11472103744745255,
        0.1234917938709259,
        0.0226027462631464,
        0.14602059125900269,
        0.47843459248542786,
        0.5246521830558777,
        0.07529964298009872,
        0.11121866106987,
        0.09801463782787323,
        0.022958192974328995
      ]
    }
  },
  "state": {
    "default": {
      "global_min": [
        -0.7464985251426697,
        0.5993409752845764,
        -1.998977780342102,
        -0.3060724139213562,
        0.66883784532547,
        -0.3932400941848755,
        0.011900000274181366,
        0.09676186740398407,
        0.854965329170227,
        -2.2580385208129883,
        -0.20426924526691437,
        0.6320310235023499,
        -0.32138824462890625,
        0.009200000204145908
      ],
      "global_max": [
        -0.12095669656991959,
        2.1204752922058105,
        -0.6369327902793884,
        0.4025202989578247,
        1.2315987348556519,
        0.42500561475753784,
        0.08030000329017639,
        0.6975506544113159,
        2.4567954540252686,
        -0.588595449924469,
        0.2802029848098755,
        1.2326802015304565,
        0.41314369440078735,
        0.0729999989271164
      ],
      "global_q01": [
        -0.7464985251426697,
        0.5993409752845764,
        -1.998977780342102,
        -0.29364916682243347,
        0.6701217293739319,
        -0.33328506350517273,
        0.01955999992787838,
        0.09676186740398407,
        0.854965329170227,
        -2.2580385208129883,
        -0.20426924526691437,
        0.6320310235023499,
        -0.3184707462787628,
        0.009999999776482582
      ],
      "global_q99": [
        -0.12095669656991959,
        2.1204752922058105,
        -0.6723440885543823,
        0.3861112892627716,
        1.2315987348556519,
        0.42500561475753784,
        0.08030000329017639,
        0.6975506544113159,
        2.452434539794922,
        -0.588595449924469,
        0.2802029848098755,
        1.2320871353149414,
        0.41314369440078735,
        0.0729999989271164
      ],
      "global_mean": [
        -0.4662761986255646,
        1.4342573881149292,
        -1.0384119749069214,
        0.023112313821911812,
        1.0949195623397827,
        0.07966329157352448,
        0.057319723069667816,
        0.4354167878627777,
        1.5485789775848389,
        -1.3128596544265747,
        0.06472509354352951,
        1.1300076246261597,
        0.01282151136547327,
        0.04389742761850357
      ],
      "global_std": [
        0.13601118326187134,
        0.29423788189888,
        0.33339396119117737,
        0.08999106287956238,
        0.1145971342921257,
        0.12161029875278473,
        0.009634767659008503,
        0.14596612751483917,
        0.4766601622104645,
        0.5251285433769226,
        0.07525888085365295,
        0.11160385608673096,
        0.09842052310705185,
        0.01438628789037466
      ]
    }
  },
  "num_episodes": 71,
  "num_transition": 65917
}
```

**未认证线性候选 std_norm**（pack依赖stats来源匹配；stack维10/12 clipping使其不能作为最终向量）：

```json
[0.9999999264127716, 0.9999999667613002, 0.9999999700757648, 0.9999998887635395, 0.9999999128320373, 0.999999919022966, 0.9999995575760812, 0.9999999315165127, 0.9999999790985018, 0.9999999809397536, 0.9999998671972632, 0.9999999100870394, 0.9999998979744332, 0.9999995644258388]
```

### stack：P/checkpoints/fastwam_stack_3_cups/dataset_stats.json

SHA256 `68c662dba501b1e110e09896238bfd0ae1d15c226e2c05683958bba915f7e0cb`；源行1起，精确key定位如下。

```json
{
  "action": {
    "default": {
      "global_min": [
        -0.8828931450843811,
        0.9373359084129333,
        -1.9212995767593384,
        -0.3218592405319214,
        0.6550222039222717,
        -0.4159870743751526,
        -0.005499999970197678,
        0.24754780530929565,
        0.9272881746292114,
        -1.5516438484191895,
        -0.22863851487636566,
        0.7242923378944397,
        -0.23350538313388824,
        -0.00139999995008111
      ],
      "global_max": [
        -0.37870922684669495,
        2.2329540252685547,
        -0.47913435101509094,
        0.37951165437698364,
        1.2274645566940308,
        0.5199881792068481,
        0.1014999970793724,
        0.847080647945404,
        2.112311363220215,
        -0.6061789989471436,
        0.3627479672431946,
        1.2246735095977783,
        0.5712561011314392,
        0.08609999716281891
      ],
      "global_q01": [
        -0.8828931450843811,
        0.9375626444816589,
        -1.8937995433807373,
        -0.3218592405319214,
        0.6550222039222717,
        -0.4159870743751526,
        -0.004999999888241291,
        0.24754780530929565,
        0.9272881746292114,
        -1.5497075319290161,
        -0.22656267881393433,
        0.7335899472236633,
        -0.23350538313388824,
        -0.00139999995008111
      ],
      "global_q99": [
        -0.37870922684669495,
        2.2329540252685547,
        -0.47913435101509094,
        0.37951165437698364,
        1.226609706878662,
        0.5199881792068481,
        0.1007160022854805,
        0.847080647945404,
        2.111142635345459,
        -0.6061789989471436,
        0.3627479672431946,
        1.2246735095977783,
        0.5712561011314392,
        0.08349999785423279
      ],
      "global_mean": [
        -0.6894557476043701,
        1.5656464099884033,
        -1.1716187000274658,
        0.004325909074395895,
        1.0453052520751953,
        0.03831141069531441,
        0.05651909485459328,
        0.5989316701889038,
        1.387856125831604,
        -0.8538634777069092,
        0.15540240705013275,
        1.0240881443023682,
        0.05600779876112938,
        0.06151824817061424
      ],
      "global_std": [
        0.12918493151664734,
        0.45512017607688904,
        0.4152217507362366,
        0.09500952810049057,
        0.14504417777061462,
        0.16094309091567993,
        0.027720002457499504,
        0.10946279019117355,
        0.2674810588359833,
        0.23620723187923431,
        0.07615114748477936,
        0.07938586175441742,
        0.09848468005657196,
        0.012749281711876392
      ]
    }
  },
  "state": {
    "default": {
      "global_min": [
        -0.8828931450843811,
        0.9373359084129333,
        -1.9212995767593384,
        -0.3218592405319214,
        0.6550222039222717,
        -0.4159870743751526,
        0.0034000000450760126,
        0.24754780530929565,
        0.9272881746292114,
        -1.5516438484191895,
        -0.22863851487636566,
        0.7242923378944397,
        -0.23350538313388824,
        0.024000000208616257
      ],
      "global_max": [
        -0.37870922684669495,
        2.2329540252685547,
        -0.47913435101509094,
        0.37951165437698364,
        1.2274645566940308,
        0.5199881792068481,
        0.09950000047683716,
        0.847080647945404,
        2.112311363220215,
        -0.6061789989471436,
        0.3627479672431946,
        1.2246735095977783,
        0.5712561011314392,
        0.08340000361204147
      ],
      "global_q01": [
        -0.8828931450843811,
        0.9375626444816589,
        -1.8937995433807373,
        -0.3218592405319214,
        0.6550222039222717,
        -0.4159870743751526,
        0.034258000552654266,
        0.24754780530929565,
        0.9272881746292114,
        -1.5497075319290161,
        -0.22656267881393433,
        0.7335899472236633,
        -0.23350538313388824,
        0.044881999492645264
      ],
      "global_q99": [
        -0.37870922684669495,
        2.2329540252685547,
        -0.47913435101509094,
        0.37951165437698364,
        1.226609706878662,
        0.5199881792068481,
        0.09950000047683716,
        0.847080647945404,
        2.111142635345459,
        -0.6061789989471436,
        0.3627479672431946,
        1.2246735095977783,
        0.5712561011314392,
        0.08340000361204147
      ],
      "global_mean": [
        -0.6888009905815125,
        1.5630414485931396,
        -1.1747010946273804,
        0.003932037856429815,
        1.048166275024414,
        0.03627729415893555,
        0.06569500267505646,
        0.598768413066864,
        1.3817412853240967,
        -0.853697657585144,
        0.15122731029987335,
        1.0238310098648071,
        0.059391409158706665,
        0.06334267556667328
      ],
      "global_std": [
        0.1291448175907135,
        0.457602858543396,
        0.4115191400051117,
        0.0959930568933487,
        0.1418948918581009,
        0.16244736313819885,
        0.012795019894838333,
        0.10964760184288025,
        0.2730674743652344,
        0.2363443374633789,
        0.08008632063865662,
        0.0797310546040535,
        0.1018328070640564,
        0.007514797616750002
      ]
    }
  },
  "num_episodes": 70,
  "num_transition": 46421
}
```

**未认证线性候选 std_norm**（pack依赖stats来源匹配；stack维10/12 clipping使其不能作为最终向量）：

```json
[0.9999999225915972, 0.9999999780277818, 0.9999999759164838, 0.9999998947474096, 0.9999999310554931, 0.9999999378662401, 0.9999996392498013, 0.9999999086447635, 0.9999999626141767, 0.9999999576642955, 0.9999998686822325, 0.9999998740330014, 0.9999998984613752, 0.9999992156427019]
```

## 附录C：必要函数签名和关键代码上下文

以下为审计时原实现摘录，尚未加入任何适配。Q基类/准备helper完整内嵌；大模型模块须完整转移，不能只靠摘录补齐。

### S:src/qi/deployment/wam_policy_client.py（源行 146–211）

```python
 146 |     def get_action(self) -> np.ndarray:
 147 |         """Run WAM inference and return a denormalized ``[T,D]`` action chunk.
 148 |
 149 |         ``D`` is the checkpoint's action dimension (14 for dual-arm, 7 for a
 150 |         single arm); the policy stack is action-dim agnostic.
 151 |         """
 152 |         if self.observation is None:
 153 |             raise RuntimeError("Policy observation is empty. Call update_observation(obs) before get_action().")
 154 |
 155 |         prompt, context, context_mask = self._resolve_text_condition(self.observation.prompt)
 156 |         image = preprocess_real_images(self.observation, self.model.device, self.model.torch_dtype)
 157 |         proprio = normalize_proprio(self.processor, self.observation.state)
 158 |
 159 |         with torch.no_grad():
 160 |             infer_kwargs: dict[str, Any] = dict(
 161 |                 prompt=prompt,
 162 |                 input_image=image,
 163 |                 proprio=proprio,
 164 |                 context=context,
 165 |                 context_mask=context_mask,
 166 |                 action_horizon=self.config.action_horizon,
 167 |                 num_inference_steps=self.config.num_inference_steps,
 168 |                 seed=self.config.seed,
 169 |                 rand_device=self.config.rand_device,
 170 |             )
 171 |             if self.config.num_video_frames is not None:
 172 |                 infer_kwargs["num_video_frames"] = self.config.num_video_frames
 173 |             output = self.model.infer_action(**infer_kwargs)
 174 |
 175 |         actions = denormalize_actions(self.processor, output["action"])
 176 |         return actions.detach().cpu().numpy().astype(np.float32)
 177 |
 178 |     def reset(self) -> None:
 179 |         """Clear per-episode observation state."""
 180 |         self.observation = None
 181 |
 182 |     def _load_config(self) -> DictConfig:
 183 |         if self.config.train_config:
 184 |             cfg = OmegaConf.load(self.config.train_config)
 185 |             if not isinstance(cfg, DictConfig):
 186 |                 raise ValueError(f"Expected DictConfig in {self.config.train_config}, got {type(cfg)}")
 187 |             if self.config.use_text_encoder:
 188 |                 cfg.model.load_text_encoder = True
 189 |             return cfg
 190 |
 191 |         config_dir = str(Path(self.config.config_dir).expanduser().resolve())
 192 |         overrides = [f"task={self.config.task}", f"mixed_precision={self.config.mixed_precision}"]
 193 |         if self.config.use_text_encoder:
 194 |             overrides.append("model.load_text_encoder=true")
 195 |         with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
 196 |             return compose(config_name="train", overrides=overrides)
 197 |
 198 |     def _resolve_text_condition(self, task_prompt: str) -> tuple[str | None, torch.Tensor | None, torch.Tensor | None]:
 199 |         formatted_prompt = format_prompt(task_prompt)
 200 |         if self.config.use_text_encoder:
 201 |             return formatted_prompt, None, None
 202 |
 203 |         if formatted_prompt not in self._context_cache:
 204 |             cache_dir = self.config.context_cache_dir or self.cfg.data.train.get("text_embedding_cache_dir")
 205 |             if cache_dir is None:
 206 |                 raise ValueError("No context cache dir found. Set context_cache_dir or use_text_encoder=true.")
 207 |             context_len = int(self.cfg.data.train.get("context_len", 128))
 208 |             self._context_cache[formatted_prompt] = load_cached_context(cache_dir, formatted_prompt, context_len)
 209 |
 210 |         context, context_mask = self._context_cache[formatted_prompt]
 211 |         return None, context, context_mask
```

### S:src/qi/deployment/wam_policy_client.py（源行 224–335）

```python
 224 | def format_prompt(task_prompt: str) -> str:
 225 |     return DEFAULT_PROMPT.format(task=task_prompt)
 226 |
 227 |
 228 | def load_cached_context(cache_dir: str | Path, formatted_prompt: str, context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
 229 |     cache_dir = Path(cache_dir)
 230 |     cache_hash = hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
 231 |     cache_path = cache_dir / f"{cache_hash}.t5_len{context_len}.wan22ti2v5b.pt"
 232 |     if not cache_path.exists():
 233 |         raise FileNotFoundError(
 234 |             f"Missing text embedding cache: {cache_path}. "
 235 |             "Run scripts/precompute_text_embeds.py for this prompt, or set use_text_encoder=true."
 236 |         )
 237 |     payload = torch.load(cache_path, map_location="cpu")
 238 |     context = payload["context"]
 239 |     context_mask = payload["mask"].bool()
 240 |     if context.ndim != 2 or context_mask.ndim != 1:
 241 |         raise ValueError(f"Invalid cached context shapes in {cache_path}: {context.shape}, {context_mask.shape}")
 242 |     if context.shape[0] != context_len or context_mask.shape[0] != context_len:
 243 |         raise ValueError(
 244 |             f"Cached context length mismatch in {cache_path}: "
 245 |             f"context={context.shape[0]}, mask={context_mask.shape[0]}, expected={context_len}"
 246 |         )
 247 |     context = context.clone()
 248 |     context[~context_mask] = 0.0
 249 |     context_mask = torch.ones_like(context_mask, dtype=torch.bool)
 250 |     return context, context_mask
 251 |
 252 |
 253 | def rgb_to_float_chw(rgb_uint8: np.ndarray) -> torch.Tensor:
 254 |     if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
 255 |         raise ValueError(f"Expected RGB image as [H,W,3], got {rgb_uint8.shape}")
 256 |     rgb_uint8 = np.ascontiguousarray(rgb_uint8).copy()
 257 |     return torch.from_numpy(rgb_uint8).permute(2, 0, 1).to(torch.float32) / 255.0
 258 |
 259 |
 260 | def preprocess_real_images(observation: WAMObservation, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
 261 |     cam_high = transforms_F.resize(
 262 |         rgb_to_float_chw(observation.cam_high),
 263 |         [240, 320],
 264 |         interpolation=transforms_F.InterpolationMode.BILINEAR,
 265 |         antialias=True,
 266 |     )
 267 |     cam_left = transforms_F.resize(
 268 |         rgb_to_float_chw(observation.cam_left_wrist),
 269 |         [240, 320],
 270 |         interpolation=transforms_F.InterpolationMode.BILINEAR,
 271 |         antialias=True,
 272 |     )
 273 |     cam_right = transforms_F.resize(
 274 |         rgb_to_float_chw(observation.cam_right_wrist),
 275 |         [240, 320],
 276 |         interpolation=transforms_F.InterpolationMode.BILINEAR,
 277 |         antialias=True,
 278 |     )
 279 |     cam_high = transforms_F.resize(
 280 |         cam_high,
 281 |         [256, 320],
 282 |         interpolation=transforms_F.InterpolationMode.BILINEAR,
 283 |         antialias=True,
 284 |     )
 285 |     cam_left = transforms_F.resize(
 286 |         cam_left,
 287 |         [128, 160],
 288 |         interpolation=transforms_F.InterpolationMode.BILINEAR,
 289 |         antialias=True,
 290 |     )
 291 |     cam_right = transforms_F.resize(
 292 |         cam_right,
 293 |         [128, 160],
 294 |         interpolation=transforms_F.InterpolationMode.BILINEAR,
 295 |         antialias=True,
 296 |     )
 297 |     bottom = torch.cat([cam_left, cam_right], dim=-1)
 298 |     image = torch.cat([cam_high, bottom], dim=-2)
 299 |     resize = ResizeSmallestSideAspectPreserving(args={"img_w": 320, "img_h": 384})
 300 |     crop = CenterCrop(args={"img_w": 320, "img_h": 384})
 301 |     image = crop(resize(image))
 302 |     image = image.mul(2.0).sub(1.0)
 303 |     return image.unsqueeze(0).to(device=device, dtype=dtype)
 304 |
 305 |
 306 | def normalize_proprio(processor: Any, state: np.ndarray) -> torch.Tensor:
 307 |     state_tensor = torch.as_tensor(state, dtype=torch.float32)
 308 |     expected_dim = int(processor.proprio_output_dim)
 309 |     if state_tensor.numel() != expected_dim:
 310 |         raise ValueError(f"Expected proprio/state dim {expected_dim}, got {state_tensor.numel()}")
 311 |     batch = {"state": {"default": state_tensor.view(1, -1)}}
 312 |     batch = processor.normalizer.forward(batch)
 313 |     return batch["state"]["default"].squeeze(0)
 314 |
 315 |
 316 | def denormalize_actions(processor: Any, action_norm: torch.Tensor) -> torch.Tensor:
 317 |     if action_norm.ndim == 3 and action_norm.shape[0] == 1:
 318 |         action_norm = action_norm.squeeze(0)
 319 |     if action_norm.ndim == 1:
 320 |         action_norm = action_norm.view(1, -1)
 321 |     if action_norm.ndim != 2:
 322 |         raise ValueError(f"Expected normalized action [T,D], got {tuple(action_norm.shape)}")
 323 |     dummy_state = torch.zeros(
 324 |         1,
 325 |         action_norm.shape[0],
 326 |         int(processor.proprio_output_dim),
 327 |         dtype=torch.float32,
 328 |         device=action_norm.device,
 329 |     )
 330 |     batch = {
 331 |         "action": {"default": action_norm.unsqueeze(0).to(torch.float32)},
 332 |         "state": {"default": dummy_state},
 333 |     }
 334 |     batch = processor.normalizer.backward(batch)
 335 |     return batch["action"]["default"].squeeze(0)
```

### S:src/qi/models/wan22/fastwam_joint.py（源行 153–174）

```python
 153 |     @torch.no_grad()
 154 |     def infer_action(
 155 |         self,
 156 |         prompt: Optional[str],
 157 |         input_image: torch.Tensor,
 158 |         action_horizon: int,
 159 |         num_video_frames: int,
 160 |         proprio: Optional[torch.Tensor] = None,
 161 |         context: Optional[torch.Tensor] = None,
 162 |         context_mask: Optional[torch.Tensor] = None,
 163 |         negative_prompt: Optional[str] = None,
 164 |         text_cfg_scale: float = 1.0,
 165 |         num_inference_steps: int = 20,
 166 |         sigma_shift: Optional[float] = None,
 167 |         seed: Optional[int] = None,
 168 |         rand_device: str = "cpu",
 169 |         tiled: bool = False,
 170 |         expert_cache: bool = False,
 171 |         expert_cache_reuse_steps: int = 1,
 172 |         expert_cache_warmup_steps: int = 1,
 173 |         expert_cache_cooldown_steps: int = 1,
 174 |     ) -> dict[str, Any]:
```

### S:src/qi/models/wan22/fastwam_joint.py（源行 197–274）

```python
 197 |         if proprio is not None:
 198 |             if self.proprio_dim is None:
 199 |                 raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
 200 |             if proprio.ndim == 1:
 201 |                 proprio = proprio.unsqueeze(0)
 202 |             elif proprio.ndim == 2 and proprio.shape[0] == 1:
 203 |                 pass
 204 |             else:
 205 |                 raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
 206 |             if proprio.shape[1] != self.proprio_dim:
 207 |                 raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
 208 |             proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
 209 |
 210 |         latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
 211 |         latent_h = height // self.vae.upsampling_factor
 212 |         latent_w = width // self.vae.upsampling_factor
 213 |
 214 |         video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
 215 |         action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
 216 |         latents_video = torch.randn(
 217 |             (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
 218 |             generator=video_generator,
 219 |             device=rand_device,
 220 |             dtype=torch.float32,
 221 |         ).to(device=self.device, dtype=self.torch_dtype)
 222 |         latents_action = torch.randn(
 223 |             (1, action_horizon, self.action_expert.action_dim),
 224 |             generator=action_generator,
 225 |             device=rand_device,
 226 |             dtype=torch.float32,
 227 |         ).to(device=self.device, dtype=self.torch_dtype)
 228 |
 229 |         input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
 230 |         first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
 231 |         latents_video[:, :, 0:1] = first_frame_latents.clone()
 232 |         fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
 233 |
 234 |         use_prompt = prompt is not None
 235 |         use_context = context is not None or context_mask is not None
 236 |         if use_prompt and use_context:
 237 |             raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
 238 |         if not use_prompt and not use_context:
 239 |             raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
 240 |
 241 |         if use_prompt:
 242 |             context, context_mask = self.encode_prompt(prompt)
 243 |         else:
 244 |             if context is None or context_mask is None:
 245 |                 raise ValueError("`context` and `context_mask` must be both provided together.")
 246 |             if context.ndim == 2:
 247 |                 context = context.unsqueeze(0)
 248 |             if context_mask.ndim == 1:
 249 |                 context_mask = context_mask.unsqueeze(0)
 250 |             if context.ndim != 3 or context_mask.ndim != 2:
 251 |                 raise ValueError(
 252 |                     f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
 253 |                 )
 254 |             context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
 255 |             context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
 256 |         if proprio is not None:
 257 |             context, context_mask = self._append_proprio_to_context(
 258 |                 context=context,
 259 |                 context_mask=context_mask,
 260 |                 proprio=proprio,
 261 |             )
 262 |
 263 |         infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
 264 |             num_inference_steps=num_inference_steps,
 265 |             device=self.device,
 266 |             dtype=latents_video.dtype,
 267 |             shift_override=sigma_shift,
 268 |         )
 269 |         infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
 270 |             num_inference_steps=num_inference_steps,
 271 |             device=self.device,
 272 |             dtype=latents_action.dtype,
 273 |             shift_override=sigma_shift,
 274 |         )
```

### S:src/qi/models/wan22/fastwam_joint.py（源行 299–343）

```python
 299 |             if step_idx == 0 or version1_controller is None:
 300 |                 if latents_video is None:
 301 |                     raise RuntimeError("Dense video latents were released before Step0")
 302 |                 pred_video_posi, pred_action_posi = self._predict_joint_noise(
 303 |                     latents_video=latents_video,
 304 |                     latents_action=latents_action,
 305 |                     timestep_video=timestep_video,
 306 |                     timestep_action=timestep_action,
 307 |                     context=context,
 308 |                     context_mask=context_mask,
 309 |                     fuse_vae_embedding_in_latents=fuse_flag,
 310 |                     gt_action=None,
 311 |                     expert_cache_state=action_expert_cache,
 312 |                     expert_cache_step=step_idx,
 313 |                 )
 314 |                 latents_video = self.infer_video_scheduler.step(
 315 |                     pred_video_posi, step_delta_video, latents_video
 316 |                 )
 317 |                 latents_video[:, :, 0:1] = first_frame_latents.clone()
 318 |                 if version1_controller is not None:
 319 |                     packed_latents_video = version1_controller.pack_persistent_video_state(
 320 |                         latents_video
 321 |                     )
 322 |                     latents_video = None
 323 |             else:
 324 |                 if packed_latents_video is None:
 325 |                     raise RuntimeError("Persistent Version1 video patch state is unavailable")
 326 |                 pred_video_posi, pred_action_posi = self._predict_version1_persistent_noise(
 327 |                     packed_latents_video=packed_latents_video,
 328 |                     latents_action=latents_action,
 329 |                     timestep_video=timestep_video,
 330 |                     timestep_action=timestep_action,
 331 |                     step_index=step_idx,
 332 |                 )
 333 |                 packed_latents_video = self.infer_video_scheduler.step(
 334 |                     pred_video_posi,
 335 |                     step_delta_video,
 336 |                     packed_latents_video,
 337 |                 )
 338 |             latents_action = self.infer_action_scheduler.step(pred_action_posi, step_delta_action, latents_action)
 339 |
 340 |         self._log_action_expert_cache(action_expert_cache)
 341 |         return {
 342 |             "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
 343 |         }
```

### S:src/qi/models/wan22/fastwam.py（源行 507–590）

```python
 507 |     def __init__(
 508 |         self,
 509 |         video_expert,
 510 |         action_expert: ActionDiT,
 511 |         mot: MoT,
 512 |         vae,
 513 |         text_encoder=None,
 514 |         tokenizer=None,
 515 |         text_dim: Optional[int] = None,
 516 |         proprio_dim: Optional[int] = None,
 517 |         device: str = "cpu",
 518 |         torch_dtype: torch.dtype = torch.float32,
 519 |         video_train_shift: float = 5.0,
 520 |         video_infer_shift: float = 5.0,
 521 |         video_num_train_timesteps: int = 1000,
 522 |         action_train_shift: float = 5.0,
 523 |         action_infer_shift: float = 5.0,
 524 |         action_num_train_timesteps: int = 1000,
 525 |         loss_lambda_video: float = 1.0,
 526 |         loss_lambda_action: float = 1.0,
 527 |     ):
 528 |         super().__init__()
 529 |         self.video_expert = video_expert
 530 |         self.action_expert = action_expert
 531 |         self.mot = mot
 532 |         # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
 533 |         self.dit = self.mot
 534 |
 535 |         self.vae = vae
 536 |         self.text_encoder = text_encoder
 537 |         self.tokenizer = tokenizer
 538 |         if text_dim is None:
 539 |             if self.text_encoder is None:
 540 |                 raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
 541 |             text_dim = int(self.text_encoder.dim)
 542 |         self.text_dim = int(text_dim)
 543 |         self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
 544 |         if self.proprio_dim is not None:
 545 |             self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
 546 |         else:
 547 |             self.proprio_encoder = None
 548 |
 549 |         self.train_video_scheduler = WanContinuousFlowMatchScheduler(
 550 |             num_train_timesteps=video_num_train_timesteps,
 551 |             shift=video_train_shift,
 552 |         )
 553 |         self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
 554 |             num_train_timesteps=video_num_train_timesteps,
 555 |             shift=video_infer_shift,
 556 |         )
 557 |         self.train_action_scheduler = WanContinuousFlowMatchScheduler(
 558 |             num_train_timesteps=action_num_train_timesteps,
 559 |             shift=action_train_shift,
 560 |         )
 561 |         self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
 562 |             num_train_timesteps=action_num_train_timesteps,
 563 |             shift=action_infer_shift,
 564 |         )
 565 |         # Optional aliases for consistency with Wan22Core naming.
 566 |         self.train_scheduler = self.train_video_scheduler
 567 |         self.infer_scheduler = self.infer_video_scheduler
 568 |
 569 |         self.device = torch.device(device)
 570 |         self.torch_dtype = torch_dtype
 571 |         self.loss_lambda_video = float(loss_lambda_video)
 572 |         self.loss_lambda_action = float(loss_lambda_action)
 573 |         self._module_torch_compile_enabled = False
 574 |
 575 |         # Optional accelerated MoT body and module paths.
 576 |         self._graph_mgr: Optional[CUDAGraphManager] = None
 577 |         self._action_body_fn: Optional[Any] = None
 578 |         self._vae_encode_graph_mgr: Optional[VaeEncodeCUDAGraphManager] = None
 579 |         self._prefill_graph_mgr: Optional[PrefillCUDAGraphManager] = None
 580 |         self._module_vae_encode_fn = None
 581 |         self._module_prefill_fn = None
 582 |
 583 |         # Cache for VideoPreDiT invariant tensors across infer_action chunks.
 584 |         # Keyed by (context_id, context_mask_id, input_shape) so it auto-invalidates
 585 |         # if the caller changes any of these (shouldn't happen within a session).
 586 |         self._video_predit_cache: Optional[dict[str, Any]] = None
 587 |
 588 |         self.to(self.device)
 589 |
 590 |     @classmethod
```

### S:src/qi/models/wan22/fastwam.py（源行 1261–1302）

```python
1261 |     @torch.inference_mode()
1262 |     def _predict_joint_noise(
1263 |         self,
1264 |         latents_video: torch.Tensor,
1265 |         latents_action: torch.Tensor,
1266 |         timestep_video: torch.Tensor,
1267 |         timestep_action: torch.Tensor,
1268 |         context: torch.Tensor,
1269 |         context_mask: torch.Tensor,
1270 |         fuse_vae_embedding_in_latents: bool,
1271 |         gt_action: Optional[torch.Tensor] = None,
1272 |         expert_cache_state: Optional[FixedStepCacheState] = None,
1273 |         expert_cache_step: Optional[int] = None,
1274 |     ) -> tuple[torch.Tensor, torch.Tensor]:
1275 |         if getattr(self.mot, "nvtx_profile_label", None):
1276 |             self.mot.nvtx_profile_step = expert_cache_step
1277 |         video_pre = self.video_expert.pre_dit(
1278 |             x=latents_video,
1279 |             timestep=timestep_video,
1280 |             context=context,
1281 |             context_mask=context_mask,
1282 |             action=gt_action,
1283 |             fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
1284 |         )
1285 |         action_pre = self.action_expert.pre_dit(
1286 |             action_tokens=latents_action,
1287 |             timestep=timestep_action,
1288 |             context=context,
1289 |             context_mask=context_mask,
1290 |         )
1291 |
1292 |         attention_mask = self._build_mot_attention_mask(
1293 |             video_seq_len=video_pre["tokens"].shape[1],
1294 |             action_seq_len=action_pre["tokens"].shape[1],
1295 |             video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
1296 |             device=video_pre["tokens"].device,
1297 |         )
1298 |
1299 |         attention_observer = getattr(self.mot, "action_future_attention_observer", None)
1300 |         if attention_observer is not None:
1301 |             if expert_cache_state is not None:
1302 |                 raise ValueError("Action-future attention capture requires expert_cache=false.")
```

### S:src/qi/models/wan22/fastwam.py（源行 1348–1378）

```python
1348 |         tokens_out = self.mot(
1349 |             embeds_all={
1350 |                 "video": video_pre["tokens"],
1351 |                 "action": action_pre["tokens"],
1352 |             },
1353 |             attention_mask=attention_mask,
1354 |             freqs_all={
1355 |                 "video": video_pre["freqs"],
1356 |                 "action": action_pre["freqs"],
1357 |             },
1358 |             context_all={
1359 |                 "video": {
1360 |                     "context": video_pre["context"],
1361 |                     "mask": video_pre["context_mask"],
1362 |                 },
1363 |                 "action": {
1364 |                     "context": action_pre["context"],
1365 |                     "mask": action_pre["context_mask"],
1366 |                 },
1367 |             },
1368 |             t_mod_all={
1369 |                 "video": video_pre["t_mod"],
1370 |                 "action": action_pre["t_mod"],
1371 |             },
1372 |         )
1373 |
1374 |         pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
1375 |         pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
1376 |         if version1_controller is not None:
1377 |             pred_video = version1_controller.process_video_prediction(pred_video)
1378 |         return pred_video, pred_action
```

### S:src/qi/models/wan22/fastwam.py（源行 2185–2204）

```python
2185 |     def load_checkpoint(self, path, optimizer=None):
2186 |         payload = torch.load(path, map_location="cpu")
2187 |         if "mot" in payload:
2188 |             self.mot.load_state_dict(payload["mot"], strict=False)
2189 |         elif "dit" in payload:
2190 |             logger.warning("Loading legacy `dit` checkpoint into video expert only.")
2191 |             self.video_expert.load_state_dict(payload["dit"], strict=False)
2192 |         else:
2193 |             raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
2194 |         if self.proprio_encoder is not None:
2195 |             if "proprio_encoder" in payload:
2196 |                 self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
2197 |             else:
2198 |                 logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
2199 |         elif "proprio_encoder" in payload:
2200 |             logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")
2201 |
2202 |         if optimizer is not None and "optimizer" in payload:
2203 |             optimizer.load_state_dict(payload["optimizer"])
2204 |         return payload
```

### S:src/qi/models/wan22/schedulers/scheduler_continuous.py（源行 1–88）

```python
   1 | import torch
   2 |
   3 |
   4 | class WanContinuousFlowMatchScheduler:
   5 |     """Continuous-time Flow-Matching scheduler with shift-based sampling."""
   6 |
   7 |     def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
   8 |         if num_train_timesteps <= 0:
   9 |             raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
  10 |         if shift <= 0:
  11 |             raise ValueError(f"`shift` must be positive, got {shift}")
  12 |         self.num_train_timesteps = int(num_train_timesteps)
  13 |         self.shift = float(shift)
  14 |         self.eps = float(eps)
  15 |         self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()
  16 |
  17 |     @staticmethod
  18 |     def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
  19 |         return shift * u / (1.0 + (shift - 1.0) * u)
  20 |
  21 |     def _precompute_training_weight_stats(self) -> tuple[float, float]:
  22 |         steps = self.num_train_timesteps
  23 |         u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
  24 |         t_grid = self._phi(u_grid, self.shift) * float(steps)
  25 |         y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
  26 |         y_min = float(y_grid.min().item())
  27 |         y_shifted_grid = y_grid - y_min
  28 |         norm_const = float(y_shifted_grid.mean().item())
  29 |         return y_min, norm_const
  30 |
  31 |     def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
  32 |         if batch_size <= 0:
  33 |             raise ValueError(f"`batch_size` must be positive, got {batch_size}")
  34 |         u = torch.rand((batch_size,), device=device, dtype=torch.float32)
  35 |         sigma = self._phi(u, self.shift)
  36 |         timestep = sigma * float(self.num_train_timesteps)
  37 |         return timestep.to(dtype=dtype)
  38 |
  39 |     def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
  40 |         t = timestep.to(dtype=torch.float32)
  41 |         steps = float(self.num_train_timesteps)
  42 |         y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
  43 |         y_shifted = y - self._y_min
  44 |         weight = y_shifted / (self._weight_norm_const + self.eps)
  45 |         if weight.numel() == 1:
  46 |             return weight.reshape(())
  47 |         return weight
  48 |
  49 |     def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
  50 |         sigma = (timestep / float(self.num_train_timesteps)).to(
  51 |             original_samples.device, dtype=original_samples.dtype
  52 |         )
  53 |         if sigma.ndim == 0:
  54 |             return (1 - sigma) * original_samples + sigma * noise
  55 |         sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
  56 |         return (1 - sigma) * original_samples + sigma * noise
  57 |
  58 |     @staticmethod
  59 |     def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
  60 |         del timestep
  61 |         return noise - sample
  62 |
  63 |     def build_inference_schedule(
  64 |         self,
  65 |         num_inference_steps: int,
  66 |         device: torch.device,
  67 |         dtype: torch.dtype,
  68 |         shift_override: float | None = None,
  69 |     ) -> tuple[torch.Tensor, torch.Tensor]:
  70 |         if num_inference_steps <= 0:
  71 |             raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
  72 |         shift = self.shift if shift_override is None else float(shift_override)
  73 |         if shift <= 0:
  74 |             raise ValueError(f"`shift` must be positive, got {shift}")
  75 |
  76 |         u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
  77 |         sigma_steps = self._phi(u_steps, shift)
  78 |         timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
  79 |         deltas = sigma_steps[1:] - sigma_steps[:-1]
  80 |         return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)
  81 |
  82 |     @staticmethod
  83 |     def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
  84 |         delta = delta.to(sample.device, dtype=sample.dtype)
  85 |         if delta.ndim == 0:
  86 |             return sample + model_output * delta
  87 |         delta = delta.view(-1, *([1] * (sample.ndim - 1)))
  88 |         return sample + model_output * delta
```

### S:src/qi/datasets/lerobot/utils/normalizer.py（源行 97–153）

```python
  97 |         self.stats = stats
  98 |         self.mode = mode
  99 |
 100 |         if mode == "z-score":
 101 |             input_mean, input_std = stats["mean"], stats["std"]
 102 |             scale = 1.0 / (input_std + self.std_reg)
 103 |             offset = - input_mean / (input_std + self.std_reg)
 104 |         else:
 105 |             if mode == "min/max":
 106 |                 input_min, input_max = stats["min"], stats["max"]
 107 |             elif mode == "q01/q99":
 108 |                 input_min, input_max = stats["q01"], stats["q99"]
 109 |             else:
 110 |                 # parse const_min/const_max
 111 |                 input_min, input_max = map(float, mode.split("/"))
 112 |                 input_min = torch.full_like(stats["min"], input_min)
 113 |                 input_max = torch.full_like(stats["max"], input_max)
 114 |
 115 |             input_range = input_max - input_min
 116 |             ignore_dim = input_range < self.range_tol
 117 |             input_range[ignore_dim] = self.output_max - self.output_min
 118 |             scale = (self.output_max - self.output_min) / input_range
 119 |             offset = self.output_min - scale * input_min
 120 |             offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[ignore_dim]
 121 |
 122 |         self.scale = scale
 123 |         self.offset = offset
 124 |     def get_stats(self):
 125 |         return self.stats
 126 |
 127 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 128 |         x = x * self.scale + self.offset
 129 |         x = torch.clamp(x, -5.0, 5.0)
 130 |         return x
 131 |     def backward(self, x: torch.Tensor) -> torch.Tensor:
 132 |         x = (x - self.offset) / self.scale
 133 |         return x
 134 |
 135 | def save_dataset_stats_to_json(dataset_stats: dict, file_path: str):
 136 |
 137 |     def convert_tensor(obj):
 138 |         if isinstance(obj, torch.Tensor):
 139 |             return obj.detach().cpu().numpy().tolist()
 140 |         elif isinstance(obj, (defaultdict, dict)):
 141 |             return {k: convert_tensor(v) for k, v in dict(obj).items()}
 142 |         elif isinstance(obj, (list, tuple)):
 143 |             return [convert_tensor(item) for item in obj]
 144 |         elif isinstance(obj, (int, float, str, bool, type(None))):
 145 |             return obj
 146 |         else:
 147 |             return str(obj)
 148 |
 149 |     serializable_stats = convert_tensor(dataset_stats)
 150 |
 151 |     with open(file_path, 'w', encoding='utf-8') as f:
 152 |         json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
 153 |
```

### S:src/qi/datasets/lerobot/base_lerobot_dataset.py（源行 71–111）

```python
  71 |         delta_timestamps = {}
  72 |         for meta in self.image_meta:
  73 |             key = meta["key"]
  74 |             meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"
  75 |             delta_timestamps[meta["lerobot_key"]] = [
  76 |                 (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
  77 |             ]
  78 |
  79 |         for meta in self.state_meta:
  80 |             key = meta["key"]
  81 |             meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"
  82 |             delta_timestamps[meta["lerobot_key"]] = [
  83 |                 (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
  84 |             ]
  85 |
  86 |         for meta in self.action_meta:
  87 |             key = meta["key"]
  88 |             meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"
  89 |             delta_timestamps[meta["lerobot_key"]] = [(t * global_sample_stride) / fps for t in range(-past_action_size, -past_action_size + action_size)]
  90 |
  91 |         episodes = {}
  92 |         if val_set_proportion < 1e-6:
  93 |             for meta in metas:
  94 |                 episodes.update({meta.repo_id: list(range(meta.total_episodes))})
  95 |         else:
  96 |             for meta in metas:
  97 |                 split_idx = int(meta.total_episodes * (1 - val_set_proportion))
  98 |                 # random shuffle episode indices before splitting
  99 |                 episode_indices = list(range(meta.total_episodes))
 100 |                 rng = np.random.default_rng(seed)
 101 |                 rng.shuffle(episode_indices)
 102 |                 if self.is_training_set:
 103 |                     episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx)]})
 104 |                 else:
 105 |                     episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx, meta.total_episodes)]})
 106 |
 107 |         self.multi_dataset = MultiLeRobotDataset(
 108 |             dataset_dirs=self.dataset_dirs,
 109 |             episodes=episodes,
 110 |             delta_timestamps=delta_timestamps,
 111 |         )
```

### S:src/qi/datasets/lerobot/base_lerobot_dataset.py（源行 356–367）

```python
 356 |
 357 |         # assume that each minibatch has equal number of samples
 358 |         def get_mean_std(means, vars):
 359 |             means = torch.stack(means)
 360 |             vars = torch.stack(vars)
 361 |             stepwise_mean = means.mean(0)
 362 |             stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
 363 |             global_mean = means.mean((0, 1))
 364 |             global_std = (vars + (means - global_mean) ** 2).mean((0, 1)).sqrt()
 365 |             return stepwise_mean, stepwise_std, global_mean, global_std
 366 |
 367 |         stats = {"state": DefaultDict(dict), "action": DefaultDict(dict), "num_episodes": episodes_num, "num_transition": self.multi_dataset.num_frames}
```

### S:src/qi/deployment/rtc_buffer.py（源行 69–101）

```python
  69 |     def get_action(self, current_time: int) -> np.ndarray | None:
  70 |         """Fuse all cached action predictions that cover the current control step."""
  71 |         current_time = int(current_time)
  72 |         with self.lock:
  73 |             relevant = {}
  74 |             expired = []
  75 |             before_keys = sorted(self.chunks.keys())
  76 |
  77 |             for cursor, chunk in self.chunks.items():
  78 |                 end = cursor + self.chunk_size
  79 |                 if cursor <= current_time < end:
  80 |                     relevant[cursor] = chunk[current_time - cursor]
  81 |                 elif end <= current_time:
  82 |                     expired.append(cursor)
  83 |
  84 |             for cursor in expired:
  85 |                 del self.chunks[cursor]
  86 |
  87 |             if self.debug:
  88 |                 print(
  89 |                     f"[action_chunks] t={current_time} before={before_keys} "
  90 |                     f"delete={sorted(expired)} after={sorted(self.chunks.keys())}"
  91 |                 )
  92 |
  93 |             if not relevant:
  94 |                 return None
  95 |
  96 |             sorted_items = sorted(relevant.items(), key=lambda item: item[0])
  97 |             candidate_actions = np.asarray([action for _, action in sorted_items], dtype=np.float32)
  98 |
  99 |         exp_weights = np.exp(self.exp_weight_factor * np.arange(len(candidate_actions), dtype=np.float32))
 100 |         exp_weights = (exp_weights / exp_weights.sum())[:, None]
 101 |         return (candidate_actions * exp_weights).sum(axis=0).astype(np.float32)
```

### Q:src/fastwam_steerquant/adapters/base.py（源行 1–90）

```python
   1 | from __future__ import annotations
   2 |
   3 | import hashlib
   4 | import importlib
   5 | import inspect
   6 | import json
   7 | from dataclasses import dataclass
   8 | from pathlib import Path
   9 | from typing import Callable
  10 |
  11 | import torch
  12 |
  13 | from ..streams import FastWAMStreamConfig
  14 | from ..recovery import file_identity
  15 |
  16 |
  17 | def read_config(path) -> dict:
  18 |     """JSON paths in assets are relative to the config file, not the cwd."""
  19 |     path = Path(path).expanduser().resolve()
  20 |     config = json.loads(path.read_text())
  21 |     for key in ("adapter", "num_calls", "stream_config", "expected_sites"):
  22 |         if key not in config:
  23 |             raise ValueError(f"Missing configuration field: {key}")
  24 |     if config["num_calls"] <= 0 or config["expected_sites"] <= 0:
  25 |         raise ValueError("num_calls and expected_sites must be positive.")
  26 |     FastWAMStreamConfig(**config["stream_config"]).validate()
  27 |     for key, value in config.get("assets", {}).items():
  28 |         p = Path(value).expanduser()
  29 |         config["assets"][key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
  30 |     return config
  31 |
  32 |
  33 | def _factory(config):
  34 |     module_name, separator, symbol = config["adapter"].partition(":")
  35 |     if not separator:
  36 |         raise ValueError("adapter must be a module:factory import path.")
  37 |     module = importlib.import_module(module_name)
  38 |     return getattr(module, symbol)
  39 |
  40 |
  41 | def config_identity(config) -> dict:
  42 |     """Guard resume against changed settings/assets/adapter implementation.
  43 |
  44 |     Large asset files use path/size/mtime identity; archive content hashes in
  45 |     config['provenance'] for stronger cross-server provenance when available.
  46 |     List all external config/statistics/weights files under assets.
  47 |     """
  48 |     identity = {"config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
  49 |                 "assets": {k: file_identity(v) for k, v in config.get("assets", {}).items()}}
  50 |     source = inspect.getsourcefile(_factory(config))
  51 |     if source:
  52 |         identity["adapter_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
  53 |     return identity
  54 |
  55 |
  56 | def load_adapter(config):
  57 |     config = read_config(config) if not isinstance(config, dict) else config
  58 |     adapter = _factory(config)(config)
  59 |     for name in ("load_model", "build_model", "prepare_inputs", "infer_kwargs", "action_scale"):
  60 |         if not callable(getattr(adapter, name, None)):
  61 |             raise TypeError(f"Adapter must implement {name}.")
  62 |     return adapter
  63 |
  64 |
  65 | @dataclass
  66 | class CallbackAdapter:
  67 |     """Connect existing robot code without changing the quantization algorithm.
  68 |
  69 |     load_model(device=...) -> ORIGINAL trained model, eval mode
  70 |     build_model(model_config, device=...) -> structure only, no checkpoint load
  71 |     prepare_inputs(model, record, seed=...) -> five differentiable-denoise tensors
  72 |     infer_kwargs(model, record, seed=...) -> kwargs for production infer_action
  73 |     action_scale() -> positive finite std in MODEL-NORMALIZED action coordinates
  74 |     finalize_model(model, device=...) -> rebuild device-dependent non-state assets
  75 |     dataset_records() -> iterable of standard records (optional raw-data export)
  76 |     """
  77 |     load_model: Callable
  78 |     build_model: Callable
  79 |     prepare_inputs: Callable
  80 |     infer_kwargs: Callable
  81 |     action_scale: Callable
  82 |     finalize_model: Callable | None = None
  83 |     dataset_records: Callable | None = None
  84 |
  85 |
  86 | def validate_action_scale(value, *, device):
  87 |     value = torch.as_tensor(value, dtype=torch.float32, device=device)
  88 |     if value.ndim != 1 or value.numel() == 0 or not torch.isfinite(value).all() or (value <= 0).any():
  89 |         raise ValueError("action_scale must be a positive finite vector in normalized action coordinates.")
  90 |     return value
```

### Q:src/fastwam_steerquant/adapters/fastwam.py（源行 1–39）

```python
   1 | """Reusable input preparation for adapters using the existing FastWAM API."""
   2 | import torch
   3 |
   4 |
   5 | @torch.no_grad()
   6 | def prepare_from_infer_kwargs(model, kwargs, *, seed):
   7 |     """kwargs must match production infer_action's ALREADY normalized inputs.
   8 |
   9 |     input_image: batched CHW float in [-1,1], including correct camera packing.
  10 |     proprio: batched, normalized exactly as at training/deployment.
  11 |     This function does not guess camera order, resize or normalize raw robot data.
  12 |     """
  13 |     image = kwargs["input_image"].to(device=model.device, dtype=model.torch_dtype)
  14 |     proprio = kwargs["proprio"].to(device=model.device, dtype=model.torch_dtype)
  15 |     if image.ndim != 4 or image.shape[0] != 1 or proprio.ndim != 2 or proprio.shape[0] != 1:
  16 |         raise ValueError("Calibration currently expects batch size one, BCHW images and batched proprio.")
  17 |     if not image.is_floating_point() or not torch.isfinite(image).all() or image.abs().max() > 1.001:
  18 |         raise ValueError("Provide production-normalized image values in [-1,1].")
  19 |     if float(kwargs.get("text_cfg_scale", 1.0)) != 1:
  20 |         raise ValueError("This calibration preparation supports text_cfg_scale=1.")
  21 |     if kwargs.get("rand_device", "cpu") != "cpu" or kwargs.get("tiled", False):
  22 |         raise ValueError("This helper expects CPU noise generation and tiled=False.")
  23 |     first = model._encode_input_image_latents_tensor(input_image=image, tiled=False)
  24 |     context, mask = model.encode_prompt(kwargs["prompt"])
  25 |     context, mask = model._append_proprio_to_context(context=context, context_mask=mask, proprio=proprio)
  26 |     latent_t = (int(kwargs["num_video_frames"]) - 1) // int(model.vae.temporal_downsample_factor) + 1
  27 |     h, w = image.shape[-2:]
  28 |     factor = int(model.vae.upsampling_factor)
  29 |     # Production infer_action initializes two generators with the SAME seed.
  30 |     # Sharing one advancing generator would change the action-noise sample.
  31 |     generator = torch.Generator(device="cpu").manual_seed(seed)
  32 |     video = torch.randn((1, model.vae.model.z_dim, latent_t, h // factor, w // factor),
  33 |                         generator=generator).to(device=image.device, dtype=model.torch_dtype)
  34 |     action = torch.randn((1, int(kwargs["action_horizon"]), model.action_expert.action_dim),
  35 |                          generator=torch.Generator(device="cpu").manual_seed(seed)).to(
  36 |                              device=image.device, dtype=model.torch_dtype)
  37 |     video[:, :, :1] = first
  38 |     return dict(latents_video=video, latents_action=action, first_frame_latents=first,
  39 |                 context=context, context_mask=mask)
```

### Q:src/fastwam_steerquant/state.py（源行 77–138）

```python
  77 |     latents_video: torch.Tensor,
  78 |     latents_action: torch.Tensor,
  79 |     first_frame_latents: torch.Tensor,
  80 |     context: torch.Tensor,
  81 |     context_mask: torch.Tensor,
  82 |     num_inference_steps: int,
  83 |     state: DenoiseCallState,
  84 |     sigma_shift: float | None = None,
  85 | ) -> torch.Tensor:
  86 |     """Differentiable counterpart of FastWAMJoint's joint denoising loop.
  87 |
  88 |     VAE/text preparation stays outside this function. The returned action is
  89 |     not detached, so final-action VJPs can reach any selected Linear output.
  90 |     """
  91 |
  92 |     if state.num_calls != int(num_inference_steps):
  93 |         raise ValueError("state.num_calls must equal num_inference_steps.")
  94 |     video_steps, video_deltas = model.infer_video_scheduler.build_inference_schedule(
  95 |         num_inference_steps=num_inference_steps,
  96 |         device=latents_video.device,
  97 |         dtype=latents_video.dtype,
  98 |         shift_override=sigma_shift,
  99 |     )
 100 |     action_steps, action_deltas = model.infer_action_scheduler.build_inference_schedule(
 101 |         num_inference_steps=num_inference_steps,
 102 |         device=latents_action.device,
 103 |         dtype=latents_action.dtype,
 104 |         shift_override=sigma_shift,
 105 |     )
 106 |     fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
 107 |     for call, (tv, dv, ta, da) in enumerate(zip(video_steps, video_deltas, action_steps, action_deltas)):
 108 |         state.set(call)
 109 |         timestep_video = tv.unsqueeze(0).to(device=latents_video.device, dtype=latents_video.dtype)
 110 |         timestep_action = ta.unsqueeze(0).to(device=latents_action.device, dtype=latents_action.dtype)
 111 |         latent_t, latent_h, latent_w = latents_video.shape[-3:]
 112 |         patch_t, patch_h, patch_w = (int(value) for value in model.video_expert.patch_size)
 113 |         tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
 114 |         mask = model._build_mot_attention_mask(
 115 |             video_seq_len=(latent_t // patch_t) * tokens_per_frame,
 116 |             action_seq_len=latents_action.shape[1],
 117 |             video_tokens_per_frame=tokens_per_frame,
 118 |             device=latents_video.device,
 119 |         )
 120 |         pred_video, pred_action = model._joint_denoise_core(
 121 |             latents_video=latents_video,
 122 |             latents_action=latents_action,
 123 |             timestep_video=timestep_video,
 124 |             timestep_action=timestep_action,
 125 |             context=context,
 126 |             context_mask=context_mask,
 127 |             attention_mask=mask,
 128 |             fuse_vae_embedding_in_latents=fuse,
 129 |             action_condition=None,
 130 |         )
 131 |         latents_video = model.infer_video_scheduler.step(pred_video, dv, latents_video)
 132 |         latents_action = model.infer_action_scheduler.step(pred_action, da, latents_action)
 133 |         latents_video = torch.cat(
 134 |             [first_frame_latents, latents_video[:, :, first_frame_latents.shape[2] :]],
 135 |             dim=2,
 136 |         )
 137 |     state.clear()
 138 |     return latents_action
```

### Q:src/fastwam_steerquant/policy.py（源行 1–46）

```python
   1 | """Integration into an existing real-robot loop; no hardware actions here."""
   2 | import threading
   3 | import torch
   4 | from .deployment import load_deployment
   5 | from .state import FastWAMCallTracker
   6 | from .rollout_graph import LiveJointDiTGraph
   7 |
   8 |
   9 | class QuantizedPolicy:
  10 |     def __init__(self, deployment, adapter, *, device="cuda", construct_device="meta", cuda_graph=False):
  11 |         self.model, self.state, self.metadata = load_deployment(
  12 |             deployment, adapter.build_model, device=device, construct_device=construct_device)
  13 |         if callable(getattr(adapter, "finalize_model", None)):
  14 |             adapter.finalize_model(self.model, device=device)
  15 |         self.adapter = adapter
  16 |         self.tracker = FastWAMCallTracker(self.model, self.state)
  17 |         self.tracker.install()
  18 |         self.model._wam_call_tracker = self.tracker
  19 |         if cuda_graph:
  20 |             LiveJointDiTGraph(self.model, num_calls=self.state.num_calls).install()
  21 |         self._lock = threading.Lock()
  22 |
  23 |     @torch.inference_mode()
  24 |     def infer(self, record, *, seed=42):
  25 |         if not self._lock.acquire(blocking=False):
  26 |             raise RuntimeError("Use one policy instance per serialized inference loop.")
  27 |         try:
  28 |             self.tracker.reset()
  29 |             kwargs = self.adapter.infer_kwargs(self.model, record, seed=seed)
  30 |             if kwargs.get("num_inference_steps") != self.state.num_calls:
  31 |                 raise ValueError("infer_kwargs must explicitly match calibrated num_inference_steps.")
  32 |             if float(kwargs.get("text_cfg_scale", 1)) != 1:
  33 |                 raise ValueError("Current calibration/deployment protocol requires text_cfg_scale=1.")
  34 |             if kwargs.get("sigma_shift") != self.metadata["model_config"].get("sigma_shift"):
  35 |                 raise ValueError("Inference sigma schedule differs from calibration configuration.")
  36 |             for key in ("action_horizon", "num_video_frames"):
  37 |                 configured = self.metadata["model_config"].get("model", {}).get(key)
  38 |                 if configured is not None and kwargs.get(key) != configured:
  39 |                     raise ValueError(f"Inference {key} differs from calibration configuration.")
  40 |             result = self.model.infer_action(**kwargs)
  41 |             if self.tracker.total_calls != self.state.num_calls:
  42 |                 raise RuntimeError("Actual denoising calls differ from calibration.")
  43 |             return result
  44 |         finally:
  45 |             self.state.clear()
  46 |             self._lock.release()
```

### Q:src/fastwam_steerquant/deployment.py（源行 30–85）

```python
  30 | def export_deployment(model: nn.Module, checkpoint: QuantizationCheckpoint, path: str | Path,
  31 |                       *, model_config: dict, provenance: dict | None = None) -> Path:
  32 |     """Export from the ORIGINAL model, not a rotated or already replaced model.
  33 |
  34 |     Runs on CPU when the caller loads the original model on CPU. Original FP
  35 |     weights are not mutated; target weights are omitted from the output.
  36 |     """
  37 |     checkpoint.validate()
  38 |     if checkpoint.weight_bits != 4 or checkpoint.activation_bits not in (4, 8):
  39 |         raise ValueError("Deployment requires W4A8 or W4A4 calibration.")
  40 |     original_sites = enumerate_fastwamjoint_linears(model)
  41 |     if {s.module_name for s in original_sites} != {s.module_name for s in checkpoint.sites}:
  42 |         raise ValueError("Export requires calibration of every target Linear, not a partial shard.")
  43 |     if any(hasattr(s.module, "_wam_rotation") for s in original_sites):
  44 |         raise ValueError("Export requires the original, unrotated model.")
  45 |     # JSON metadata cannot serialize arbitrary Python objects or execute code on load.
  46 |     metadata = json.loads(json.dumps({"model_config": model_config, "provenance": provenance or {}}))
  47 |     excluded = {s.module_name + "." + name for s in checkpoint.sites for name in ("weight", "bias")}
  48 |     state = {name: _cpu(value) for name, value in model.state_dict().items() if name not in excluded}
  49 |     specs = []
  50 |     for entry in checkpoint.sites:
  51 |         source = resolve_site_module(model, entry.module_name)
  52 |         if tuple(source.weight.shape) != tuple(entry.qweight.shape):
  53 |             raise ValueError(f"Shape mismatch: {entry.module_name}")
  54 |         n, k = entry.qweight.shape
  55 |         alignment = 256 if checkpoint.activation_bits == 4 else 32
  56 |         if k % alignment or n % 8:
  57 |             raise ValueError(f"Unsupported native shape {entry.module_name}: N={n}, K={k}")
  58 |         if entry.rotation != "none" and checkpoint.activation_bits != 4:
  59 |             raise ValueError("RHT requires W4A4.")
  60 |         spec = {"module_name": entry.module_name, "site_index": entry.site_index,
  61 |                 "expert": entry.expert, "operation": entry.operation,
  62 |                 "stream_names": list(entry.stream_names), "rotation": entry.rotation,
  63 |                 "in_features": k, "out_features": n, "has_bias": source.bias is not None}
  64 |         values = {"weight": pack_signed_int4(entry.qweight.cpu()),
  65 |                   "weight_scales": entry.weight_scales.float().reshape(-1),
  66 |                   "input_scale": entry.input_scale.float().reciprocal(),
  67 |                   "gamma_gains": entry.gamma_gains.float(),
  68 |                   "clipping_ranges": entry.clipping_ranges.float() / (2 ** (checkpoint.activation_bits - 1) - 1),
  69 |                   "bias": source.bias, "input_rotation_signs": entry.input_rotation_signs}
  70 |         for name, value in values.items():
  71 |             if value is not None:
  72 |                 state[entry.module_name + "." + name] = _cpu(value)
  73 |         specs.append(spec)
  74 |     # Include registered nonpersistent buffers (e.g. positional caches) so meta
  75 |     # construction does not leave an uninitialized buffer at model.to(device).
  76 |     nonpersistent = {}
  77 |     for prefix, module in model.named_modules():
  78 |         for name in module._non_persistent_buffers_set:
  79 |             value = module._buffers[name]
  80 |             if value is not None:
  81 |                 nonpersistent[f"{prefix}.{name}" if prefix else name] = _cpu(value)
  82 |     payload = {"format": FORMAT, "weight_bits": 4, "activation_bits": checkpoint.activation_bits,
  83 |                "num_calls": checkpoint.num_calls, "stream_config": checkpoint.stream_config,
  84 |                "sites": specs, "state_dict": state, "nonpersistent_buffers": nonpersistent,
  85 |                "d_config": checkpoint.d_config, "gamma_config": checkpoint.gamma_config, **metadata}
```

### Q:src/fastwam_steerquant/deployment.py（源行 135–175）

```python
 135 | def load_deployment(path: str | Path, build_model: Callable, *, device="cuda",
 136 |                     construct_device="meta") -> tuple[nn.Module, DenoiseCallState, dict]:
 137 |     """build_model(model_config, device=...) must construct WITHOUT loading a ckpt.
 138 |
 139 |     No original FP target weight is ever moved to GPU. CPU construction is an
 140 |     explicit alternative for models whose constructors do not support meta.
 141 |     Unregistered tensors/external assets are the adapter's responsibility.
 142 |     """
 143 |     if construct_device not in ("meta", "cpu"):
 144 |         raise ValueError("Construct on meta or CPU to avoid FP GPU allocation.")
 145 |     payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True, mmap=True)
 146 |     validate_payload(payload)
 147 |     model = build_model(payload["model_config"], device=construct_device)
 148 |     if any(t.device.type not in ("meta", "cpu") for t in (*model.parameters(), *model.buffers())):
 149 |         raise ValueError("Builder placed weights on GPU before packed replacement.")
 150 |     existing = enumerate_fastwamjoint_linears(model)
 151 |     if {s.module_name for s in existing} != {s["module_name"] for s in payload["sites"]}:
 152 |         raise ValueError("Deployment/model topology mismatch.")
 153 |     state = DenoiseCallState(payload["num_calls"])
 154 |     streams = FastWAMStreamConfig(**payload["stream_config"])
 155 |     for spec in payload["sites"]:
 156 |         source = resolve_site_module(model, spec["module_name"])
 157 |         if (source.in_features, source.out_features, source.bias is not None) != (
 158 |                 spec["in_features"], spec["out_features"], spec["has_bias"]):
 159 |             raise ValueError(f"Builder shape/bias differs at {spec['module_name']}")
 160 |         values = {name: payload["state_dict"].get(spec["module_name"] + "." + name) for name in BUFFERS}
 161 |         parent, child = resolve_site_parent(model, spec["module_name"])
 162 |         setattr(parent, child, WAMQuantLinear.from_packed(spec, values, state=state,
 163 |                 stream_config=streams, activation_bits=payload["activation_bits"]))
 164 |     model.load_state_dict(payload["state_dict"], strict=True, assign=True)
 165 |     for name, value in payload["nonpersistent_buffers"].items():
 166 |         prefix, _, child = name.rpartition(".")
 167 |         parent = resolve_site_module(model, prefix) if prefix else model
 168 |         if child not in parent._non_persistent_buffers_set:
 169 |             raise ValueError(f"Nonpersistent buffer contract changed: {name}")
 170 |         parent._buffers[child] = value
 171 |     if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
 172 |         raise ValueError("Builder left unmaterialized buffers; export or rebuild them explicitly.")
 173 |     model.to(device=device).eval()
 174 |     metadata = {k: v for k, v in payload.items() if k not in ("state_dict", "nonpersistent_buffers")}
 175 |     return model, state, metadata
```

## 附录D：仅作来源参考的训练配置

**这些文件不等于两份15000步Joint权重的已证实训练配置。** 保留它们是为了说明现用推理YAML和stats的参考来源，以及为何仍需真实训练manifest。10step配置含spectral训练、max_steps10；旧stack是普通FastWAM。以下完整内容没有执行。

### P:checkpoints/10step_joint/config.yaml（源行 1–274）

```yaml
   1 | output_dir: /project/peilab/wam/GR-LAM/artifacts/piper_grlam/runs_fixed_full_k4_aa97bba/piper14_grlam_spectral_fixed_k4/pack3_joint_fixed_full_k4_smoke_544527
   2 | batch_size: 8
   3 | num_workers: 2
   4 | lr_scheduler_type: cosine
   5 | learning_rate: 1.0e-05
   6 | backbone_learning_rate: 1.0e-05
   7 | new_module_learning_rate: 1.0e-05
   8 | num_epochs: 100
   9 | max_steps: 10
  10 | log_every: 10
  11 | save_every: 0
  12 | checkpoint_keep_last: 1
  13 | checkpoint_state_keep_last: null
  14 | eval_every: 0
  15 | eval_num_inference_steps: 10
  16 | gradient_accumulation_steps: 1
  17 | mixed_precision: bf16
  18 | seed: 42
  19 | max_grad_norm: 1.0
  20 | weight_decay: 0.01
  21 | optimizer_type: torch_adamw
  22 | training_phase: null
  23 | resume: null
  24 | resume_allow_incomplete_state: false
  25 | partial_resume: null
  26 | phase_init: null
  27 | wandb:
  28 |   enabled: false
  29 |   workspace: null
  30 |   project: cy_grlam
  31 |   name: piper14_grlam_spectral_fixed_k4
  32 |   group: pack_3_objects_plus
  33 |   mode: online
  34 | tensorboard:
  35 |   enabled: false
  36 |   log_dir: /project/peilab/wam/GR-LAM/artifacts/piper_grlam/runs_fixed_full_k4_aa97bba/piper14_grlam_spectral_fixed_k4/pack3_joint_fixed_full_k4_smoke_544527/tensorboard
  37 |   flush_secs: 30
  38 | data:
  39 |   train:
  40 |     _target_: fastwam.datasets.gr_spectral.GRSpectralTargetDataset
  41 |     cache_root: /project/peilab/wam/GR-LAM/artifacts/piper_grlam/pack_3_objects_plus/gr_cache
  42 |     horizon: 8
  43 |     target_dim: 32
  44 |     max_cached_episodes: 16
  45 |     require_complete_cache: true
  46 |     verify_manifest_identity: false
  47 |     selected_dataset_roots:
  48 |     - /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect_lerobot
  49 |     episode_split: all
  50 |     episode_holdout_mod: 10
  51 |     episode_holdout_remainder: 0
  52 |     base_dataset:
  53 |       _target_: fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
  54 |       dataset_dirs:
  55 |       - /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect_lerobot
  56 |       pretrained_norm_stats: /project/peilab/wam/GR-LAM/artifacts/piper_grlam/pack_3_objects_plus/normalization/dataset_stats.json
  57 |       shape_meta:
  58 |         images:
  59 |         - key: cam_high
  60 |           raw_shape:
  61 |           - 3
  62 |           - 480
  63 |           - 640
  64 |           shape:
  65 |           - 3
  66 |           - 240
  67 |           - 320
  68 |         - key: cam_left_wrist
  69 |           raw_shape:
  70 |           - 3
  71 |           - 480
  72 |           - 640
  73 |           shape:
  74 |           - 3
  75 |           - 240
  76 |           - 320
  77 |         - key: cam_right_wrist
  78 |           raw_shape:
  79 |           - 3
  80 |           - 480
  81 |           - 640
  82 |           shape:
  83 |           - 3
  84 |           - 240
  85 |           - 320
  86 |         action:
  87 |         - key: default
  88 |           raw_shape: 14
  89 |           shape: 14
  90 |         state:
  91 |         - key: default
  92 |           raw_shape: 14
  93 |           shape: 14
  94 |       num_frames: 33
  95 |       global_sample_stride: 1
  96 |       action_video_freq_ratio: 4
  97 |       video_size:
  98 |       - 384
  99 |       - 320
 100 |       camera_key: null
 101 |       val_set_proportion: 0.0
 102 |       is_training_set: true
 103 |       episode_indices: null
 104 |       skip_padding_as_possible: false
 105 |       concat_multi_camera: robotwin
 106 |       override_instruction: Put the three objects on the table into the container.
 107 |       processor:
 108 |         _target_: fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
 109 |         shape_meta:
 110 |           images:
 111 |           - key: cam_high
 112 |             raw_shape:
 113 |             - 3
 114 |             - 480
 115 |             - 640
 116 |             shape:
 117 |             - 3
 118 |             - 240
 119 |             - 320
 120 |           - key: cam_left_wrist
 121 |             raw_shape:
 122 |             - 3
 123 |             - 480
 124 |             - 640
 125 |             shape:
 126 |             - 3
 127 |             - 240
 128 |             - 320
 129 |           - key: cam_right_wrist
 130 |             raw_shape:
 131 |             - 3
 132 |             - 480
 133 |             - 640
 134 |             shape:
 135 |             - 3
 136 |             - 240
 137 |             - 320
 138 |           action:
 139 |           - key: default
 140 |             raw_shape: 14
 141 |             shape: 14
 142 |           state:
 143 |           - key: default
 144 |             raw_shape: 14
 145 |             shape: 14
 146 |         num_obs_steps: 33
 147 |         num_output_cameras: 3
 148 |         action_output_dim: 14
 149 |         proprio_output_dim: 14
 150 |         action_state_transforms: null
 151 |         use_stepwise_action_norm: false
 152 |         norm_default_mode: z-score
 153 |         norm_exception_mode: null
 154 |         action_state_merger:
 155 |           _target_: fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
 156 |         train_transforms:
 157 |         - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
 158 |         - _target_: torchvision.transforms.Resize
 159 |           size:
 160 |           - 240
 161 |           - 320
 162 |         val_transforms:
 163 |         - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
 164 |         - _target_: torchvision.transforms.Resize
 165 |           size:
 166 |           - 240
 167 |           - 320
 168 |         image_augmentation: true
 169 |         train_aug_transforms:
 170 |           cam_high:
 171 |           - _target_: torchvision.transforms.RandomCrop
 172 |             size:
 173 |             - 228
 174 |             - 304
 175 |           - _target_: torchvision.transforms.Resize
 176 |             size:
 177 |             - 240
 178 |             - 320
 179 |           - _target_: torchvision.transforms.RandomRotation
 180 |             degrees: 5
 181 |           - _target_: torchvision.transforms.ColorJitter
 182 |             brightness: 0.3
 183 |             contrast: 0.4
 184 |             saturation: 0.5
 185 |           cam_left_wrist:
 186 |           - _target_: torchvision.transforms.ColorJitter
 187 |             brightness: 0.3
 188 |             contrast: 0.4
 189 |             saturation: 0.5
 190 |           cam_right_wrist:
 191 |           - _target_: torchvision.transforms.ColorJitter
 192 |             brightness: 0.3
 193 |             contrast: 0.4
 194 |             saturation: 0.5
 195 |       text_embedding_cache_dir: /project/peilab/wam/physical_WM/data/text_embeds_cache/pack_3_objects_plus
 196 |       context_len: 128
 197 |   val: null
 198 | model:
 199 |   _target_: fastwam.runtime.create_fastwam_joint
 200 |   model_id: /project/peilab/wam/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B
 201 |   tokenizer_model_id: /project/peilab/wam/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B
 202 |   dit_checkpoint_path: null
 203 |   tokenizer_max_len: 128
 204 |   load_text_encoder: false
 205 |   proprio_dim: 14
 206 |   redirect_common_files: false
 207 |   mot_checkpoint_mixed_attn: true
 208 |   action_dit_pretrained_path: /project/peilab/wam/physical_WM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
 209 |   skip_dit_load_from_pretrain: false
 210 |   video_dit_config:
 211 |     has_image_input: false
 212 |     patch_size:
 213 |     - 1
 214 |     - 2
 215 |     - 2
 216 |     in_dim: 48
 217 |     hidden_dim: 3072
 218 |     ffn_dim: 14336
 219 |     freq_dim: 256
 220 |     text_dim: 4096
 221 |     out_dim: 48
 222 |     num_heads: 24
 223 |     attn_head_dim: 128
 224 |     num_layers: 30
 225 |     eps: 1.0e-06
 226 |     seperated_timestep: true
 227 |     require_clip_embedding: false
 228 |     require_vae_embedding: false
 229 |     fuse_vae_embedding_in_latents: true
 230 |     use_gradient_checkpointing: true
 231 |     video_attention_mask_mode: first_frame_causal
 232 |     action_conditioned: false
 233 |     action_dim: 14
 234 |     action_group_causal_mask_mode: group_diagonal
 235 |   action_dit_config:
 236 |     action_dim: 14
 237 |     hidden_dim: 1024
 238 |     ffn_dim: 4096
 239 |     num_heads: 24
 240 |     attn_head_dim: 128
 241 |     num_layers: 30
 242 |     text_dim: 4096
 243 |     freq_dim: 256
 244 |     eps: 1.0e-06
 245 |     use_gradient_checkpointing: true
 246 |   video_scheduler:
 247 |     train_shift: 5.0
 248 |     infer_shift: 5.0
 249 |     num_train_timesteps: 1000
 250 |   action_scheduler:
 251 |     train_shift: 5.0
 252 |     infer_shift: 5.0
 253 |     num_train_timesteps: 1000
 254 |   loss:
 255 |     lambda_action: 1.0
 256 |     lambda_video: 1.0
 257 |   spectral_plan:
 258 |     enabled: true
 259 |     training_mode: joint
 260 |     detach_backbone: true
 261 |     routing_mode: fixed
 262 |     horizon: 8
 263 |     latent_dim: 32
 264 |     low_frequencies: 4
 265 |     normalization_stats_path: /project/peilab/wam/GR-LAM/artifacts/piper_grlam/pack_3_objects_plus/normalization/gr_dct_k4_train.pt
 266 |     plan_dim: 1024
 267 |     plan_heads: 8
 268 |     plan_depth: 1
 269 |     adapter_layers:
 270 |     - 9
 271 |     - 19
 272 |     init_gate: 0.05
 273 |     lambda_plan: 0.1
 274 | freeze_video_expert: true
```

### P:checkpoints/k4_disjoint_pack_3_objecst/config.yaml（源行 1–287）

```yaml
   1 | output_dir: /project/peilab/wam/sepcwam-disjoint/runs/pack_3_objects_full_k4_disjoint/pack_3_objects_full_k4_disjoint_60k
   2 | batch_size: 8
   3 | num_workers: 2
   4 | lr_scheduler_type: cosine
   5 | learning_rate: 1.0e-05
   6 | backbone_learning_rate: 1.0e-05
   7 | new_module_learning_rate: 1.0e-05
   8 | num_epochs: 1000
   9 | max_steps: 60000
  10 | log_every: 10
  11 | save_every: 2000
  12 | checkpoint_keep_last: 2
  13 | checkpoint_state_keep_last: 2
  14 | eval_every: 0
  15 | eval_num_inference_steps: 10
  16 | gradient_accumulation_steps: 1
  17 | mixed_precision: bf16
  18 | seed: 42
  19 | max_grad_norm: 1.0
  20 | weight_decay: 0.01
  21 | optimizer_type: torch_adamw
  22 | training_phase: null
  23 | resume: /project/peilab/wam/sepcwam-disjoint/runs/pack_3_objects_full_k4_disjoint/pack_3_objects_full_k4_disjoint_60k/checkpoints/state/step_046000
  24 | resume_allow_incomplete_state: false
  25 | partial_resume: null
  26 | phase_init: null
  27 | wandb:
  28 |   enabled: true
  29 |   workspace: null
  30 |   project: ours
  31 |   name: pack_3_objects_full_k4_disjoint
  32 |   group: pack_3_objects
  33 |   mode: online
  34 | tensorboard:
  35 |   enabled: false
  36 |   log_dir: /project/peilab/wam/sepcwam-disjoint/runs/pack_3_objects_full_k4_disjoint/pack_3_objects_full_k4_disjoint_60k/tensorboard
  37 |   flush_secs: 30
  38 | fixed_validation:
  39 |   enabled: false
  40 |   num_samples: 64
  41 |   seed: 42
  42 |   eval_at_start: true
  43 |   action_only: true
  44 |   eval_every: 1000
  45 |   jsonl_path: /project/peilab/wam/sepcwam-disjoint/runs/pack_3_objects_full_k4_disjoint/pack_3_objects_full_k4_disjoint_60k/fixed_validation.jsonl
  46 |   arm: null
  47 |   pretrain_step: null
  48 | data:
  49 |   train:
  50 |     _target_: fastwam.datasets.gr_spectral.GRSpectralTargetDataset
  51 |     cache_root: /project/peilab/wam/GR-LAM/artifacts/piper_grlam/pack_3_objects_plus/gr_cache
  52 |     horizon: 8
  53 |     target_dim: 32
  54 |     max_cached_episodes: 16
  55 |     require_complete_cache: true
  56 |     verify_manifest_identity: false
  57 |     selected_dataset_roots:
  58 |     - /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect_lerobot
  59 |     episode_split: all
  60 |     episode_holdout_mod: 10
  61 |     episode_holdout_remainder: 0
  62 |     base_dataset:
  63 |       _target_: fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
  64 |       dataset_dirs:
  65 |       - /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect_lerobot
  66 |       pretrained_norm_stats: /project/peilab/wam/GR-LAM/artifacts/piper_grlam/pack_3_objects_plus/normalization/dataset_stats.json
  67 |       shape_meta:
  68 |         images:
  69 |         - key: cam_high
  70 |           raw_shape:
  71 |           - 3
  72 |           - 480
  73 |           - 640
  74 |           shape:
  75 |           - 3
  76 |           - 240
  77 |           - 320
  78 |         - key: cam_left_wrist
  79 |           raw_shape:
  80 |           - 3
  81 |           - 480
  82 |           - 640
  83 |           shape:
  84 |           - 3
  85 |           - 240
  86 |           - 320
  87 |         - key: cam_right_wrist
  88 |           raw_shape:
  89 |           - 3
  90 |           - 480
  91 |           - 640
  92 |           shape:
  93 |           - 3
  94 |           - 240
  95 |           - 320
  96 |         action:
  97 |         - key: default
  98 |           raw_shape: 14
  99 |           shape: 14
 100 |         state:
 101 |         - key: default
 102 |           raw_shape: 14
 103 |           shape: 14
 104 |       num_frames: 33
 105 |       global_sample_stride: 1
 106 |       action_video_freq_ratio: 4
 107 |       video_size:
 108 |       - 384
 109 |       - 320
 110 |       camera_key: null
 111 |       val_set_proportion: 0.0
 112 |       is_training_set: true
 113 |       skip_padding_as_possible: false
 114 |       concat_multi_camera: robotwin
 115 |       override_instruction: Put the three objects on the table into the container.
 116 |       processor:
 117 |         _target_: fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
 118 |         shape_meta:
 119 |           images:
 120 |           - key: cam_high
 121 |             raw_shape:
 122 |             - 3
 123 |             - 480
 124 |             - 640
 125 |             shape:
 126 |             - 3
 127 |             - 240
 128 |             - 320
 129 |           - key: cam_left_wrist
 130 |             raw_shape:
 131 |             - 3
 132 |             - 480
 133 |             - 640
 134 |             shape:
 135 |             - 3
 136 |             - 240
 137 |             - 320
 138 |           - key: cam_right_wrist
 139 |             raw_shape:
 140 |             - 3
 141 |             - 480
 142 |             - 640
 143 |             shape:
 144 |             - 3
 145 |             - 240
 146 |             - 320
 147 |           action:
 148 |           - key: default
 149 |             raw_shape: 14
 150 |             shape: 14
 151 |           state:
 152 |           - key: default
 153 |             raw_shape: 14
 154 |             shape: 14
 155 |         num_obs_steps: 33
 156 |         num_output_cameras: 3
 157 |         action_output_dim: 14
 158 |         proprio_output_dim: 14
 159 |         action_state_transforms: null
 160 |         use_stepwise_action_norm: false
 161 |         norm_default_mode: z-score
 162 |         norm_exception_mode: null
 163 |         action_state_merger:
 164 |           _target_: fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
 165 |         train_transforms:
 166 |         - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
 167 |         - _target_: torchvision.transforms.Resize
 168 |           size:
 169 |           - 240
 170 |           - 320
 171 |         val_transforms:
 172 |         - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
 173 |         - _target_: torchvision.transforms.Resize
 174 |           size:
 175 |           - 240
 176 |           - 320
 177 |         image_augmentation: true
 178 |         train_aug_transforms:
 179 |           cam_high:
 180 |           - _target_: torchvision.transforms.RandomCrop
 181 |             size:
 182 |             - 228
 183 |             - 304
 184 |           - _target_: torchvision.transforms.Resize
 185 |             size:
 186 |             - 240
 187 |             - 320
 188 |           - _target_: torchvision.transforms.RandomRotation
 189 |             degrees: 5
 190 |           - _target_: torchvision.transforms.ColorJitter
 191 |             brightness: 0.3
 192 |             contrast: 0.4
 193 |             saturation: 0.5
 194 |           cam_left_wrist:
 195 |           - _target_: torchvision.transforms.ColorJitter
 196 |             brightness: 0.3
 197 |             contrast: 0.4
 198 |             saturation: 0.5
 199 |           cam_right_wrist:
 200 |           - _target_: torchvision.transforms.ColorJitter
 201 |             brightness: 0.3
 202 |             contrast: 0.4
 203 |             saturation: 0.5
 204 |       text_embedding_cache_dir: /project/peilab/wam/physical_WM/data/text_embeds_cache/pack_3_objects_plus
 205 |       context_len: 128
 206 |   val: null
 207 | model:
 208 |   _target_: fastwam.runtime.create_fastwam
 209 |   model_id: /project/peilab/wam/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B
 210 |   tokenizer_model_id: /project/peilab/wam/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B
 211 |   dit_checkpoint_path: null
 212 |   tokenizer_max_len: 128
 213 |   load_text_encoder: false
 214 |   proprio_dim: 14
 215 |   proprio_dims: null
 216 |   redirect_common_files: false
 217 |   mot_checkpoint_mixed_attn: true
 218 |   action_dit_pretrained_path: /project/peilab/wam/physical_WM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
 219 |   skip_dit_load_from_pretrain: false
 220 |   video_dit_config:
 221 |     has_image_input: false
 222 |     patch_size:
 223 |     - 1
 224 |     - 2
 225 |     - 2
 226 |     in_dim: 48
 227 |     hidden_dim: 3072
 228 |     ffn_dim: 14336
 229 |     freq_dim: 256
 230 |     text_dim: 4096
 231 |     out_dim: 48
 232 |     num_heads: 24
 233 |     attn_head_dim: 128
 234 |     num_layers: 30
 235 |     eps: 1.0e-06
 236 |     seperated_timestep: true
 237 |     require_clip_embedding: false
 238 |     require_vae_embedding: false
 239 |     fuse_vae_embedding_in_latents: true
 240 |     use_gradient_checkpointing: true
 241 |     video_attention_mask_mode: first_frame_causal
 242 |     action_conditioned: false
 243 |     action_dim: 14
 244 |     action_group_causal_mask_mode: group_diagonal
 245 |   action_dit_config:
 246 |     action_dim: 14
 247 |     action_head_mode: shared
 248 |     num_embodiments: 1
 249 |     action_dims: null
 250 |     hidden_dim: 1024
 251 |     ffn_dim: 4096
 252 |     num_heads: 24
 253 |     attn_head_dim: 128
 254 |     num_layers: 30
 255 |     text_dim: 4096
 256 |     freq_dim: 256
 257 |     eps: 1.0e-06
 258 |     use_gradient_checkpointing: true
 259 |   video_scheduler:
 260 |     train_shift: 5.0
 261 |     infer_shift: 5.0
 262 |     num_train_timesteps: 1000
 263 |   action_scheduler:
 264 |     train_shift: 5.0
 265 |     infer_shift: 5.0
 266 |     num_train_timesteps: 1000
 267 |   loss:
 268 |     lambda_action: 1.0
 269 |     lambda_video: 1.0
 270 |   spectral_plan:
 271 |     enabled: true
 272 |     training_mode: joint
 273 |     detach_backbone: true
 274 |     routing_mode: fixed
 275 |     horizon: 8
 276 |     latent_dim: 32
 277 |     low_frequencies: 4
 278 |     normalization_stats_path: /project/peilab/wam/GR-LAM/artifacts/piper_grlam/pack_3_objects_plus/normalization/gr_dct_k4_train.pt
 279 |     plan_dim: 1024
 280 |     plan_heads: 8
 281 |     plan_depth: 1
 282 |     adapter_layers:
 283 |     - 9
 284 |     - 19
 285 |     adapter_init_scale: 0.05
 286 |     lambda_plan: 0.1
 287 | freeze_video_expert: false
```

### P:checkpoints/fastwam_stack_3_cups/config.yaml（源行 1–347）

```yaml
   1 | output_dir: ./runs/stack_3_cups_ft_28dpartial_worstconf_uncond_3cam_384_1e-4/2026-06-15_19-44-58
   2 | batch_size: 8
   3 | num_workers: 2
   4 | lr_scheduler_type: cosine
   5 | learning_rate: 0.0001
   6 | num_epochs: 100
   7 | max_steps: 20000
   8 | log_every: 10
   9 | save_every: 2000
  10 | eval_every: 500
  11 | eval_num_inference_steps: 10
  12 | gradient_accumulation_steps: 1
  13 | mixed_precision: bf16
  14 | seed: 42
  15 | max_grad_norm: 1.0
  16 | weight_decay: 0.01
  17 | resume: /project/peilab/wam/physical_WM/runs/stack_3_cups_ft_28dpartial_worstconf_uncond_3cam_384_1e-4/2026-06-14_20-40-03/checkpoints/state/step_010000
  18 | partial_resume: null
  19 | wandb:
  20 |   enabled: true
  21 |   workspace: null
  22 |   project: fast-wam-real
  23 |   name: stack_3_cups_ft_28dpartial_worstconf_uncond_3cam_384_1e-4
  24 |   group: null
  25 |   mode: online
  26 | data:
  27 |   train:
  28 |     _target_: fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
  29 |     dataset_dirs:
  30 |     - ./data/stack_3_cups/perfect_lerobot
  31 |     shape_meta:
  32 |       images:
  33 |       - key: cam_high
  34 |         raw_shape:
  35 |         - 3
  36 |         - 480
  37 |         - 640
  38 |         shape:
  39 |         - 3
  40 |         - 240
  41 |         - 320
  42 |       - key: cam_left_wrist
  43 |         raw_shape:
  44 |         - 3
  45 |         - 480
  46 |         - 640
  47 |         shape:
  48 |         - 3
  49 |         - 240
  50 |         - 320
  51 |       - key: cam_right_wrist
  52 |         raw_shape:
  53 |         - 3
  54 |         - 480
  55 |         - 640
  56 |         shape:
  57 |         - 3
  58 |         - 240
  59 |         - 320
  60 |       action:
  61 |       - key: default
  62 |         raw_shape: 14
  63 |         shape: 14
  64 |       state:
  65 |       - key: default
  66 |         raw_shape: 14
  67 |         shape: 14
  68 |     num_frames: 33
  69 |     global_sample_stride: 1
  70 |     action_video_freq_ratio: 4
  71 |     video_size:
  72 |     - 384
  73 |     - 320
  74 |     camera_key: null
  75 |     val_set_proportion: 0.01
  76 |     is_training_set: true
  77 |     pretrained_norm_stats: /project/peilab/wam/physical_WM/runs/stack_3_cups_ft_28dpartial_worstconf_uncond_3cam_384_1e-4/2026-06-14_20-40-03/dataset_stats.json
  78 |     skip_padding_as_possible: false
  79 |     concat_multi_camera: robotwin
  80 |     processor:
  81 |       _target_: fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
  82 |       shape_meta:
  83 |         images:
  84 |         - key: cam_high
  85 |           raw_shape:
  86 |           - 3
  87 |           - 480
  88 |           - 640
  89 |           shape:
  90 |           - 3
  91 |           - 240
  92 |           - 320
  93 |         - key: cam_left_wrist
  94 |           raw_shape:
  95 |           - 3
  96 |           - 480
  97 |           - 640
  98 |           shape:
  99 |           - 3
 100 |           - 240
 101 |           - 320
 102 |         - key: cam_right_wrist
 103 |           raw_shape:
 104 |           - 3
 105 |           - 480
 106 |           - 640
 107 |           shape:
 108 |           - 3
 109 |           - 240
 110 |           - 320
 111 |         action:
 112 |         - key: default
 113 |           raw_shape: 14
 114 |           shape: 14
 115 |         state:
 116 |         - key: default
 117 |           raw_shape: 14
 118 |           shape: 14
 119 |       num_obs_steps: 33
 120 |       num_output_cameras: 3
 121 |       action_output_dim: 14
 122 |       proprio_output_dim: 14
 123 |       action_state_transforms: null
 124 |       use_stepwise_action_norm: false
 125 |       norm_default_mode: z-score
 126 |       norm_exception_mode: null
 127 |       action_state_merger:
 128 |         _target_: fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
 129 |       train_transforms:
 130 |       - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
 131 |       - _target_: torchvision.transforms.Resize
 132 |         size:
 133 |         - 240
 134 |         - 320
 135 |       val_transforms:
 136 |       - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
 137 |       - _target_: torchvision.transforms.Resize
 138 |         size:
 139 |         - 240
 140 |         - 320
 141 |       image_augmentation: true
 142 |       train_aug_transforms:
 143 |         cam_high:
 144 |         - _target_: torchvision.transforms.RandomCrop
 145 |           size:
 146 |           - 228
 147 |           - 304
 148 |         - _target_: torchvision.transforms.Resize
 149 |           size:
 150 |           - 240
 151 |           - 320
 152 |         - _target_: torchvision.transforms.RandomRotation
 153 |           degrees: 5
 154 |         - _target_: torchvision.transforms.ColorJitter
 155 |           brightness: 0.3
 156 |           contrast: 0.4
 157 |           saturation: 0.5
 158 |         cam_left_wrist:
 159 |         - _target_: torchvision.transforms.ColorJitter
 160 |           brightness: 0.3
 161 |           contrast: 0.4
 162 |           saturation: 0.5
 163 |         cam_right_wrist:
 164 |         - _target_: torchvision.transforms.ColorJitter
 165 |           brightness: 0.3
 166 |           contrast: 0.4
 167 |           saturation: 0.5
 168 |     text_embedding_cache_dir: ./data/text_embeds_cache/pack_3_objects_plus_stack_3_cups
 169 |     context_len: 128
 170 |   val:
 171 |     _target_: fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
 172 |     dataset_dirs:
 173 |     - ./data/stack_3_cups/perfect_lerobot
 174 |     shape_meta:
 175 |       images:
 176 |       - key: cam_high
 177 |         raw_shape:
 178 |         - 3
 179 |         - 480
 180 |         - 640
 181 |         shape:
 182 |         - 3
 183 |         - 240
 184 |         - 320
 185 |       - key: cam_left_wrist
 186 |         raw_shape:
 187 |         - 3
 188 |         - 480
 189 |         - 640
 190 |         shape:
 191 |         - 3
 192 |         - 240
 193 |         - 320
 194 |       - key: cam_right_wrist
 195 |         raw_shape:
 196 |         - 3
 197 |         - 480
 198 |         - 640
 199 |         shape:
 200 |         - 3
 201 |         - 240
 202 |         - 320
 203 |       action:
 204 |       - key: default
 205 |         raw_shape: 14
 206 |         shape: 14
 207 |       state:
 208 |       - key: default
 209 |         raw_shape: 14
 210 |         shape: 14
 211 |     num_frames: 33
 212 |     global_sample_stride: 1
 213 |     action_video_freq_ratio: 4
 214 |     video_size:
 215 |     - 384
 216 |     - 320
 217 |     camera_key: null
 218 |     val_set_proportion: 0.01
 219 |     is_training_set: false
 220 |     pretrained_norm_stats: /project/peilab/wam/physical_WM/runs/stack_3_cups_ft_28dpartial_worstconf_uncond_3cam_384_1e-4/2026-06-14_20-40-03/dataset_stats.json
 221 |     skip_padding_as_possible: false
 222 |     concat_multi_camera: robotwin
 223 |     processor:
 224 |       _target_: fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
 225 |       shape_meta:
 226 |         images:
 227 |         - key: cam_high
 228 |           raw_shape:
 229 |           - 3
 230 |           - 480
 231 |           - 640
 232 |           shape:
 233 |           - 3
 234 |           - 240
 235 |           - 320
 236 |         - key: cam_left_wrist
 237 |           raw_shape:
 238 |           - 3
 239 |           - 480
 240 |           - 640
 241 |           shape:
 242 |           - 3
 243 |           - 240
 244 |           - 320
 245 |         - key: cam_right_wrist
 246 |           raw_shape:
 247 |           - 3
 248 |           - 480
 249 |           - 640
 250 |           shape:
 251 |           - 3
 252 |           - 240
 253 |           - 320
 254 |         action:
 255 |         - key: default
 256 |           raw_shape: 14
 257 |           shape: 14
 258 |         state:
 259 |         - key: default
 260 |           raw_shape: 14
 261 |           shape: 14
 262 |       num_obs_steps: 33
 263 |       num_output_cameras: 3
 264 |       action_output_dim: 14
 265 |       proprio_output_dim: 14
 266 |       action_state_transforms: null
 267 |       use_stepwise_action_norm: false
 268 |       norm_default_mode: z-score
 269 |       norm_exception_mode: null
 270 |       action_state_merger:
 271 |         _target_: fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
 272 |       train_transforms:
 273 |       - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
 274 |       - _target_: torchvision.transforms.Resize
 275 |         size:
 276 |         - 240
 277 |         - 320
 278 |       val_transforms:
 279 |       - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
 280 |       - _target_: torchvision.transforms.Resize
 281 |         size:
 282 |         - 240
 283 |         - 320
 284 |     text_embedding_cache_dir: ./data/text_embeds_cache/pack_3_objects_plus_stack_3_cups
 285 |     context_len: 128
 286 | model:
 287 |   _target_: fastwam.runtime.create_fastwam
 288 |   model_id: Wan-AI/Wan2.2-TI2V-5B
 289 |   tokenizer_model_id: Wan-AI/Wan2.2-TI2V-5B
 290 |   dit_checkpoint_path: null
 291 |   tokenizer_max_len: 128
 292 |   load_text_encoder: false
 293 |   proprio_dim: 14
 294 |   proprio_dims: null
 295 |   redirect_common_files: false
 296 |   mot_checkpoint_mixed_attn: false
 297 |   action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
 298 |   skip_dit_load_from_pretrain: false
 299 |   video_dit_config:
 300 |     has_image_input: false
 301 |     patch_size:
 302 |     - 1
 303 |     - 2
 304 |     - 2
 305 |     in_dim: 48
 306 |     hidden_dim: 3072
 307 |     ffn_dim: 14336
 308 |     freq_dim: 256
 309 |     text_dim: 4096
 310 |     out_dim: 48
 311 |     num_heads: 24
 312 |     attn_head_dim: 128
 313 |     num_layers: 30
 314 |     eps: 1.0e-06
 315 |     seperated_timestep: true
 316 |     require_clip_embedding: false
 317 |     require_vae_embedding: false
 318 |     fuse_vae_embedding_in_latents: true
 319 |     use_gradient_checkpointing: false
 320 |     video_attention_mask_mode: first_frame_causal
 321 |     action_conditioned: false
 322 |     action_dim: 14
 323 |     action_group_causal_mask_mode: group_diagonal
 324 |   action_dit_config:
 325 |     action_dim: 14
 326 |     action_head_mode: shared
 327 |     num_embodiments: 1
 328 |     action_dims: null
 329 |     hidden_dim: 1024
 330 |     ffn_dim: 4096
 331 |     num_heads: 24
 332 |     attn_head_dim: 128
 333 |     num_layers: 30
 334 |     text_dim: 4096
 335 |     freq_dim: 256
 336 |     eps: 1.0e-06
 337 |     use_gradient_checkpointing: false
 338 |   video_scheduler:
 339 |     train_shift: 5.0
 340 |     infer_shift: 5.0
 341 |     num_train_timesteps: 1000
 342 |   action_scheduler:
 343 |     train_shift: 5.0
 344 |     infer_shift: 5.0
 345 |     num_train_timesteps: 1000
 346 |   loss:
 347 |     lambda_action: 1.0
```

## 附录E：小型文件与源码指纹

此表SHA256针对本次磁盘原文件；部署YAML哈希对应脱敏前原文件，仅用于本机版本识别，传输脱敏文件应另算manifest，报告不披露凭证。源码位置使用前缀+相对路径；行号由附录代码和正文定位。

|文件|字节数|SHA256|
|---|---:|---|
|`S:scripts/serve_policy.py`|1256|`996f324499dacda4422a460647ad5dae1a217e9d8775c997facedd8c877f7a04`|
|`S:scripts/deploy_real_rtc_wam.py`|4471|`6e02356c1f2437ad1622ba356c2fdc83d5f7368590c2db2d5c6736f5c8b89a17`|
|`S:scripts/smoke_joint_pack3_local.py`|2768|`96140669112e2c3f2d2e29ca7c343d85259bb6da4b82017d2bb2f036caaf58be`|
|`S:configs/real_deploy_joint_pack3.yaml`|2746|`9622d1f46adb023f555cfcd101ad3be1b61b8e2212a3a6aa96addd1769b18631`|
|`S:configs/real_deploy_joint_stack3.yaml`|2862|`ae4e1216691a7df5776bca01a994bfb5e9380f37f981ead0abe2bb7b79f1cca5`|
|`S:configs/train_config_joint_pack3_infer.yaml`|3316|`b25f1650faf28d71311b310904a21f05d1ba1490fd0f890f7735ecaf25b71952`|
|`S:configs/train_config_joint_stack3_infer.yaml`|2934|`2c78e6053b0e59347a7c10075098e1661aa227c4cdda6038c4714a243bb697db`|
|`S:src/qi/deployment/wam_policy_client.py`|14777|`0621ddebcb47de8c7c62e3efe86efb3bbfc9420a1860aea3cf5ad6ae76a18367`|
|`S:src/qi/deployment/wam_policy_server.py`|2511|`0433edf6706040f11e92973e49b0128564e1bbd0157fafac5ee9f1f78da0866c`|
|`S:src/qi/deployment/wam_remote_policy_client.py`|1968|`ed7309a63d6ac6b07a9530dec25b7654f0135a7089f3262010df15a71cf0e8b3`|
|`S:src/qi/deployment/robot_runtime.py`|21917|`9e5646967693ad74e20d06cb2bdfd04f6176aa991348a857636f5d7e2e0b38eb`|
|`S:src/qi/deployment/rtc_buffer.py`|3855|`6c79307a98f1952bd218869e505159c01ab51c569c3d68384ca16cb49c505562`|
|`S:src/qi/runtime.py`|17000|`9b0cef57561c7c4df6513dd91754815b710b382742987a670183e1bcec4db370`|
|`S:src/qi/models/wan22/fastwam.py`|97144|`60b3cf2c62c3d3ebd8ef61c3afb8e6e6b4de228e5fc3286cc0abcb65e278e1d8`|
|`S:src/qi/models/wan22/fastwam_joint.py`|14974|`5cfbb17c444ed7807a70164646f578992e5172eb6a55f2ab1ab94789511e68e2`|
|`S:src/qi/models/wan22/mot.py`|50289|`d7d8fd07beb8fb96ea67c98a3c5c065c63b813f232c1ffdbdfe4b02a1fb737d6`|
|`S:src/qi/models/wan22/action_dit.py`|13678|`306ad0417c5a8a13125e1bb2df071bae16cd9a9f0f2e29db37a95eff4830b7e5`|
|`S:src/qi/models/wan22/wan_video_dit.py`|31918|`4249f1ecb71891c60a99934e2ba669b482ebf283d9978091cfdcefd44e2dede6`|
|`S:src/qi/models/wan22/wan_video_vae.py`|50058|`31b3fb9f844a76f5db5af4854e14802aae640b85b61d0de03a4b53b52515abb7`|
|`S:src/qi/models/wan22/helpers/loader.py`|8273|`57a186275c646c1218ae11bac5ad1e4d2ddf48f35ebc993af088f9b064d40a7d`|
|`S:src/qi/models/wan22/schedulers/scheduler_continuous.py`|4068|`2ececce284f7b9fe876965fbf5e11ca7b69caaf693979266b0f63062a589f181`|
|`S:src/qi/datasets/lerobot/robot_video_dataset.py`|12348|`014648b1a000f15eea003ed31024c938688a5ba561363452357112559207eb5f`|
|`S:src/qi/datasets/lerobot/base_lerobot_dataset.py`|19648|`259dfca4c70ea639fcc44fe9de530c6b17dd4d7c5f925b3f552c36ab3a67df44`|
|`S:src/qi/datasets/lerobot/processors/fastwam_processor.py`|13306|`63ae2b597d9c1b274d0469ea3adabcd3ccbc71483246bfe5f092b41d10f9a7d7`|
|`S:src/qi/datasets/lerobot/utils/normalizer.py`|11264|`221d9b1ac78f4b4730ba44dd6e864e718cf6aa9604571315c196219d81d28ba8`|
|`Q:MIGRATION_WORKFLOW.md`|22016|`75bf815769e9c282e40277357710a6b48d9e0f128f07fd0dc07ea9b69e0e01b1`|
|`Q:src/fastwam_steerquant/adapters/base.py`|3690|`3ec9628ea7eb59c74eded7d5b603ce6ce50df9616d64bfb9d9305fedc6e6a796`|
|`Q:src/fastwam_steerquant/adapters/fastwam.py`|2582|`c2c0892361d24e8fea76173863d8741f5e27f685ef2df3849ed3251307ce8185`|
|`Q:src/fastwam_steerquant/state.py`|4977|`d0e71a239b07c1ab378a3c99b4f7e30ede293f81820001d056e84534debc8ef7`|
|`Q:src/fastwam_steerquant/deployment.py`|10295|`7c4027645b34332358f8be782e7aa0db3ba6bb8cbd38b26db564e96369ccb56a`|
|`Q:src/fastwam_steerquant/policy.py`|2476|`643962a0b2e4c37e92b6a99419791ee42d77668f3fcb585ddf52dea748e9d498`|
|`Q:src/fastwam_steerquant/topology.py`|2791|`d0109c2b80d18bc559516bd2a1722957e43667325b441e01779b4ff2aae08562`|
|`Q:src/fastwam_steerquant/streams.py`|4257|`445f8210d3845df5cf0aa843b1a7a856548f2750d5cca78b82bf0609487e28a8`|
|`Q:src/fastwam_steerquant/rollout_graph.py`|9174|`03c7e180b6780ac5b4de3b8c761863456ee1a654b958612813b9124b336e4fbb`|
|`P:scripts/data_process/convert_real_hdf5_to_lerobot.py`|10398|`ef2166371d35f1e2161f901a4d4dd8895f2a2351794da3077f5d58a72e4998ec`|
|`P:checkpoints/10step_joint/config.yaml`|7642|`b65e99f5b97a34e4fd18f4c6d8b6076c67e89ebfacab10443b6434646566bd2e`|
|`P:checkpoints/k4_disjoint_pack_3_objecst/config.yaml`|8074|`13eb539dcbd2d41aec57f5b7e42ebca0b485364cb60b6b2ec1f674e7d391f43d`|
|`P:checkpoints/fastwam_stack_3_cups/config.yaml`|8642|`5b50d9f31dbea775d3d2b78c7f0462ba2d7deb5122a92fc317746bd2fb56fc58`|
|`P:checkpoints/k4_disjoint_pack_3_objecst/dataset_stats.json`|94809|`30561edb2897b9784fbf13f4e42ca9897e141b4ebd60119d047d255758aaa6bf`|
|`P:checkpoints/fastwam_stack_3_cups/dataset_stats.json`|94783|`68c662dba501b1e110e09896238bfd0ae1d15c226e2c05683958bba915f7e0cb`|
|`P:checkpoints/10step_joint/dataset_stats.json`|94809|`30561edb2897b9784fbf13f4e42ca9897e141b4ebd60119d047d255758aaa6bf`|
|`P:data/text_embeds_cache/pack_3_objects_plus/ad085ca9b81861e511916a552aef91b865cb6b0b0c6e2c7c3d7db8629eab5ba3.t5_len128.wan22ti2v5b.pt`|1051869|`927573ff154e4303f06a3a719167c198b0b4ed5ea1bd72b23768217fd8381960`|
|`P:data/text_embeds_cache/stack_3_cups_gen/072dbf0aee70fe628914d55762f3732ffb939ec8e9af90d5f0b49117f4680b0e.t5_len128.wan22ti2v5b.pt`|1051869|`a83dfdf3a215e8314206bbf5080565617c33e23622ad05a18dd3227afa5fddf6`|

文本缓存的实际安全读取结果均为context=[128,4096] BF16、mask=[128] bool。其他大型预训练资产未做内容哈希/加载验证，接收方必须固定VAE等来源版本；检查大checkpoint哈希与meta不等于完整模型严格加载成功。

## 附录F：本次交付检查

报告写入后复核：Markdown代码围栏成对；两份部署YAML的authkey均为`<LOCAL_AUTHKEY>`；报告不含密码/私钥/token值。Q Git状态仅新增本报告，S状态与审计开始时一致。没有运行模型、校准脚本、smoke、硬件入口或环境安装命令。证据中的现有脚本和将来建议命令均只作交接说明。
