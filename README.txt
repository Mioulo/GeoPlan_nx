GeoPlan_nx：独立的 116 TensorRT 部署仓库

本仓库包含部署代码、配置、116 权重及两个 ONNX；无需原 YOPO-RGB 仓库。
模型文件由 Git LFS 管理；TensorRT engine 在目标 NX 上构建，不提交 Git。
保留上游 YOPO-RGB 的 MIT LICENSE；本仓库复用其推理和规划代码。

NX 获取仓库（须先安装 Git LFS）：
  git lfs install
  git clone git@github.com:Mioulo/GeoPlan_nx.git
  cd GeoPlan_nx
  git lfs pull

在具备 ROS1、CUDA PyTorch、TensorRT 的 NX 环境中：
  bash export.sh --build_only
  source /opt/ros/noetic/setup.bash
  bash run.sh --check_only
  bash run.sh

PyTorch 仿真测试（不需要 TensorRT engine）：
  source /opt/ros/noetic/setup.bash
  bash run_pt.sh --check_only
  bash run_pt.sh

run_pt.sh 使用相同 deployment.yaml、116/epoch160.pth 和双帧特征缓存。
也可 bash run.sh --backend torch。默认 bash run.sh 仍为 TensorRT。
PyTorch 路径不读取或构建 engine，不需要安装 TensorRT Python 包。
需要兼容的 torch、depth-capable ultralytics==8.4.174 及已有 ROS/Python 依赖。
有 CUDA 时使用 GPU，无 CUDA 时可用 CPU 检查，但实时规划频率需另测。
仿真图像、里程计和控制器照常运行，关闭原 YOPO 节点以免重复发布命令。
默认等待目标消息；用 RViz 的 2D Nav Goal 设置目标后开始规划。

先沿用 deployment.yaml 中的仿真话题验证，再修改为实物话题。
TRT 程序输出 PositionCommand，正常轨迹由独立的 NetworkControl 将加速度和 yaw
转换为姿态及推力；NetworkControl/飞控程序不包含在本仓库中。
具体依赖、输入约定和验证限制见下文。

本地仓库推送：
  git push -u origin main
Git LFS 的模型对象会随 push 一起上传；下载端必须使用 Git LFS 拉取完整模型。

116 / epoch160 TensorRT 实物部署程序（ROS1）

本目录提供 deploy_trt.py + deployment.yaml + run.sh。
运行只加载 encoder/planner 两个 TensorRT engine，不构造 PyTorch 网络，
不加载 .pth；torch 仅用于 CUDA 张量、流和后处理。复用原节点的规划、
Top8 代价重排和位置命令插值方法，避免重写 116 算法。

打包（仓库根目录）：
  python3 deploy/nx116/pack.py /tmp/yopo116_trt_nx
将整个目录拷到 NX。运行时读取 checkpoint.sha256 校验 engine 的权重身份。
包内 .pth 和导出脚本仅供导出/误差验证，不参与部署节点的模型执行。

构建 TensorRT：
  可在 PC 执行 bash export.sh --no_engine，得到 116 的两个 ONNX 和 .onnx.json。
  将 ONNX 和各自 JSON 复制到 NX 包根目录。
  在 NX 执行 bash export.sh --build_only，使用该设备 TensorRT 构建 FP16 engine。
  如果 NX 具备兼容的 PyTorch 模型构造环境，可加 --verify_after_build，
  或 bash export.sh 一次完成导出、构建及误差/延迟验证。
  验证环节需要 depth-capable ultralytics==8.4.174 和匹配的 Python/torch；
  仅从 ONNX build_only 和启动部署程序不需要构造 YOLO 模型。
  两个 engine 和两个 .engine.json 必须保持配套，禁止复用 112/106 engine。
  116 的 engine 当前尚未在 NX 上构建，必须在目标设备完成此步骤。

部署环境：
  ROS1 rospy、cv_bridge、sensor_msgs、nav_msgs、geometry_msgs、std_msgs、genpy；
  JetPack 对应的 TensorRT Python 包、支持 CUDA 的 Jetson PyTorch；
  numpy、scipy、ruamel.yaml、OpenCV。
  ONNX 构建另需 onnx。不要安装训练 requirements.txt 的 x86 cu118 固定版本。
  不需要训练、数据集、外部深度、地图、open3d、tensorboard、深度专家程序。

配置：
  编辑 deployment.yaml，配置 rgb_topic、odom_topic、goal_topic、ctrl_topic、
  相机 pitch_angle_deg、plan_from_reference 和目标高度。
  默认为现有仿真话题，先在仿真验证，再换成实物话题。
  模型配置保留 116 的 rgb_visual_fusion=false、wc=1.5、Top8 策略。
  initial_goal=null 时等待目标消息；不会沿用仿真入口写死的 [40,0,1.2]。
  goal_height=null 使用目标消息的 Z；不需要评估程序或 flight_enable 消息。

NX 启动：
  source /opt/ros/noetic/setup.bash
  bash run.sh --check_only
  bash run.sh
  也可 python3 deploy_trt.py --config /absolute/path/deployment.yaml
  PYTHON=/absolute/path/python bash run.sh 可选择已有环境。
  check_only 检查文件哈希、双 engine 配对、模型身份、配置和三输出语义，
  不反序列化 engine，不替代真实 TensorRT 推理验证。
  CUDA 不可用时直接报错，无 CPU/PyTorch 模型回退。

输入输出：
  /rgb/image              sensor_msgs/Image，原始图像，非 compressed
  /sim/odom               nav_msgs/Odometry
  /move_base_simple/goal  geometry_msgs/PoseStamped
  /so3_control/pos_cmd    quadrotor_msgs/PositionCommand
  control_msg 的消息 MD5 为 4712f0609ca29a79af79a35ca3e3967a，控制端需匹配。
  50Hz 是命令插值频率，规划由图像触发，NX 上实际推理频率需要实测。

仿真接入：
  原电脑继续运行模拟器和控制器，关闭原 YOPO 节点，避免重复发布。
  NX 和仿真端共用 ROS_MASTER_URI，ROS_IP/ROS_HOSTNAME 必须相互可访问。
  两边使用一致的时间基准；RViz 2D Nav Goal 或已有目标发布程序发送目标。

实物接入约定：
  当前规划代码按世界坐标速度解读 odom.twist.twist.linear，姿态为机体到世界。
  常见里程计可能发布机体系速度，需要在接入端转换，不能只改话题名。
  图像转换 bgr8 -> 480x270 -> /255；相机 FOV、畸变与轴方向需匹配训练输入。
  当前实现用最新里程计配图像，没有按图像时间戳插值或通用 TF 外参变换。
  pitch_angle_deg 仅覆盖原模型的俯仰角约定，不等于任意六自由度相机外参。
  本程序输出 PositionCommand 给已有 NetworkControl 姿态控制链路，不包含 MAVROS 转换、
  飞控解锁、起飞、模式切换；控制器/飞控需维持原有失联处置。
  规划速度取训练配置 6m/s，更改速度必须重新导出构建，旧 engine 会拒绝误配。

验证范围：
  已完成的 CPU 模型推理检查不代表 TensorRT 或实物闭环通过。
  仍需 NX 上验证 engine 反序列化、三输出误差、输入输出联通、耗时和内存。
