# 榴莲边缘AI——跨平台调度与缓存独占性研究

对双模型(叶片病害检测+虫害检测)实时边缘推理做跨平台性能测试，对比
三种调度策略(staggered / parallel / sequential)在duty-cycle与持续
运行两种模式下的表现，覆盖两个硬件平台:

- **Jetson Orin Nano Super**(JetPack 6.2，CUDA加速)——[`jetson/`](jetson/)——**已完成，6/6组**
- **Raspberry Pi 5**(纯CPU)——[`raspberry-pi/`](raspberry-pi/)——**尚未开始**

> **English version: [README.md](README.md)**

## 为什么要跨平台

Jetson这边的结果(详见`jetson/README.md`)显示出一个一致的模式：
`sequential`调度下虫害模型的推理延迟明显低于`parallel`/`staggered`
模式(约35ms vs 约62ms)，这跟cache-exclusivity(缓存独占性)效应的
预期一致——两个模型先后执行、不是同时进行，就不会互相把对方的数据
挤出CPU/GPU缓存。

这个仓库要回答的核心问题是：**这个效应在缓存更小的平台(RPi5的
Cortex-A76)上还成不成立，还是只是Jetson这套缓存架构特有的现象？**
如果sequential的优势在RPi5上缩小甚至消失，这就证明这个效应是跟
缓存大小挂钩的，不是调度策略本身固有的属性——这比单纯在两个平台上
重复验证同一个结论更有意思，也更有发表价值。

## 仓库结构

```
.
├── README.md / README_zh.md     # 本文件
├── compare_platforms.py          # 跨平台对比脚本
│                                  # (随时可以跑——RPi5数据没跑完的
│                                  # 时候会优雅地显示"暂无数据")
├── jetson/                       # Jetson Orin Nano Super——已完成
│   ├── README.md / README_zh.md
│   ├── main.py
│   ├── docker/
│   ├── data/                     # 6组CSV，全部配置都有
│   └── analysis/
└── raspberry-pi/                 # Raspberry Pi 5——尚未开始
    ├── README.md
    ├── main.py                   # 已就绪，跟jetson/main.py逻辑对称
    ├── docker/                   # 已就绪
    ├── data/                     # 空，等实验跑完
    └── analysis/
```

两个平台目录都是自包含的(各自有README、脚本、Dockerfile、数据)，
可以单独打开其中一个来用——`jetson/README.md`和
`raspberry-pi/README.md`里有各自平台的完整运行说明。这份顶层README
只讲两边之间共通、需要保持一致的部分。

## 实验协议(两平台完全一致的部分)

为了保证对比有效，以下这些参数在两个平台之间是固定不变的——只有
硬件本身、摄像头接口、ONNX Runtime的执行引擎这几项允许不同:

| 参数 | 数值 |
|---|---|
| 叶片模型 | YOLOv11s ONNX，5个类别 |
| 虫害模型 | YOLOv11n ONNX，7个类别 |
| 置信度阈值 | 0.35 |
| 推理分辨率 | 640×640 |
| 叶片推理间隔 | 0.8秒 |
| 虫害推理间隔 | 1.2秒 |
| Duty cycle(开启时) | 180秒运行/45秒休眠 |
| 单组运行时长 | 3小时 |
| Telemetry记录间隔 | 0.5秒 |
| 测试配置数 | 3种调度模式×2种duty-cycle状态=6组 |

**允许不同的部分：**摄像头接口(Jetson用USB/MJPG；RPi5用
Picamera2/CSI)、温度读取路径(平台各自的sysfs路径)、ONNX Runtime
执行引擎(Jetson用CUDA，RPi5用CPU——这个本身就是研究要对比的变量，
不是需要控制掉的干扰因素)。

## 怎么跑对比

等RPi5数据收集完(把CSV放进`raspberry-pi/data/`，命名规则跟
`jetson/data/`保持一致)，在仓库根目录跑:

```bash
python3 compare_platforms.py
```

这会按配置名把两个平台的结果拼在一起，打印出每组配置的差值对比
(温度、CPU、FPS、各模型延迟)。不需要改代码，新CSV放进去会自动识别。

## 目前进度

- [x] Jetson：6组配置全部完成，每组3小时，GPU加速
- [x] Jetson：Docker容器化已验证(CUDA passthrough正常工作)
- [ ] RPi5：6组配置——已规划，尚未执行
- [ ] RPi5：Docker容器化——Dockerfile已就绪，尚未测试
- [ ] Field test现场验证(两个平台)——已规划

## 安全提醒

`jetson/main.py`和`raspberry-pi/main.py`都是从环境变量读取Telegram
告警凭证的——千万不要把token硬编码进脚本或者提交到版本控制里。
详见`jetson/docker/DOCKER_NOTES.md`第9节。
