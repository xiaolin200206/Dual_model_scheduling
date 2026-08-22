# 榴莲双模型边缘AI — Jetson Orin Nano Super 性能测试

在Jetson Orin Nano Super上跑双模型(叶片病害检测+虫害检测)实时推理，
对比三种调度策略(staggered / parallel / sequential)在duty-cycle与
持续运行两种模式下的表现。是跨平台研究的一部分，另一条对照线是
Raspberry Pi 5。

> 如果你还不熟悉Docker，`docker/DOCKER_NOTES.md`(英文)是专门写成
> 一份独立学习资料的——记录了在这台设备上把GPU加速推理跑进容器
> 过程中真实遇到的每一个坑，以及怎么解决的。哪怕这个具体项目跟你
> 无关，也值得读一读。

---

> 是跨平台研究的一部分——两边(RPi5那边)怎么对应起来，见
> [顶层README](../README_zh.md)。

## 仓库结构

```
.
├── main.py                  # 核心推理脚本(通过环境变量配置,见下方"运行方式")
├── requirements.txt          # 原生(不用Docker)安装时的Python依赖
├── docker/
│   ├── Dockerfile             # 已验证GPU可用的Jetson容器构建文件
│   └── DOCKER_NOTES.md        # 详细的Docker调试日记/学习笔记(英文)
├── data/
│   └── dual_*.csv             # 6组原始telemetry日志，每种配置一份
└── analysis/
    └── summarize_results.py   # 复现下方对比表的脚本
```

## 实验设计

两个独立的YOLOv11 ONNX模型同时对同一路摄像头画面做推理:

- **叶片模型**(YOLOv11s，36.2 MB)——5个病害类别
- **虫害模型**(YOLOv11n，10.1 MB)——7个虫害类别

测试了三种调度策略，每种策略下又分别测了两种duty-cycle条件
(开=180秒运行/45秒休眠，关=持续运行不休眠)，一共6组配置:

| 模式 | 说明 |
|---|---|
| `staggered` | 两个独立线程；虫害推理启动延迟0.4秒，减少时间上的重叠 |
| `parallel` | 两个独立线程；启动零延迟(重叠程度最大) |
| `sequential` | 单一worker；叶片推理跑完才开始虫害推理，同一帧上先后执行(结构上完全没有重叠) |

每组配置跑满3小时，每0.5秒记录一次FPS、各模型推理延迟、CPU利用率、
内存、温度、时钟频率，写入CSV文件。

## 结果汇总

跑`python3 analysis/summarize_results.py`可以从原始CSV重新生成这张表:

```
Config                         Rows         Duration  AvgTemp  MaxTemp  AvgCPU  MaxCPU  AvgFPS  LeafLat  PestLat
----------------------------------------------------------------------------------------------------------------
parallel_dutycycle            21584   3:02:46.962000    49.71    50.70    8.02   29.20    8.20    83.66    62.27
parallel_nodutycycle          21162   2:59:54.634000    49.12    50.80    8.94   19.50    8.19    84.23    62.43
sequential_dutycycle          21340   3:00:38.166000    48.20    50.20    6.69   32.20    8.21    85.35    34.70
sequential_nodutycycle        21145   2:59:40.579000    48.25    50.50    8.14   33.00    8.19    87.07    35.96
staggered_dutycycle           21134   2:59:00.484000    48.17    50.90    8.00  100.00    8.19    83.71    61.87
staggered_nodutycycle         21091   2:59:16.884000    50.38    51.10    8.93   30.00    8.19    83.97    62.57
```

**值得注意的现象：** `sequential`模式的虫害推理延迟明显更低
(约35ms，其余四组配置都在62ms左右)。这跟cache-exclusivity(缓存独占)
效应的预期一致——因为sequential这个worker是先跑完叶片推理再跑虫害推理，
不是同时进行，每个模型在自己的推理阶段能独占CPU/GPU缓存，比
`parallel`/`staggered`这种并发模式的缓存争用开销更低。

## 已知的数据质量问题

- **FPS异常尖峰：**`parallel_dutycycle`和`staggered_dutycycle`这两组
  里出现了少量FPS读数飙到几百的情况(其余行都稳定在约8.2)，大概率是
  duty-cycle在active/sleep切换的瞬间，FPS计算公式的分母出现了接近零
  的极端值导致的。`analysis/summarize_results.py`在算平均值之前会把
  超过20 FPS的读数当作计时异常剔除掉。
- **`staggered_dutycycle`里的CPU峰值：**同一组数据里出现过一次CPU
  瞬间冲到100%(同组其余时间都在35%以下)，跟上面那个FPS异常出现的
  时间点吻合，大概率是同一次调度切换事件导致的。目前已经标记出来，
  但还没有找到根本原因。

## 运行方式

### 原生安装(不用Docker)

```bash
pip3 install --no-deps <Jetson专属的torch/torchvision/onnxruntime-gpu wheel，见requirements.txt>
pip3 install -r requirements.txt

export SCHEDULE_MODE=sequential        # staggered | parallel | sequential
export DUTY_CYCLE_ENABLED=True         # True | False
export TELEGRAM_BOT_TOKEN=...          # 可选，不设置就不发告警
export TELEGRAM_CHAT_ID=...            # 可选
python3 main.py
```

### Docker(推荐——具体原因见docker/DOCKER_NOTES.md)

```bash
# 1. 把.onnx模型文件(yolov11s.onnx, yolov11n.onnx)放进docker/目录
cp yolov11s.onnx yolov11n.onnx docker/

# 2. 构建镜像
cd docker && sudo docker build -t durian-jetson:test .

# 3. 正式长跑之前，先验证容器里GPU真的能用
sudo docker run --rm --runtime nvidia -it durian-jetson:test \
  python3 -c "import torch; print(torch.cuda.is_available())"

# 4. 用tmux挂后台跑完整实验(这样SSH断开也不影响)
tmux new -s docker_run
timeout 10800 sudo docker run --rm --runtime nvidia --device=/dev/video0 \
  -v ~/docker_build/output:/app/output \
  -e SCHEDULE_MODE="sequential" -e DUTY_CYCLE_ENABLED="True" \
  -e TELEGRAM_BOT_TOKEN="..." -e TELEGRAM_CHAT_ID="..." \
  durian-jetson:test python3 main.py 2>&1 | tee ~/docker_build/run_log.txt
# Ctrl+B 然后按 D 退出tmux界面(进程继续在后台跑)；
# 之后用 tmux attach -t docker_run 重新连回去查看
```

## 硬件/软件环境

- **设备：**Jetson Orin Nano Super Developer Kit
- **JetPack：**6.2 (L4T R36.4.4)
- **摄像头：**USB 2K自动对焦摄像头(强制用MJPG格式@1280x720——具体
  原因见`main.py`里的代码注释，纯V4L2默认设置会导致这款摄像头的
  fps被严重限制)
- **推理引擎：**ONNX Runtime 1.23.0，CUDAExecutionProvider
- **Docker基础镜像：**`ultralytics/ultralytics:latest-jetson-jetpack6`

## 安全提醒

`main.py`里的Telegram凭证是从环境变量(`TELEGRAM_BOT_TOKEN`、
`TELEGRAM_CHAT_ID`)读取的——千万不要把这些硬编码写进脚本或者提交到
版本控制里。想知道这一点对Docker来说具体为什么重要，看
`docker/DOCKER_NOTES.md`第9节。
