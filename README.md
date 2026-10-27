# Ascend Model Pipeline

面向昇腾 NPU 的模型转换与 ACL Runtime 推理代码。仓库将模型无关的公共工具与模型专用的导出、预处理和后处理分开，当前已完成 PointPillars KITTI 三分类模型的完整链路：

```text
PTH → ONNX → OM → AISBench 性能测试 / ACL Runtime 真实点云推理
```

当前实现采用 CPU 完成 PointPillars 的 voxelization、Scatter 和后处理，在 Ascend 310P1 上执行两个 OM：

- `pillar_encoder.om`
- `backbone_head.om`

## 目录结构

```text
.
├── common/
│   ├── acl_runner.py
│   ├── check_acl.py
│   └── convert_onnx_to_om.py
├── pointpillars/
│   ├── export_onnx.py
│   ├── preprocess.py
│   ├── postprocess.py
│   └── run.py
├── requirements-export.txt
└── README.md
```

`common/` 中的脚本可复用于其他静态输入形状的 OM 模型。`pointpillars/` 中的代码负责当前模型的网络导出、点云预处理、Anchor 解码和旋转 NMS。

模型权重、ONNX、OM、测试数据、日志和推理结果不会提交到仓库。使用时按下面的结构自行准备：

```text
pointpillars/
├── weights/
│   └── pointpillars_kitti_3class.pth
├── onnx/
├── om/
├── data/
│   └── 000008.bin
├── results/
└── logs/
```

## 已验证环境

- Ascend 310P1
- aarch64 openEuler 22.03
- CANN 9.2.0
- Python 3.12
- PyTorch 2.10 CPU
- NumPy 1.26.4
- ONNX 1.23.2

ACL Python 模块由 CANN Toolkit 提供，不通过 PyPI 安装。运行前需要加载 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

检查 CANN、NPU 和 ACL：

```bash
npu-smi info
command -v atc
python3 common/check_acl.py
```

## 安装 ONNX 导出依赖

建议在独立 Python 环境中安装：

```bash
python3 -m pip install -r requirements-export.txt
```

实际版本应与当前 CANN 容器和 PyTorch 环境兼容。不要让 pip 自动替换 CANN 镜像中已经适配的核心依赖。

## 安装 AISBench

OM 生成后统一使用华为官方 AISBench 测试加载和 NPU 执行性能。在 CANN 容器中执行：

```bash
git clone --depth 1 https://gitee.com/ascend/tools.git /tmp/ascend-tools

python3 -m pip install --no-build-isolation \
  /tmp/ascend-tools/ais-bench_workload/tool/ais_bench/backend

python3 -m pip install --no-build-isolation \
  /tmp/ascend-tools/ais-bench_workload/tool/ais_bench
```

`aclruntime` 需要针对当前 aarch64/CANN 环境编译，安装可能需要几分钟。

## 1. 导出 PointPillars ONNX

准备权重：

```text
pointpillars/weights/pointpillars_kitti_3class.pth
```

执行导出：

```bash
python3 pointpillars/export_onnx.py \
  --checkpoint pointpillars/weights/pointpillars_kitti_3class.pth \
  --output-dir pointpillars/onnx \
  --opset 11
```

生成：

```text
pointpillars/onnx/pillar_encoder.onnx
pointpillars/onnx/backbone_head.onnx
```

该 checkpoint 使用旧版 10 维 PillarFeatureNet 输入。导出脚本要求 PFN 第一层权重形状为 `[64, 10]`。

## 2. ONNX 转 OM

转换 Pillar Encoder：

```bash
python3 common/convert_onnx_to_om.py \
  --model pointpillars/onnx/pillar_encoder.onnx \
  --output pointpillars/om/pillar_encoder \
  --soc-version Ascend310P1 \
  --input-format ND \
  --input-shape 'decorated_features:40000,32,10' \
  --precision-mode allow_fp32_to_fp16 \
  --log-file pointpillars/logs/atc_pillar_encoder.log
```

转换 Backbone 和检测头：

```bash
python3 common/convert_onnx_to_om.py \
  --model pointpillars/onnx/backbone_head.onnx \
  --output pointpillars/om/backbone_head \
  --soc-version Ascend310P1 \
  --input-format NCHW \
  --input-shape 'spatial_features:1,64,496,432' \
  --precision-mode allow_fp32_to_fp16 \
  --log-file pointpillars/logs/atc_backbone_head.log
```

## 3. 使用 AISBench 测试 OM

```bash
python3 -m ais_bench \
  --model pointpillars/om/pillar_encoder.om \
  --device 0 \
  --batchsize 1 \
  --warmup_count 10 \
  --loop 100 \
  --display_all_summary true

python3 -m ais_bench \
  --model pointpillars/om/backbone_head.om \
  --device 0 \
  --batchsize 1 \
  --warmup_count 10 \
  --loop 100 \
  --display_all_summary true
```

不传 `--input` 时，AISBench 会按 OM 的输入形状自动生成零值数据，因此测时不需要 KITTI 数据集。重点关注：

- `NPU_compute_time`：NPU 模型执行耗时。
- `H2D_latency` / `D2H_latency`：主机与设备间的传输耗时。
- `mean`、`min`、`max`、`median`、`P99`：多次执行的耗时分布。
- `throughput`：单位时间的推理量。

AISBench 在这里分别测试两个 OM，不统计 CPU voxelization、Scatter、后处理以及两个 OM 之间的整体链路耗时。

在 Ascend 310P1 上使用 10 次预热、100 次正式循环的一次实测结果：

| OM | NPU 平均耗时 | P99 | Throughput |
| --- | ---: | ---: | ---: |
| `pillar_encoder.om` | 46.934 ms | 47.571 ms | 21.307 infer/s |
| `backbone_head.om` | 7.859 ms | 7.979 ms | 127.243 infer/s |

测试值会受 NPU 负载、温度和容器环境影响，应以当次 AISBench 输出为准。

## 4. 真实 KITTI 点云推理

KITTI Velodyne `.bin` 文件中的每个点应包含四个连续的 `float32`：`x、y、z、反射强度`。

```bash
python3 pointpillars/run.py \
  --points pointpillars/data/000008.bin \
  --model-dir pointpillars/om \
  --output pointpillars/results/000008_detections.json
```

输出 JSON 包含类别、置信度和 LiDAR 坐标系中的 3D 框：

```json
{
  "class_name": "Car",
  "score": 0.952,
  "box_lidar_bottom_center": {
    "x": 14.76,
    "y": -1.10,
    "z": -1.54,
    "width_x": 3.74,
    "length_y": 1.54,
    "height_z": 1.49,
    "yaw": -0.312
  }
}
```

## 公共工具适用范围

- `convert_onnx_to_om.py`：通过命令行参数转换其他 ONNX。
- `acl_runner.py`：提供静态形状 OM 的加载、内存复制、执行和资源释放。

OM 生成后的通用性能测试直接使用 AISBench，不再维护自定义检查和冒烟测试脚本。

增加其他模型时，新建一个模型目录并实现该模型的 `export_onnx.py`、`preprocess.py`、`postprocess.py` 和 `run.py`，继续复用 `common/` 中的工具。

## 当前限制

- `acl_runner.py` 当前面向静态输入形状。
- PointPillars 的 voxelization、Scatter 和后处理在 CPU 上执行。
- 旋转 NMS 使用 OpenCV CPU 实现，与 MMDetection3D CUDA NMS 的边界结果可能略有差异。
- 当前实现面向所验证的 KITTI 三分类 checkpoint，其他 PointPillars 权重需要核对配置、类别顺序、Anchor 和输入特征维度。
