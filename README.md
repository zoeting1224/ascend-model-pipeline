# Ascend Model Pipeline

面向昇腾 NPU 的模型转换与 ACL Runtime 推理代码。仓库将模型无关的公共工具与模型专用的导出、预处理和后处理分开，当前已完成 PointPillars KITTI 三分类模型的完整链路：

```text
PTH → ONNX → OM → ACL Runtime → KITTI 点云 3D 检测结果
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
│   ├── convert_onnx_to_om.py
│   ├── inspect_om.py
│   └── smoke_test_om.py
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

## 3. 检查 OM 接口

```bash
python3 common/inspect_om.py \
  pointpillars/om/pillar_encoder.om \
  pointpillars/om/backbone_head.om
```

预期接口：

```text
pillar_encoder.om
输入  decorated_features [40000,32,10] float32
输出  pillar_features    [40000,64]    float32

backbone_head.om
输入  spatial_features [1,64,496,432] float32
输出  cls_scores       [1,18,248,216] float32
输出  bbox_preds       [1,42,248,216] float32
输出  dir_cls_preds    [1,12,248,216] float32
```

## 4. OM 冒烟测试

```bash
python3 common/smoke_test_om.py \
  --device 0 \
  pointpillars/om/pillar_encoder.om \
  pointpillars/om/backbone_head.om
```

该步骤使用零输入检查模型加载、显存申请、H2D、模型执行、D2H 和输出有限值，不代表真实检测精度。

## 5. 真实 KITTI 点云推理

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
- `inspect_om.py`：检查任意数量的 OM 输入输出接口。
- `smoke_test_om.py`：测试静态形状 OM 的基础执行链路。
- `acl_runner.py`：提供静态形状 OM 的加载、内存复制、执行和资源释放。

增加其他模型时，新建一个模型目录并实现该模型的 `export_onnx.py`、`preprocess.py`、`postprocess.py` 和 `run.py`，继续复用 `common/` 中的工具。

## 当前限制

- `acl_runner.py` 当前面向静态输入形状。
- PointPillars 的 voxelization、Scatter 和后处理在 CPU 上执行。
- 旋转 NMS 使用 OpenCV CPU 实现，与 MMDetection3D CUDA NMS 的边界结果可能略有差异。
- 当前实现面向所验证的 KITTI 三分类 checkpoint，其他 PointPillars 权重需要核对配置、类别顺序、Anchor 和输入特征维度。
