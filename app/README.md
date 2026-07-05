# 多模态植物计数 —— Web 交互界面

## 快速启动

```bash
# 1. 进入 app 目录
cd app

# 2. 安装依赖
pip install flask pillow torch torchvision matplotlib numpy

# 3. 启动服务
python app.py

# 4. 在浏览器打开 http://127.0.0.1:5000
```

## 支持的模型

| 模型 | 状态 | 说明 |
|------|------|------|
| CLIP-Count | 已实现 | 需要下载预训练权重到 `weights/clip_count_pretrained.ckpt` |
| CountingDINO | 待实现 | 需要 DINOv2 环境 |
| CountGD | 待实现 | 需要 GroundingDINO + SAM 环境 |
| PACNet | 已实现 | 需要训练后权重到 `results/pacnet/best_model.pth` |

## API 接口

### POST /api/predict
上传图像并返回计数结果。

参数：
- `image`: 图像文件
- `model`: 模型名称（CLIP-Count / CountingDINO / CountGD / PACNet）
- `prompt`: 文本提示（leaf / flower / fruit 等）

### GET /api/status
返回各模型的加载状态。
