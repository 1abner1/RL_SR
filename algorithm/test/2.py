import torch
import torchvision.transforms as T
from PIL import Image

# 加载DETR模型
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

# 加载图像
img = Image.open('path/to/image.jpg')

# 对图像进行预处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img = transform(img)

# 将图像输入DETR模型
model.eval()
outputs = model(img.unsqueeze(0))

# 解析输出
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.7
bboxes_scaled = outputs['pred_boxes'][0, keep].cpu()

print(bboxes_scaled)
