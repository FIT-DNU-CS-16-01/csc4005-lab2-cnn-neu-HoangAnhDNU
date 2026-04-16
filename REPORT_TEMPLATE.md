# CSC4005 – Lab 2 Report

## 1. Thông tin chung
- **Họ và tên:** Nguyễn Hoàng Anh
- **Lớp:** KHMT 17-01
- **Repo:** csc4005-lab2-cnn-neu-HoangAnhDNU
- **W&B project:** [csc4005-lab2-neu-cnn](https://wandb.ai/hoanganhdnu-dainam-vietnam/csc4005-lab2-neu-cnn)

## 2. Bài toán
Phân loại ảnh lỗi bề mặt thép trên bộ dữ liệu **NEU Surface Defect Database (NEU-CLS)** gồm 6 lớp: Crazing, Inclusion, Patches, Pitted_Surface, Rolled-in_Scale, Scratches. Ảnh đầu vào là grayscale, được resize về 64×64 (CNN scratch) hoặc 128×128 (transfer learning). Mục tiêu là so sánh CNN from scratch, transfer learning và fine-tuning với MLP baseline từ Lab 1.

## 3. Mô hình và cấu hình

### 3.1. MLP baseline từ Lab 1
- **Kiến trúc:** MLP 3 hidden layers (512 → 256 → 64), Dropout 0.3
- **Input:** ảnh grayscale 64×64 flatten thành vector 4096 chiều
- **Optimizer:** SGD, lr=0.01
- **Best run:** `run_b_sgd` – 20 epochs
- **Trainable params:** ~2,245,830
- **Avg epoch time:** ~1.85 sec

### 3.2. CNN from scratch
- **Kiến trúc:** SmallCNN – 3 ConvBlock (Conv2d → BatchNorm → ReLU → MaxPool) với 16→32→64 filters, kernel=3, padding=1; sau đó AdaptiveAvgPool2d(1) → Flatten → FC(64→128) → ReLU → Dropout(0.3) → FC(128→6)
- **Input:** ảnh grayscale 1×64×64
- **Optimizer:** AdamW, lr=0.001, weight_decay=1e-4
- **Epochs:** 20, patience=5 (early stopping)
- **Augmentation:** RandomHorizontalFlip, RandomRotation(15)
- **Trainable params:** 32,614
- **W&B run:** [cnn_small_baseline](https://wandb.ai/hoanganhdnu-dainam-vietnam/csc4005-lab2-neu-cnn/runs/ux7wb0ep)

### 3.3. Transfer learning (freeze backbone)
- **Kiến trúc:** ResNet18 pretrained trên ImageNet, freeze toàn bộ backbone, chỉ train classifier head (FC 512→6)
- **Input:** ảnh grayscale chuyển thành 3 channels, 128×128, chuẩn hóa theo ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Optimizer:** AdamW, lr=0.001, weight_decay=1e-4
- **Epochs:** 10, patience=3
- **Augmentation:** RandomHorizontalFlip, RandomRotation(15)
- **Trainable params:** 3,078 (trong tổng 11,179,590)
- **W&B run:** [resnet18_transfer](https://wandb.ai/hoanganhdnu-dainam-vietnam/csc4005-lab2-neu-cnn/runs/ssgt5u52)

### 3.4. Fine-tune (unfreeze toàn bộ)
- **Kiến trúc:** ResNet18 pretrained, unfreeze toàn bộ layers, train end-to-end
- **Input:** giống transfer learning
- **Optimizer:** AdamW, lr=0.0001, weight_decay=1e-4
- **Epochs:** 10, patience=3 (early stop tại epoch 8)
- **Augmentation:** RandomHorizontalFlip, RandomRotation(15)
- **Trainable params:** 11,179,590
- **W&B run:** [resnet18_finetune](https://wandb.ai/hoanganhdnu-dainam-vietnam/csc4005-lab2-neu-cnn/runs/j50i28lw)

## 4. Bảng kết quả

| Model | Train mode | Best Val Acc | Test Acc | Epoch time (s) | Trainable Params | Nhận xét |
|---|---|---:|---:|---:|---:|---|
| MLP (Lab 1) | scratch | 0.4630 | 0.4556 | ~1.85 | ~2,245,830 | Baseline – flatten mất spatial info |
| CNN-small | scratch | 0.9444 | 0.9481 | 3.82 | 32,614 | Cải thiện vượt trội so với MLP, ít params hơn 70× |
| ResNet18 | transfer (freeze) | 0.9667 | 0.9630 | 13.41 | 3,078 | Pretrained features tốt, chỉ cần train head |
| ResNet18 | finetune (unfreeze) | **1.0000** | **1.0000** | 28.44 | 11,179,590 | Kết quả hoàn hảo, fine-tune tận dụng tối đa pretrained weights |

### Nhận xét tổng quan
- CNN scratch đã đạt **94.81% test accuracy** so với chỉ **45.56%** của MLP — cải thiện **+49.25 điểm phần trăm**.
- Transfer learning (freeze) đạt **96.30%** với chỉ **3,078 trainable params** — hiệu quả cực cao.
- Fine-tune đạt **100% test accuracy** — mô hình phân loại hoàn hảo trên tập test 270 mẫu.

## 5. Phân tích learning curves

### CNN from scratch
- Training hội tụ nhanh trong 5 epoch đầu, train_acc đạt ~98% sau epoch 10.
- Val accuracy ổn định quanh mức 0.93–0.94 từ epoch 5 trở đi.
- Khoảng cách train_loss vs val_loss nhỏ, cho thấy mô hình **không bị overfitting nghiêm trọng** dù dataset nhỏ.
- Dropout 0.3 và augmentation giúp regularize hiệu quả.

### ResNet18 transfer (freeze)
- Hội tụ chậm hơn so với CNN scratch vì chỉ train head với 3,078 params.
- Val accuracy dao động nhẹ quanh 0.95–0.97, cho thấy frozen features của ImageNet backbone phù hợp tốt với bài toán lỗi bề mặt nhưng chưa tối ưu hoàn toàn.
- Early stopping không kích hoạt — mô hình vẫn cải thiện dần qua 10 epochs.

### ResNet18 fine-tune
- Hội tụ rất nhanh: val_acc đạt 1.0000 từ epoch 2.
- Train_loss giảm mạnh từ 0.42 (epoch 1) xuống 0.03 (epoch 7).
- Early stopping kích hoạt tại epoch 8 do val_loss không cải thiện thêm sau epoch 5 (best val_loss = 0.0018).
- LR scheduler giảm lr từ 1e-4 xuống 5e-5 ở epoch 8.

## 6. Confusion matrix và lỗi dự đoán sai

### CNN from scratch (test acc = 94.81%)
Confusion matrix cho thấy:
- **Crazing, Patches**: phân loại hoàn hảo (45/45).
- **Inclusion**: 40/45 đúng, nhầm 5 mẫu (3 sang Pitted_Surface, 2 sang Scratches). Inclusion và Scratches có texture tương tự ở cấp độ cục bộ.
- **Pitted_Surface**: 40/45 đúng, nhầm 5 mẫu sang Inclusion. Hai lớp này có đặc trưng giao thoa (đốm lõm vs tạp chất).
- **Rolled-in_Scale**: phân loại hoàn hảo (45/45).
- **Scratches**: 41/45 đúng, nhầm 4 mẫu (2 sang Inclusion, 1 sang Pitted_Surface, 1 sang Rolled-in_Scale).

**Phân tích mẫu dự đoán sai tiêu biểu:**
1. **Inclusion → Pitted_Surface**: Cả hai lớp đều có các đốm nhỏ trên bề mặt, trình CNN nhỏ khó phân biệt khi đốm tạp chất nhỏ giống hố lõm.
2. **Scratches → Inclusion**: Vết xước mảnh có thể giống dải tạp chất hẹp.
3. **Pitted_Surface → Inclusion**: Hố lõm nhỏ và tạp chất đều thể hiện dạng blob tối trên nền sáng.

### ResNet18 transfer (test acc = 96.30%)
- Cải thiện so với CNN scratch, đặc biệt ở Pitted_Surface (45/45) và Rolled-in_Scale (45/45).
- **Inclusion** vẫn là lớp khó nhất: 38/45 đúng, nhầm 7 mẫu (4 sang Pitted_Surface, 2 sang Crazing, 1 sang Scratches).
- Crazing nhầm 2 mẫu sang Patches (cả hai có vân nứt nhỏ).

### ResNet18 fine-tune (test acc = 100%)
- Confusion matrix là **ma trận đơn vị** hoàn hảo: 45/45 cho mỗi lớp.
- Fine-tune cho phép backbone học các đặc trưng chuyên biệt cho lỗi bề mặt thép, vượt qua giới hạn của frozen ImageNet features.

## 7. Kết luận

### CNN có cải thiện so với MLP không?
**Có, cải thiện rất lớn.** CNN scratch (32K params) đạt 94.81% test accuracy so với 45.56% của MLP (2.2M params). CNN tận dụng được cấu trúc không gian 2D của ảnh qua convolution và pooling, trong khi MLP phải flatten ảnh thành vector 1 chiều làm mất hoàn toàn thông tin vị trí tương đối giữa các pixel. Đáng chú ý, CNN đạt kết quả tốt hơn với **ít hơn 70 lần** số tham số — cho thấy weight sharing trong convolution cực kỳ hiệu quả.

### Transfer learning có tốt hơn không?
**Có.** Transfer learning (freeze backbone) đạt 96.30% so với 94.81% từ scratch, dù chỉ train 3,078 params. Fine-tune đạt 100% — hoàn hảo. Pretrained features từ ImageNet cung cấp biểu diễn cạnh, texture, và hình dạng đã được học sẵn từ 1.2 triệu ảnh, rất phú hợp cho bài toán phân loại lỗi bề mặt.

### Khi nào nên chọn transfer learning thay vì train from scratch?
- **Transfer learning phù hợp khi:**
  - Dataset nhỏ (NEU-CLS chỉ có ~1,800 ảnh) — pretrained backbone cung cấp prior knowledge tốt.
  - Cần kết quả nhanh với ít data — chỉ cần train classifier head.
  - Bài toán liên quan đến visual features tổng quát (cạnh, texture, shape).

- **Train from scratch phù hợp khi:**
  - Dữ liệu rất khác biệt so với ImageNet (ví dụ: ảnh y tế chuyên biệt, ảnh satellite đa kênh).
  - Cần mô hình nhẹ, triển khai trên thiết bị edge (CNN-small chỉ 32K params, chạy 3.82s/epoch so với 28.44s/epoch).
  - Không cần accuracy tối đa mà ưu tiên tốc độ inference.

### Tóm tắt
| Tiêu chí | MLP | CNN scratch | ResNet18 transfer | ResNet18 finetune |
|---|---|---|---|---|
| Test Accuracy | 45.56% | 94.81% | 96.30% | **100%** |
| Trainable Params | ~2.2M | 32.6K | 3.1K | 11.2M |
| Epoch time | ~1.85s | 3.82s | 13.41s | 28.44s |
| Phù hợp khi | — | Lightweight, edge deploy | Quick prototyping, ít data | Cần accuracy cao nhất |
