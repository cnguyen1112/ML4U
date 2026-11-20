# Kế hoạch Học & Thực hành Trước Khi Bắt Đầu Task 7

> Tài liệu tổng hợp các đầu việc lập trình và kiến thức lý thuyết cần nắm vững (dựa trên nội dung review với Mentor) trước khi bước vào **Task 7 (YOLO / Object Detection)**.

---

## Phần 1: Nhiệm vụ Thực hành & Lập trình (Hands-on)

### 1. Tự cài đặt Decision Tree bằng tay (không dùng thư viện)
- [ ] Tự viết thuật toán **Decision Tree (ID3 / CART)** từ đầu bằng Python thuần / NumPy (không gọi `DecisionTreeClassifier` của `sklearn`).
- [ ] Làm rõ logic trả lời 2 câu hỏi cốt lõi trong code:
  - **Làm sao chọn Root Node (nút gốc)?** Tính độ đo hỗn loạn (Entropy / Gini) trước khi chia và **Information Gain** của từng thuộc tính; thuộc tính nào cho Information Gain cao nhất thì chọn làm Root.
  - **Làm sao rẽ nhánh tiếp tục?** Chia tập dữ liệu theo điều kiện tại nút đó rồi đệ quy lặp lại quy trình cho từng tập con cho đến khi đạt điều kiện dừng (nút thuần khiết $H(S)=0$, hết thuộc tính, hoặc chạm `max_depth`).
- [ ] **Xử lý thuộc tính là số thực (Continuous Features):**
  - Giải bài toán ví dụ: *Nếu "giờ học" là số thực (VD: $6.5, 8.0, 19.5, 21.0\dots$) thay vì nhãn rời rạc ("Sáng", "Tối") thì chia nhánh như thế nào?*
  - **Yêu cầu cài đặt:** Sắp xếp các giá trị số thực tăng dần, thử các ngưỡng cắt (midpoint threshold $t$ giữa 2 giá trị liên tiếp để tách thành 2 nhánh $x \le t$ và $x > t$), tính Information Gain tại mọi ngưỡng và chọn ngưỡng $t^*$ tốt nhất.

### 2. Tự xây dựng lại hàm vẽ cây (`plot_tree` như trong `sklearn`)
- [ ] Nghiên cứu cách hàm `sklearn.tree.plot_tree` biểu diễn thông tin ở mỗi nút (`điều kiện rẽ nhánh`, `entropy/gini`, `samples`, `value` từng lớp, `class` dự đoán).
- [ ] Tự viết một hàm hiển thị / vẽ lại cây quyết định từ cấu trúc cây tự build ở Mục 1 (dạng text tree có căn lề nhánh `|--`, hoặc dùng `matplotlib` / `graphviz` vẽ sơ đồ khối từng node và mũi tên nhánh True/False).

### 3. Thực hành chia Train / Validation (Eval) / Test & Đọc biểu đồ Loss
- [ ] Phân biệt rạch ròi vai trò của 3 tập dữ liệu:
  - **Train (*đi học, làm đề trên trường*):** Tập dùng để mô hình học và cập nhật trọng số.
  - **Validation / Eval (*thi cuối kỳ*):** Tập đánh giá trong quá trình huấn luyện (không cập nhật trọng số) để theo dõi `val_loss`, tinh chỉnh siêu tham số và chọn checkpoint tốt nhất.
  - **Test (*thi tốt nghiệp*):** Tập giữ kín hoàn toàn đến phút cuối để đánh giá năng lực tổng quát hóa thực tế của mô hình.
- [ ] Vẽ biểu đồ **Train Loss vs Eval (Val) Loss** theo từng epoch và nhận diện 2 trạng thái:
  - **Overfitting (*học trên lớp rất tốt nhưng đi thi làm sai*):** Train Loss giảm sâu / Train Acc rất cao, nhưng Val/Test Loss bắt đầu tăng lại hoặc Test Acc kém.
  - **Underfitting (*học trên lớp còn chưa đạt nên đi thi cũng rớt*):** Cả Train Loss và Val/Test Loss đều cao, mô hình chưa học được quy luật dữ liệu.

### 4. Khảo sát hiện tượng Nhãn Train bị sai (Label Noise)
- [ ] Làm thí nghiệm nhỏ (ví dụ đảo sai nhãn $10\%\text{–}30\%$ mẫu trong tập train) để quan sát:
  - **Mô hình sẽ dự đoán ra gì?** (Dễ bị kéo lệch ranh giới quyết định hoặc học thuộc luôn các điểm sai $\rightarrow$ Overfitting vào nhiễu, giảm Test Accuracy).
- [ ] Tổng hợp **cách phát hiện và xử lý / "sửa" khi gặp nhãn train sai:**
  - *Trước/trong khi xử lý dữ liệu:* Lọc mẫu có loss cao bất thường khi dùng K-Fold Cross-Validation để kiểm tra lại nhãn thủ công (Data Cleaning).
  - *Khi huấn luyện:* Dùng **Regularization** (`Dropout`, `Weight Decay`), giới hạn độ sâu cây (`max_depth`, `min_samples_leaf`), **Early Stopping** (dừng trước khi mô hình học vẹt nhãn sai), hoặc **Label Smoothing** để mô hình không quá tự tin vào nhãn nhiễu.

---

## Phần 2: Nhiệm vụ Nghiên cứu Lý thuyết (Core ML / Deep Learning)

### 5. Activation Function & Vì sao phải dùng ReLU?
- [ ] Chứng minh được bằng toán học: Nếu chỉ xếp chồng các lớp tuyến tính (`Linear` $\rightarrow$ `Linear`) **mà không có hàm kích hoạt**, thì:
  $$W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W' x + b'$$
  $\rightarrow$ Kết quả vẫn chỉ là **một phép biến đổi tuyến tính duy nhất**, số **bậc tự do** (năng lực biểu diễn hình học) không tăng lên, nên mạng nhiều tầng không hơn gì mạng 1 tầng và mãi mãi không học được các biên quyết định phi tuyến.
- [ ] Hiểu vai trò của Activation Function: Đưa bài toán từ **tuyến tính về phi tuyến**, giúp mạng xấp xỉ được các hàm phức tạp.
- [ ] Hiểu vì sao **ReLU** được ưa chuộng: tính toán siêu nhanh ($\max(0, x)$), không bị bão hòa đạo hàm ở vùng dương ($x > 0 \implies \text{grad} = 1$, giảm Vanishing Gradient), và tạo tính thưa (sparsity) cho mạng.

### 6. Phân biệt Softmax, Tanh, Sigmoid và ReLU
- [ ] Lập bảng so sánh đầy đủ 4 hàm kích hoạt trên các tiêu chí:
  - **Công thức & Miền giá trị đầu ra** (VD: Sigmoid $(0, 1)$, Tanh $(-1, 1)$, ReLU $[0, +\infty)$, Softmax cho vector xác suất tổng bằng $1$).
  - **Đạo hàm & Nhược điểm** (Vanishing Gradient khi $|x|$ lớn ở Sigmoid/Tanh; không đối xứng quanh $0$ ở Sigmoid; Dead ReLU ở ReLU).
  - **Vị trí áp dụng chuẩn:** Tầng ẩn (ReLU/Tanh) vs Tầng đầu ra (Sigmoid cho nhị phân/đa nhãn, Softmax cho phân loại đa lớp).

### 7. Dead ReLU (Dying ReLU) & Cách né tránh
- [ ] Hiểu cơ chế chết của neuron: Khi đầu vào $z \le 0$ cho mọi mẫu dữ liệu, $\text{ReLU}(z) = 0$ và đạo hàm bằng $0$ $\rightarrow$ gradient truyền về bằng $0$, trọng số không bao giờ cập nhật nữa.
- [ ] Nắm vững các cách né Dead ReLU:
  - Dùng các hàm kích hoạt cải tiến của ReLU: **Leaky ReLU**, **PReLU**, **ELU**, **GELU**, **SiLU/Swish** (cho độ dốc nhỏ $\alpha > 0$ ở phần âm để gradient vẫn đi qua được).
  - Kiểm soát `learning_rate` không quá lớn ngay từ đầu.
  - Khởi tạo trọng số hợp lý (**He / Kaiming Initialization**) và dùng **Batch Normalization**.

### 8. Learning Rate ($\eta$), Tác dụng phụ & Learning Rate Động
- [ ] Giải thích được ý nghĩa của **Learning Rate (tốc độ học)** và tác dụng phụ khi chọn sai:
  - **Quá lớn:** Bước nhảy vượt qua đáy cực tiểu $\rightarrow$ Loss dao động mạnh, không hội tụ hoặc phân kỳ thành `NaN`.
  - **Quá nhỏ:** Hội tụ rất chậm, tốn nhiều epoch và dễ mắc kẹt ở cực tiểu địa phương / vùng phẳng (plateau).
- [ ] Tìm hiểu **Learning Rate động (Learning Rate Scheduler):**
  - Tại sao cần thay đổi `lr` theo thời gian (đầu lớn để đi nhanh, cuối nhỏ để hội tụ mịn vào đáy loss)?
  - Phân biệt nhanh các chiến lược phổ biến: `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, **Warmup**, và cơ chế tự thích nghi của **Adam**.

### 9. Epochs, Batch Size & Iteration
- [ ] Phân biệt chính xác 3 khái niệm:
  - **Batch Size:** Số mẫu đưa vào mạng trong $1$ lần cập nhật trọng số.
  - **Iteration:** $1$ bước forward + backward + update trên $1$ mini-batch.
  - **Epoch:** $1$ lần mô hình duyệt qua **toàn bộ** tập dữ liệu train.
- [ ] Hiểu mối liên hệ số học (VD: $120$ mẫu, `batch_size=16` $\implies 8$ iterations/epoch) và ảnh hưởng khi đặt quá ít epoch (Underfitting) hoặc quá nhiều epoch (Overfitting).

### 10. Lớp Tích chập (Convolutional Layer)
- [ ] Hiểu bản chất của phép tích chập (`Conv1d`, `Conv2d`): Bộ lọc (Kernel/Filter) trượt qua đầu vào để trích xuất đặc trưng cục bộ (cạnh, góc, họa tiết, chuỗi quy luật).
- [ ] Nắm rõ ý nghĩa các tham số: `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` và công thức tính kích thước đầu ra.
- [ ] Trả lời được: *Tại sao dùng `Conv` thay vì đưa thẳng toàn bộ ảnh vào `Linear` (Fully Connected)?* (Nhờ **Local Receptive Field** và **Weight Sharing** giúp giảm mạnh số tham số, chống overfit và giữ cấu trúc không gian).

### 11. FLOPs (Đo chi phí tính toán của mô hình)
- [ ] Phân biệt **FLOPs** (*Floating Point Operations* — tổng số phép tính cộng/nhân cần thực hiện cho 1 lần forward, đo độ nặng tính toán) với **Params** (số lượng tham số, đo dung lượng bộ nhớ).
- [ ] Biết cách ước lượng FLOPs cho tầng `Linear` ($\approx 2 \times d_{\text{in}} \times d_{\text{out}}$) và tầng `Conv` để so sánh xem hai phương pháp / hai kiến trúc mạng phương pháp nào tốn tính toán nhiều hơn.

---

## Phần 3: Chuẩn bị Trực tiếp cho Task 7 (Tìm hiểu YOLO)

### 12. YOLO là gì & Cơ chế hoạt động cốt lõi
- [ ] Phân biệt bài toán **Image Classification** (ảnh là gì) vs **Object Detection** (ảnh có *những vật gì* và *nằm ở tọa độ Bounding Box nào*).
- [ ] Hiểu ý nghĩa tên gọi **YOLO (You Only Look Once):** Kiến trúc **One-stage detector** nhìn ảnh đúng $1$ lần qua mạng để dự đoán đồng thời tọa độ hộp bao và xác suất lớp (tốc độ real-time), khác với nhóm **Two-stage** (như R-CNN / Faster R-CNN phải đề xuất vùng trước rồi mới phân loại).
- [ ] Nắm vững các khái niệm nền tảng của YOLO:
  - Chia lưới **Grid cell** $(S \times S)$ & dự đoán $(x, y, w, h, \text{confidence}, \text{class probs})$.
  - **IoU (Intersection over Union):** Thước đo độ chồng lấn giữa hộp dự đoán và Ground Truth.
  - **NMS (Non-Maximum Suppression):** Thuật toán loại bỏ các hộp dự đoán trùng lặp cho cùng 1 vật thể.
  - **mAP (mean Average Precision):** Thước đo đánh giá chuẩn cho Object Detection.

### 13. Lịch sử & So sánh các phiên bản YOLO
- [ ] Tổng hợp bảng tiến hóa các phiên bản YOLO (điểm mới cốt lõi, tốc độ vs độ chính xác, FLOPs/Params của từng đời):
  - **Giai đoạn nền tảng (Darknet):** **YOLOv1** (khai sinh one-stage) $\rightarrow$ **YOLOv2 / YOLO9000** (thêm Anchor Box, BatchNorm) $\rightarrow$ **YOLOv3** (dự đoán đa tỷ lệ 3 scale kiểu FPN, Darknet-53).
  - **Giai đoạn tối ưu hiệu năng:** **YOLOv4**, **YOLOv5** (chuyển sang PyTorch, dễ train/deploy), **YOLOv6**, **YOLOv7** (E-ELAN).
  - **Giai đoạn hiện đại (Anchor-free & End-to-End):** **YOLOv8** (Ultralytics, Anchor-free, đa nhiệm Detection/Segment/Pose), **YOLOv9** (PGI/GELAN), **YOLOv10** (loại bỏ NMS để suy luận end-to-end nhanh hơn), **YOLO11** (tối ưu kiến trúc nhẹ, giảm tham số/FLOPs mà vẫn tăng mAP), và cập nhật thêm các bản mới hơn nếu có.
