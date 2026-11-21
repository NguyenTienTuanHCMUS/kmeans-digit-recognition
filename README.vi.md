# Nhận diện chữ số viết tay bằng K-means

[English](README.md) | **Tiếng Việt**

Dự án nhỏ ra đời từ một ý tưởng chợt nảy ra khi học **K-means clustering**: K-means là thuật toán **học không giám sát** và không hề biết "chữ số" là gì, vậy có dùng nó để nhận diện chữ số người dùng vẽ được không?

Câu trả lời là **được**, theo hai cách mà dự án lần lượt thử:

| Cách | K-means phân cụm cái gì | Đúng trên tập test MNIST |
|---|---|---:|
| 1. So mẫu nguyên khối | cả ảnh 28×28, nên mỗi centroid là một mẫu chữ số | 92,15% (400 cluster) |
| 2. Học đặc trưng | các mảnh nhỏ 7×7, nên mỗi centroid là một nét cơ bản | **98,83%** (400 nét) |

Cả hai giao diện (desktop và web) hiện dùng **cách 2**.

**Dùng thử ngay trên trình duyệt:** https://nguyentientuanhcmus.github.io/kmeans-digit-recognition/web/?lang=vi (trang có nút chuyển Tiếng Việt / English)

## Mục lục

- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt và chạy](#cài-đặt-và-chạy)
- [Dữ liệu huấn luyện](#dữ-liệu-huấn-luyện)
- [Cách 1: so mẫu nguyên khối](#cách-1-so-mẫu-nguyên-khối)
- [Cách 2: dùng K-means để học đặc trưng](#cách-2-dùng-k-means-để-học-đặc-trưng)
- [Tinh chỉnh cách 2 (trên GPU)](#tinh-chỉnh-cách-2-trên-gpu)
- [So sánh hai cách](#so-sánh-hai-cách)
- [Chuẩn hóa hình vẽ: chi tiết nhỏ nhưng quan trọng](#chuẩn-hóa-hình-vẽ-chi-tiết-nhỏ-nhưng-quan-trọng)
- [Hai giao diện](#hai-giao-diện)
- [Giới hạn](#giới-hạn)
- [Nguồn và giấy phép](#nguồn-và-giấy-phép)

## Cấu trúc dự án

```
├── train_kmeans.py      Cách 1: K-means trên cả ảnh, gán nhãn cluster, lưu ra clusters.npz
├── train_features.py    Cách 2: K-means trên các mảnh nhỏ + hồi quy logistic, lưu ra features_k<K>.npz
├── tune_features.py     Thử nhiều cấu hình của cách 2 trên tập kiểm định
├── digit_app.py         Giao diện Tkinter với bảng vẽ 28×28 (dùng cách 2)
├── export_web.py        Chuyển features_k400.npz sang web/model.js cho bản web
├── clusters.npz         Kết quả huấn luyện của cách 1 (K = 400)
├── features_k400.npz    Kết quả huấn luyện của cách 2 (400 nét 7×7, lưới 3×3), hai giao diện đều dùng
├── requirements.txt
├── README.md            README tiếng Anh
├── README.vi.md         README tiếng Việt (file này)
├── index.html           Chuyển hướng tới web/ (cho GitHub Pages)
├── MNIST/raw/*.gz       Dữ liệu MNIST (bản nén, các chương trình đọc trực tiếp)
└── web/
    ├── index.html       Bản chạy trên trình duyệt
    ├── model.js         Mô hình 400 nét, khoảng 340 KB
    └── sakuga/          Các thành phần giao diện Sakuga VFX (MIT)
```

## Cài đặt và chạy

Cần Python 3 (đã thử với 3.14):

```bash
pip install -r requirements.txt
```

Chạy giao diện chỉ cần NumPy và Pillow. scikit-learn chỉ cần khi huấn luyện cách 2 trên CPU.

**GPU (không bắt buộc, NVIDIA):** cách 2 có thể chạy các bước nặng trên GPU bằng CuPy. Cài CuPy cùng các thư viện CUDA đi kèm (tải về khoảng 1–2 GB):

```bash
pip install "cupy-cuda12x[ctk]"
```

**Huấn luyện** (có thể bỏ qua vì repo đã có sẵn các mô hình):

```bash
python train_kmeans.py --k 400       # cách 1, ghi ra clusters.npz
python train_features.py --k 400 --patch 7 --pool 3 --C 0.1          # cách 2, đúng mô hình đang dùng
python train_features.py --k 400 --patch 7 --pool 3 --C 0.1 --gpu    # như trên, chạy trên GPU NVIDIA
python tune_features.py --gpu                                          # so sánh các cấu hình trên tập kiểm định
```

`train_features.py` ghi ra `features_k400.npz` và một ảnh chụp từ điển nét đã học (`dictionary_k400.png`). Trên máy phát triển (Intel i7-13620H, RTX 4060 Laptop), mô hình đang dùng huấn luyện mất khoảng 30 giây trên GPU. Trên CPU mỗi lần chạy mất vài phút, phần lớn là để mô tả 70.000 ảnh bằng từ điển.

**Giao diện desktop (Tkinter):**

```bash
python digit_app.py                              # dùng features_k400.npz
python digit_app.py --model features_k200.npz    # hoặc một từ điển khác đã huấn luyện
```

**Giao diện web:** dùng bản [trên GitHub Pages](https://nguyentientuanhcmus.github.io/kmeans-digit-recognition/web/?lang=vi), hoặc mở thẳng file `web/index.html` bằng trình duyệt, không cần server hay internet. Mỗi khi huấn luyện lại, chạy thêm lệnh sau để cập nhật mô hình cho bản web:

```bash
python export_web.py                             # hoặc --model features_k200.npz
```

## Dữ liệu huấn luyện

- Cả hai cách đều huấn luyện trên **60.000 ảnh train** của MNIST.
- **10.000 ảnh test** được giữ riêng, chỉ dùng để đo độ chính xác. Mọi con số độ chính xác trong README này đều đo trên tập test, trừ khi ghi khác.
- Cách 2 học từ điển nét từ **100.000 mảnh 7×7** cắt ngẫu nhiên từ ảnh train (bỏ qua các mảnh gần như toàn nền), rồi huấn luyện bộ phân loại trên cả 60.000 ảnh train.
- Khi tinh chỉnh cách 2, 10.000 ảnh train được tách riêng làm **tập kiểm định** để chọn cấu hình, nên tập test không bị dùng vào việc đó (xem [Tinh chỉnh](#tinh-chỉnh-cách-2-trên-gpu)).
- Các hình người dùng vẽ trên giao diện không bao giờ được dùng để huấn luyện.

## Cách 1: so mẫu nguyên khối

### Cách hoạt động

1. **Gom cụm (không dùng nhãn).** Mỗi ảnh 28×28 là một điểm trong không gian 784 chiều. K-means chia 60.000 điểm này thành K cụm. **Centroid** của mỗi cụm là trung bình các ảnh trong cụm, trông như một chữ số hơi mờ.
2. **Đặt nhãn cho cụm.** Mỗi cụm nhận nhãn của chữ số xuất hiện nhiều nhất trong nó (bỏ phiếu theo đa số).
3. **Nhận diện.** Tìm centroid gần nhất (khoảng cách Euclid). Nhãn của nó là đáp án.

K-means vẫn là học không giám sát: nó tự tìm ra các "kiểu viết", còn nhãn chỉ dùng để **đặt tên** cho những gì nó tìm ra.

### Cài đặt K-means

K-means được tự viết bằng NumPy (không dùng scikit-learn) để thấy rõ từng bước:

- **Khởi tạo k-means++.** Mỗi centroid ban đầu tiếp theo được chọn với xác suất tỉ lệ với bình phương khoảng cách tới centroid gần nhất đã có. Nhờ vậy các tâm tản ra xa nhau, hội tụ nhanh và ổn định hơn.
- **Tính khoảng cách nhanh.** Dùng công thức `‖x − c‖² = ‖x‖² − 2·x·c + ‖c‖²`, nên phần `x·c` cho mọi cặp (ảnh, centroid) chỉ là một phép nhân ma trận 60.000 × 784 × K. NumPy giao phép nhân này cho thư viện BLAS đã tối ưu: trên máy phát triển, nó chỉ mất khoảng 30 ms với K = 200.
- **Lặp hai bước.** *Gán:* mỗi ảnh vào cụm có centroid gần nhất. *Cập nhật:* mỗi centroid thành trung bình các ảnh trong cụm.
- **Cụm rỗng** được đặt lại centroid vào ảnh đang nằm xa centroid của mình nhất.
- **Điều kiện dừng:** inertia giảm ít hơn 0,01% sau một vòng, hoặc sau 100 vòng. Với K = 200, thuật toán hội tụ sau 33 vòng, mất khoảng 12 giây.

### Số lượng cluster và độ chính xác

Cùng seed ngẫu nhiên, chỉ thay đổi K:

| K (số cluster) | Đúng trên tập test | |
|---:|---:|:---|
| 10 | 62,81% | `██████` |
| 20 | 71,17% | `███████████` |
| 50 | 81,17% | `████████████████` |
| 100 | 87,37% | `███████████████████` |
| 200 | 90,77% | `████████████████████` |
| 400 | **92,15%** | `█████████████████████` |

- **K càng lớn càng chính xác, nhưng lợi ích giảm dần:** từ 10 lên 50 tăng gần 20 điểm phần trăm, từ 200 lên 400 chỉ tăng khoảng 1,4.
- **Không bị học vẹt rõ rệt:** với K = 200, đúng 90,04% trên tập train và 90,77% trên tập test.
- **Chữ số khó có nhiều cụm hơn.** Với K = 200, số cluster được gán cho từng chữ số là:

  | Chữ số | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Số cluster | 23 | 10 | 30 | 23 | 17 | 21 | 21 | 15 | 20 | 20 |

- **Trường hợp cực đoan:** với K = 60.000 (mỗi ảnh một cụm), K-means chỉ còn là ghi nhớ dữ liệu và trở thành thuật toán 1 láng giềng gần nhất (1-NN). Cách này đạt khoảng 97% trên MNIST theo bảng kết quả trên trang MNIST của Yann LeCun, nhưng mỗi lần đoán phải so với 60.000 ảnh.

### Tại sao không dùng 10 cluster cho 10 chữ số?

Vì K-means chỉ gom các ảnh có điểm ảnh giống nhau, mà hai số 4 viết khác kiểu có thể khác nhau còn nhiều hơn một số 4 và một số 9. Với K = 10:

| Cluster | Nhãn được gán | Thành phần thật bên trong |
|---|---|---|
| 9 | 4 | 2.835 số **4**, 2.602 số **9**, 1.091 số 7 |
| 5 | 9 | 2.703 số **9**, 2.586 số **4**, 491 số 7 |
| 2 | 3 | 3.819 số 3, 1.756 số **5**, 1.010 số 8 |
| 3 | 6 | 1.707 số 6, 1.278 số **5**, 1.217 số 0 |
| 4 | 6 | 3.721 số 6, 140 số 0, 129 số 2 |

Số 4 và số 9 bị trộn gần như đều nhau. Hai cụm cùng mang nhãn 6, còn **số 5 không có cụm nào**, nên chương trình không bao giờ đoán ra số 5. Kết quả chỉ đúng 62,81%. Mỗi chữ số có nhiều kiểu viết, và một centroid phải lấy trung bình mọi kiểu sẽ không giống kiểu nào. Thêm cluster thì mỗi kiểu có centroid riêng, nhưng mỗi kiểu viết vẫn cần một mẫu riêng. Cách 2 vượt qua được đúng giới hạn này.

## Cách 2: dùng K-means để học đặc trưng

Thay vì học cả chữ số, K-means học một **"bảng chữ cái" của nét viết**, và mỗi chữ số được mô tả như một tổ hợp các nét. Ý tưởng này theo bài báo của Coates, Lee và Ng (2011).

### Cách hoạt động

Mô hình đang dùng có mảnh 7×7, 400 nét và lưới gộp 3×3. Mục [Tinh chỉnh](#tinh-chỉnh-cách-2-trên-gpu) giải thích vì sao chọn các thông số này.

1. **Cắt mảnh.** Cắt ngẫu nhiên 100.000 mảnh 7×7 từ ảnh train, bỏ các mảnh gần như trống.
2. **Chuẩn hóa và làm trắng.** Mỗi mảnh được trừ độ sáng trung bình và chia cho độ lệch chuẩn của nó, rồi toàn bộ mảnh được làm trắng (ZCA) để chúng khác nhau ở hình dạng chứ không ở độ đậm nhạt. Thiếu bước này, các centroid chỉ là những đốm mờ.
3. **Chạy K-means trên các mảnh** (cùng cài đặt với cách 1). Lúc này mỗi centroid là một **nét cơ bản**: nét ngang, nét dọc, nét chéo ở nhiều góc, đường cong, đầu nét, góc gãy. K centroid này là **từ điển nét**.
4. **Mô tả ảnh.** Trượt cửa sổ 7×7 qua cả 22 × 22 vị trí của ảnh. Ở mỗi vị trí, đo mức độ mảnh đó giống từng nét bằng cách mã hóa "tam giác": `max(0, khoảng cách trung bình − khoảng cách tới nét k)`.
5. **Gộp theo vùng.** Chia ảnh thành lưới 3×3 và cộng các giá trị đó trong từng vùng trong 9 vùng. Mỗi ảnh thành một vector 9 × K = 3.600 số, kiểu "phía trên có nhiều nét cong, bên phải có nét dọc...". Bước gộp này cũng giúp hình vẽ lệch vài pixel vẫn được nhận đúng.
6. **Phân loại.** Một mô hình hồi quy logistic được huấn luyện trên các vector đó cùng nhãn, và cho ra xác suất của từng chữ số.

```
Cách 1:  ảnh 28×28 ──► so với K mẫu chữ số ──► nhãn của mẫu gần nhất

Cách 2:  ảnh 28×28 ──► các mảnh 7×7 ──► so với K nét ──► gộp theo vùng
                                                           │
                              vector 9K số ──► hồi quy logistic ──► chữ số
```

### Thí nghiệm ban đầu: kích thước từ điển

Phiên bản đầu tiên dùng mảnh 6×6, lưới 2×2 (4 góc phần tư), hồi quy logistic của scikit-learn với C = 1 mặc định, và chỉ thay đổi số nét:

| Số nét (K) | Đúng trên tập train | Đúng trên tập test |
|---:|---:|---:|
| 50 | 98,55% | 97,98% |
| 100 | 99,24% | 98,28% |
| 200 | 99,74% | 98,28% |
| 400 | 99,90% | **98,54%** |

- **Chỉ 50 nét đã vượt xa 400 mẫu chữ số** (97,98% so với 92,15%). Các nét được dùng chung cho mọi chữ số và mọi kiểu viết, nên không cần một mẫu riêng cho từng kiểu.
- **Lợi ích lại giảm dần:** từ 100 lên 400 nét chỉ tăng thêm khoảng 0,3 điểm.
- **Bắt đầu học vẹt nhẹ:** với 400 nét, mô hình gần như thuộc lòng tập train (99,90%) nhưng chỉ đúng 98,54% trên tập test. Cách 1 không có khoảng chênh này vì đơn giản hơn nhiều. Bước tinh chỉnh bên dưới nhắm vào đúng khoảng chênh này.
- Ở đây K-means vẫn không dùng nhãn: nó chỉ quyết định *có những nét nào*. Nhãn được bộ phân loại dùng để học *nét nào, ở đâu, thì thành chữ số nào*. Điều này gần với việc lớp đầu tiên của mạng nơ-ron tích chập (CNN) làm, chỉ khác là bộ lọc được tìm bằng K-means thay vì bằng đạo hàm.

## Tinh chỉnh cách 2 (trên GPU)

### Chọn cấu hình mà không nhìn vào tập test

Nếu thử nhiều cấu hình rồi giữ cái nào cho điểm cao nhất **trên tập test**, ta sẽ vô tình "học thuộc" tập test, và con số cuối cùng không còn cho biết mô hình làm tốt đến đâu với ảnh mới. Vì vậy `tune_features.py` chia 60.000 ảnh train thành 50.000 ảnh để học và 10.000 ảnh làm **tập kiểm định (validation)**. Mọi cấu hình đều được chấm trên tập kiểm định. Tập test chỉ dùng một lần, ở cuối.

Ba thông số được tinh chỉnh:

- **Kích thước mảnh:** 5×5, 6×6, 7×7 (và 8×8).
- **Lưới gộp:** chia ảnh thành 2×2, 3×3 hoặc 4×4 vùng.
- **C**, mức điều chuẩn của hồi quy logistic. Nó cân bằng giữa việc khớp ảnh train và việc giữ các trọng số nhỏ: C nhỏ nghĩa là điều chuẩn mạnh, mô hình đơn giản hơn; C lớn cho phép mô hình bám sát ảnh train và dễ học vẹt.

### Chạy trên GPU

NumPy và scikit-learn chỉ dùng CPU. Khi đã cài [CuPy](https://cupy.dev), tùy chọn `--gpu` chuyển các bước nặng sang GPU NVIDIA:

- **Mô tả ảnh** chạy cùng một đoạn code, chỉ khác là trên mảng CuPy thay vì mảng NumPy. Mô tả 70.000 ảnh mất khoảng **5–7 giây** trên RTX 4060 Laptop, so với vài phút trên CPU. Đặc trưng tính trên GPU khớp với CPU, chỉ lệch do làm tròn số thực (lệch lớn nhất 0,00025 trên giá trị cỡ 149).
- **Hồi quy logistic** của scikit-learn không có bản GPU, nên hàm mất mát và đạo hàm được tính trên GPU bằng CuPy, rồi đưa cho bộ tối ưu L-BFGS của SciPy, cùng thuật toán mà scikit-learn dùng, tối ưu cùng một công thức.
- K-means trên các mảnh chỉ mất vài giây nên vẫn để trên CPU.

Việc này làm lộ ra một điều về thí nghiệm ban đầu: với C = 1, **scikit-learn dừng sau 100 vòng lặp** khi hàm mất mát còn 663, trong khi bộ tối ưu trên GPU đi tiếp tới 488. Mô hình ban đầu thực ra chưa học xong, và việc dừng sớm vô tình hoạt động như một kiểu điều chuẩn thêm. Vì vậy toàn bộ việc tinh chỉnh được làm lại trên GPU, dùng cùng một bộ tối ưu cho mọi cấu hình.

### Kết quả trên tập kiểm định

400 nét, độ chính xác (%) trên 10.000 ảnh kiểm định:

| Mảnh | Lưới | C = 0,003 | 0,01 | 0,03 | 0,1 | 0,3 | 1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 5×5 | 2×2 | 98,22 | 98,53 | 98,61 | 98,69 | 98,76 | 98,72 |
| 5×5 | 3×3 | 98,56 | 98,73 | 98,78 | 98,77 | 98,78 | 98,80 |
| 5×5 | 4×4 | 98,71 | 98,79 | 98,78 | 98,83 | 98,83 | 98,76 |
| 6×6 | 2×2 | 98,10 | 98,33 | 98,58 | 98,63 | 98,70 | 98,71 |
| 6×6 | 3×3 | 98,58 | 98,77 | 98,79 | 98,78 | 98,73 | 98,75 |
| 6×6 | 4×4 | 98,91 | 98,87 | 98,80 | 98,78 | 98,79 | 98,74 |
| 7×7 | 2×2 | 98,21 | 98,49 | 98,69 | 98,69 | 98,72 | 98,72 |
| **7×7** | **3×3** | 98,68 | 98,86 | 98,91 | **98,97** | 98,92 | 98,87 |
| 7×7 | 4×4 | 98,81 | 98,88 | 98,88 | 98,81 | 98,81 | 98,84 |

- **Lưới 2×2 kém nhất ở mọi kích thước mảnh.** Chia ảnh thành 9 hoặc 16 vùng giữ được nhiều thông tin về *vị trí* của nét hơn.
- **Mảnh 7×7, lưới 3×3, C = 0,1 đứng đầu (98,97%).** Các ô xung quanh nó trong bảng cũng cao (98,86–98,92%), nên đây không phải một ô may mắn lẻ loi.
- **Nhiều cấu hình chênh nhau trong mức dao động ngẫu nhiên.** Trên 10.000 ảnh, độ chính xác quanh 99% dao động khoảng ±0,1 điểm do may rủi, nên không nên đọc quá kỹ các chênh lệch 10–20 ảnh.
- Thử thêm hai hướng quanh cấu hình tốt nhất: **800 nét** đạt 99,00% (chỉ hơn 3 ảnh, nhưng mô hình nặng và chậm gấp đôi), **mảnh 8×8** đạt 98,93%. Mô hình 400 nét đơn giản hơn được giữ lại.

### Mô hình cuối cùng

Huấn luyện lại với mảnh 7×7, 400 nét, lưới 3×3, C = 0,1 trên cả 60.000 ảnh train, rồi đo trên tập test đúng một lần:

| | Trước tinh chỉnh | Sau tinh chỉnh |
|---|---:|---:|
| Mảnh, lưới, C | 6×6, 2×2, 1 | 7×7, 3×3, 0,1 |
| Đúng trên tập train | 99,90% | 99,99% |
| Đúng trên tập test | 98,54% (146 ảnh sai) | **98,83%** (117 ảnh sai) |

Điểm trên tập test thấp hơn tập kiểm định một chút (98,97%). Điều này thường xảy ra khi chọn cấu hình tốt nhất trong nhiều cấu hình. Mốc 99% chưa đạt được.

## So sánh hai cách

| | Cách 1: so mẫu | Cách 2: học nét |
|---|---|---|
| Đúng trên tập test | 92,15% | **98,83%** |
| Một centroid trông như | một chữ số hoàn chỉnh (hơi mờ) | một nét 7×7 |
| Nhãn được dùng để | đặt tên cho từng cụm | huấn luyện bộ phân loại trên đặc trưng nét |
| Công việc mỗi lần đoán | so với 400 mẫu | so 484 mảnh với 400 nét, rồi phân loại |
| Dễ giải thích đáp án | có: "giống mẫu này nhất" | khó hơn: "các nét này xuất hiện ở các chỗ này" |

## Chuẩn hóa hình vẽ: chi tiết nhỏ nhưng quan trọng

Ảnh MNIST được tạo theo một quy trình cố định: chữ số được thu vào khung 20×20 (giữ tỉ lệ), rồi đặt vào ảnh 28×28 sao cho **trọng tâm (center of mass)** nằm giữa ảnh. Hình người dùng vẽ có thể nhỏ, to hoặc lệch góc, nên cả hai giao diện đều chuẩn hóa hình vẽ theo đúng quy trình đó trước khi nhận diện:

1. Cắt sát nét vẽ.
2. Thu vào khung 20×20, giữ tỉ lệ.
3. Đặt vào ảnh 28×28 rồi dịch để trọng tâm nằm ở tọa độ **(14, 14)**.

Bước 3 cho một bài học khi làm cách 1. Ban đầu trọng tâm được đặt ở (13,5; 13,5), tức tâm hình học của lưới tính theo chỉ số pixel. Chỉ lệch nửa ô như vậy mà độ chính xác trên 2.000 ảnh test tụt từ 87,65% xuống 82,6%. Đổi sang (14, 14) thì khớp lại hoàn toàn: đưa cả tập test qua bước chuẩn hóa vẫn đạt 90,78%, so với 90,77% trên ảnh gốc (K = 200). Với thuật toán so sánh từng điểm ảnh, lệch nửa pixel cũng đáng kể.

## Hai giao diện

Cả hai giao diện đều dùng mô hình cách 2 đã tinh chỉnh (400 nét 7×7, lưới 3×3).

### Desktop: Tkinter (`digit_app.py`)

- Bảng vẽ 28×28 ô. Chuột trái để vẽ, chuột phải để tẩy. Cọ đậm ở giữa và nhạt dần ra mép, giống nét chữ trong MNIST.
- Nhả chuột là có đáp án kèm độ chắc chắn, ảnh sau chuẩn hóa, nét được dùng nhiều nhất, và thanh xác suất cho từng chữ số. Mỗi hình mất khoảng 2,6 ms.

### Web: `web/index.html`

- HTML, CSS và JavaScript thuần, tính toán ngay trong trình duyệt, chạy được khi không có mạng.
- Hai ngôn ngữ: nút **Tiếng Việt / English** ở đầu trang. Lần đầu vào, trang tự chọn theo ngôn ngữ của trình duyệt và nhớ lựa chọn cho lần sau. Có thể mở thẳng một ngôn ngữ bằng `?lang=vi` hoặc `?lang=en` trên đường link.
- Nhận diện ngay khi đang vẽ (có thể tắt), dùng được với chuột, bút và cảm ứng.
- Hiện xác suất của từng chữ số, 5 nét hình vẽ dùng nhiều nhất (các nét khớp nhất với nhiều vị trí có mực nhất), và toàn bộ từ điển 400 nét với 20 nét được dùng nhiều nhất được tô viền.
- Giao diện dùng [Sakuga VFX](https://github.com/finktheartist/sakuga-vfx): nút, tab, thanh trượt, công tắc và tooltip phản ứng như phim anime (nhún lò xo, khói, vụ nổ, chữ ドン!). Ô hiện đáp án là một thành phần tự viết theo cùng phong cách, nổ hiệu ứng mỗi khi đáp án thay đổi lúc nhả chuột. Hiệu ứng tự tắt nếu máy bật chế độ giảm chuyển động.
- **Đã kiểm tra:** trên 1.000 ảnh test đầu tiên, bản JavaScript cho đáp án trùng bản Python ở cả 1.000 ảnh, xác suất lệch nhau khoảng 0,0000005. Trên máy phát triển, mỗi lần nhận diện mất khoảng 9 ms trong trình duyệt, vì mỗi hình phải so 484 mảnh với 400 nét.

Phím tắt: `B` để vẽ, `E` để tẩy, `C` để xóa bảng.

## Giới hạn

- **Vẫn chưa phải tốt nhất.** Mạng nơ-ron tích chập đạt trên 99% trên MNIST. Dự án này nhằm hiểu K-means, không nhằm đạt độ chính xác cao nhất.
- **Hình vẽ khác với MNIST.** Mô hình học từ chữ viết tay được quét lại, nên hình vẽ quá mảnh, quá đậm, bị xoay hoặc theo kiểu lạ vẫn có thể bị nhận sai.
- **Trên web chậm hơn cách 1** (khoảng 9 ms trên laptop, so với dưới 1 ms mỗi lần đoán). Điện thoại yếu có thể hơi khựng khi bật "Nhận diện ngay khi đang vẽ".
- Bản web mới được thử trên trình duyệt nhân Chromium. Theo tác giả Sakuga VFX, thư viện đó chưa được thử trên Safari và Firefox.

## Nguồn và giấy phép

- **Dữ liệu MNIST** của Yann LeCun, Corinna Cortes và Christopher J.C. Burges, phát hành theo giấy phép [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
- **Cách 2** theo bài báo của Adam Coates, Honglak Lee và Andrew Y. Ng, "An Analysis of Single-Layer Networks in Unsupervised Feature Learning" (AISTATS 2011).
- **Sakuga VFX** của Justin Finkelman, giấy phép MIT. Bản sao giấy phép nằm ở [`web/sakuga/LICENSE`](web/sakuga/LICENSE).
