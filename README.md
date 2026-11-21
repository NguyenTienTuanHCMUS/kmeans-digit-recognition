# Nhận diện chữ số viết tay bằng K-means

Dự án nhỏ ra đời từ một ý tưởng chợt nảy ra khi học **K-means clustering**: nếu cho K-means gom 60.000 ảnh chữ số MNIST thành các cụm, rồi đặt tên cho từng cụm, thì có dùng các cụm đó để đoán chữ số người dùng vẽ được không?

Câu trả lời là **được**, và khá tốt: với 400 cluster, chương trình đoán đúng **92,15%** ảnh trong tập test MNIST, dù K-means là thuật toán **học không giám sát** và không hề biết "chữ số" là gì.

**Dùng thử ngay trên trình duyệt:** https://nguyentientuanhcmus.github.io/kmeans-digit-recognition/

## Mục lục

- [Ý tưởng](#ý-tưởng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt và chạy](#cài-đặt-và-chạy)
- [Thuật toán chi tiết](#thuật-toán-chi-tiết)
- [Số lượng cluster và độ chính xác](#số-lượng-cluster-và-độ-chính-xác)
- [Tại sao không dùng 10 cluster cho 10 chữ số?](#tại-sao-không-dùng-10-cluster-cho-10-chữ-số)
- [Chuẩn hóa hình vẽ: chi tiết nhỏ nhưng quan trọng](#chuẩn-hóa-hình-vẽ-chi-tiết-nhỏ-nhưng-quan-trọng)
- [Hai giao diện](#hai-giao-diện)
- [Giới hạn](#giới-hạn)
- [Nguồn và giấy phép](#nguồn-và-giấy-phép)

## Ý tưởng

Toàn bộ quy trình có 3 bước:

1. **Gom cụm (không dùng nhãn).** Mỗi ảnh 28×28 được xem là một điểm trong không gian 784 chiều. K-means chia 60.000 điểm này thành K cụm, mỗi cụm có một **tâm cụm (centroid)**: trung bình của mọi ảnh trong cụm, trông như một chữ số hơi mờ.
2. **Đặt nhãn cho cụm.** Lúc này mới dùng đến nhãn của tập train: mỗi cụm nhận nhãn của chữ số xuất hiện nhiều nhất trong nó (bỏ phiếu theo đa số).
3. **Nhận diện.** Với một hình vẽ mới, tìm centroid gần nhất (khoảng cách Euclid). Nhãn của cụm đó chính là đáp án.

Như vậy K-means vẫn là học không giám sát: nó tự tìm ra các "kiểu viết". Nhãn chỉ được dùng để **đặt tên** cho những gì K-means đã tìm ra.

## Cấu trúc dự án

```
├── train_kmeans.py      Chương trình 1: chạy K-means, gán nhãn cluster, lưu ra clusters.npz
├── digit_app.py         Chương trình 2: giao diện Tkinter, vẽ chữ số trên bảng 28×28
├── export_web.py        Chuyển clusters.npz sang web/clusters.js cho bản web
├── clusters.npz         Kết quả đã huấn luyện sẵn (K = 400)
├── requirements.txt
├── index.html           Chuyển hướng tới web/ (cho GitHub Pages)
├── MNIST/raw/*.gz       Dữ liệu MNIST (bản nén, chương trình đọc trực tiếp)
├── web/
│   ├── index.html       Bản chạy trên trình duyệt
│   ├── clusters.js      400 centroid đã lượng tử hóa, khoảng 414 KB
│   └── sakuga/          Các thành phần giao diện Sakuga VFX (MIT)
```

## Cài đặt và chạy

Cần Python 3 (đã thử với 3.14) và hai thư viện:

```bash
pip install -r requirements.txt
```

**Huấn luyện** (có thể bỏ qua vì repo đã có sẵn `clusters.npz` với K = 400):

```bash
python train_kmeans.py            # mặc định K = 200
python train_kmeans.py --k 400    # nhiều cluster hơn, chính xác hơn
```

Với K = 200, việc huấn luyện mất khoảng 12 giây trên CPU, hội tụ sau 33 vòng lặp.

**Giao diện desktop (Tkinter):**

```bash
python digit_app.py
```

**Giao diện web:** dùng bản [trên GitHub Pages](https://nguyentientuanhcmus.github.io/kmeans-digit-recognition/), hoặc mở thẳng file `web/index.html` bằng trình duyệt, không cần server hay internet. Mỗi khi huấn luyện lại, chạy thêm lệnh sau để cập nhật dữ liệu cho bản web:

```bash
python export_web.py
```

## Thuật toán chi tiết

K-means được tự cài đặt bằng NumPy (không dùng scikit-learn) để thấy rõ từng bước:

- **Khởi tạo k-means++.** Centroid đầu tiên được chọn ngẫu nhiên. Mỗi centroid tiếp theo được chọn với xác suất tỉ lệ với bình phương khoảng cách tới centroid gần nhất đã có. Cách này giúp các tâm ban đầu tản ra xa nhau, hội tụ nhanh và ổn định hơn so với chọn ngẫu nhiên hoàn toàn.
- **Tính khoảng cách nhanh.** Dùng công thức `‖x − c‖² = ‖x‖² − 2·x·c + ‖c‖²`. Phần `x·c` được tính một lần cho mọi cặp (ảnh, centroid) bằng một phép nhân ma trận 60.000 × 784 × K.
- **Lặp hai bước.** *Bước gán:* mỗi ảnh vào cụm có centroid gần nhất. *Bước cập nhật:* mỗi centroid = trung bình các ảnh trong cụm của nó.
- **Cụm rỗng.** Nếu một cụm không còn ảnh nào, centroid của nó được đặt lại vào ảnh đang nằm xa centroid của mình nhất.
- **Điều kiện dừng.** Dừng khi inertia (tổng bình phương khoảng cách) giảm ít hơn 0,01% sau một vòng, hoặc sau 100 vòng.

## Số lượng cluster và độ chính xác

Kết quả trên **10.000 ảnh test** của MNIST, cùng seed ngẫu nhiên, chỉ thay đổi K:

| K (số cluster) | Đúng trên tập test | |
|---:|---:|:---|
| 10 | 62,81% | `██████` |
| 20 | 71,17% | `███████████` |
| 50 | 81,17% | `████████████████` |
| 100 | 87,37% | `███████████████████` |
| 200 | 90,77% | `████████████████████` |
| 400 | **92,15%** | `█████████████████████` |

Vài điều rút ra:

- **K càng lớn càng chính xác, nhưng lợi ích giảm dần.** Từ 10 lên 50 cluster tăng gần 20 điểm phần trăm, còn từ 200 lên 400 chỉ tăng thêm khoảng 1,4 điểm.
- **Không bị học vẹt rõ rệt.** Với K = 200, độ chính xác trên tập train là 90,04%, trên tập test là 90,77%, gần như bằng nhau.
- **Độ thuần của cụm.** Với K = 200, trung bình 89,9% ảnh trong một cụm là cùng một chữ số.
- **Chữ số khó có nhiều cụm hơn.** Với K = 200, số cluster được gán cho từng chữ số là:

  | Chữ số | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Số cluster | 23 | 10 | 30 | 23 | 17 | 21 | 21 | 15 | 20 | 20 |

  Số 2 có nhiều kiểu viết nhất (có vòng hoặc không có vòng ở dưới, nét cong hoặc gãy) nên chiếm 30 cluster. Số 1 đơn giản nên chỉ cần 10.

- **Trường hợp cực đoan.** Nếu K = 60.000 (mỗi ảnh là một cụm riêng), K-means không còn "học" gì nữa mà thành **ghi nhớ toàn bộ dữ liệu**. Khi đó nó chính là thuật toán 1 láng giềng gần nhất (1-NN), đạt khoảng 97% trên MNIST theo bảng kết quả trên trang MNIST của Yann LeCun. Nhưng mỗi lần đoán phải so với 60.000 ảnh thay vì vài trăm centroid. K = 200 đến 400 là mức cân bằng tốt giữa độ chính xác, tốc độ và kích thước file.

## Tại sao không dùng 10 cluster cho 10 chữ số?

Vì **K-means không biết gì về "chữ số"**. Nó chỉ gom các ảnh có điểm ảnh giống nhau. Hai số 4 viết khác kiểu có thể khác nhau về điểm ảnh còn nhiều hơn một số 4 và một số 9.

Chạy thử với K = 10, đây là thành phần thật bên trong một số cụm:

| Cluster | Nhãn được gán | Thành phần thật bên trong |
|---|---|---|
| 9 | 4 | 2.835 số **4**, 2.602 số **9**, 1.091 số 7 |
| 5 | 9 | 2.703 số **9**, 2.586 số **4**, 491 số 7 |
| 2 | 3 | 3.819 số 3, 1.756 số **5**, 1.010 số 8 |
| 3 | 6 | 1.707 số 6, 1.278 số **5**, 1.217 số 0 |
| 4 | 6 | 3.721 số 6, 140 số 0, 129 số 2 |

- **Số 4 và số 9 bị trộn gần như đều nhau**, vì về điểm ảnh chúng rất giống nhau.
- **Hai cụm cùng được gán nhãn 6**, trong khi **số 5 không có cụm nào**. Nghĩa là chương trình không bao giờ đoán ra số 5.
- Kết quả: chỉ đúng **62,81%**.

Lý do sâu hơn là **mỗi chữ số có nhiều kiểu viết**: số 1 thẳng hoặc nghiêng, số 7 có hoặc không có gạch ngang, số 4 nét trên mở hoặc kín. Centroid là **trung bình** của cụm. Nếu bắt một centroid đại diện cho mọi kiểu số 4, nó sẽ thành một hình mờ nhòe không giống kiểu nào. Khi có nhiều cluster, mỗi kiểu viết có centroid riêng. Bạn có thể thấy điều này trong phần "Các centroid mà K-means tìm ra" của bản web: mỗi hàng là một chữ số, mỗi ô là một kiểu viết.

## Chuẩn hóa hình vẽ: chi tiết nhỏ nhưng quan trọng

Ảnh MNIST được tạo theo một quy trình cố định: chữ số được thu vào khung 20×20 (giữ tỉ lệ), rồi đặt vào ảnh 28×28 sao cho **trọng tâm (center of mass)** nằm giữa ảnh. Hình người dùng vẽ thì có thể nhỏ, to hoặc lệch góc, nên cả hai giao diện đều chuẩn hóa hình vẽ theo đúng quy trình đó trước khi so với centroid:

1. Cắt sát nét vẽ.
2. Thu vào khung 20×20, giữ tỉ lệ.
3. Đặt vào ảnh 28×28 rồi dịch để trọng tâm nằm ở tọa độ **(14, 14)**.

Có một bài học thú vị ở bước 3. Ban đầu trọng tâm được đặt ở (13,5; 13,5), tức tâm hình học của lưới 28×28 tính theo chỉ số pixel. Chỉ lệch nửa ô như vậy mà độ chính xác trên 2.000 ảnh test tụt từ 87,65% xuống 82,6%. Đổi sang (14, 14) thì khớp lại hoàn toàn: đưa cả tập test qua bước chuẩn hóa vẫn đạt 90,78%, so với 90,77% trên ảnh gốc (K = 200). Với thuật toán so sánh từng điểm ảnh như K-means, lệch nửa pixel cũng đáng kể.

## Hai giao diện

### Desktop: Tkinter (`digit_app.py`)

- Bảng vẽ 28×28 ô. Chuột trái để vẽ, chuột phải để tẩy. Cọ đậm ở giữa và nhạt dần ra mép, giống nét chữ trong MNIST.
- Nhả chuột là có đáp án, kèm ảnh sau chuẩn hóa, centroid gần nhất và 5 cụm gần nhất.

### Web: `web/index.html`

- HTML, CSS và JavaScript thuần, tính toán ngay trong trình duyệt, chạy được khi không có mạng.
- Nhận diện ngay khi đang vẽ (có thể tắt), dùng được với chuột, bút và cảm ứng.
- Thanh khoảng cách tới từng chữ số 0–9: thấy được hình vẽ đang gần chữ số nào và "đối thủ" sát nhất là ai.
- Bộ sưu tập toàn bộ centroid, xếp theo chữ số. Cụm khớp nhất được làm nổi bật.
- Giao diện dùng [Sakuga VFX](https://github.com/finktheartist/sakuga-vfx): nút, tab, thanh trượt, công tắc và tooltip phản ứng như phim anime (nhún lò xo, khói, vụ nổ, chữ ドン!). Ô hiện đáp án là một thành phần tự viết theo cùng phong cách: nổ hiệu ứng mỗi khi đáp án thay đổi lúc nhả chuột. Hiệu ứng tự tắt nếu máy bật chế độ giảm chuyển động.
- **Đã kiểm tra:** phần nhận diện bằng JavaScript cho kết quả trùng với bản Python trên cả 1.000/1.000 ảnh test thử, mỗi ảnh khoảng 0,3 ms. Centroid được lượng tử hóa về 1 byte/pixel để file nhỏ gọn. Trên 10.000 ảnh test, việc này chỉ làm đổi 3 dự đoán (92,15% còn 92,14%).

Phím tắt: `B` để vẽ, `E` để tẩy, `C` để xóa bảng.

## Giới hạn

- **So sánh từng điểm ảnh.** Khoảng cách Euclid không hiểu hình dạng. Chữ số vẽ quá mảnh, quá đậm, bị xoay nghiêng hoặc theo kiểu lạ (ví dụ số 1 có chân đế, số 7 có gạch ngang ở giữa) có thể bị nhận sai.
- **Phân biệt 4 và 9, 3 và 5** vẫn là điểm yếu, đúng như bảng phân tích K = 10 ở trên.
- **Không phải một mô hình phân loại "thật sự".** Các phương pháp có giám sát (SVM, mạng nơ-ron tích chập) đạt trên 99% trên MNIST. Dự án này nhằm hiểu K-means, không nhằm đạt độ chính xác cao nhất.
- Bản web mới được thử trên trình duyệt nhân Chromium. Theo tác giả Sakuga VFX, thư viện đó chưa được thử trên Safari và Firefox.

## Nguồn và giấy phép

- **Dữ liệu MNIST** của Yann LeCun, Corinna Cortes và Christopher J.C. Burges, phát hành theo giấy phép [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
- **Sakuga VFX** của Justin Finkelman, giấy phép MIT. Bản sao giấy phép nằm ở [`web/sakuga/LICENSE`](web/sakuga/LICENSE).
