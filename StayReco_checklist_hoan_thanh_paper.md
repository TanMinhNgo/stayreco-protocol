# Checklist hoàn thành paper StayReco

> Bài: *Interaction-Dependent Hybrid Weighting for Cold-Start Hotel Recommendation: A Two-Stage Design and Evaluation Protocol for StayReco*
> File chính: `main_ceur.tex` (định dạng CEUR-WS, hai cột, hiện 8 trang)
> Cập nhật: 16/11/2026
>
> Đánh dấu `[x]` khi xong. Mục nào chưa có câu trả lời thì ghi "chưa có". Tuyệt đối không điền số liệu giả định.

---

## Quyết định cần chốt trước

- [x] **Hướng 1 hay Hướng 2?** → **Hướng 2** (đã chốt)
  - **Hướng 1:** nộp dạng bài đề xuất, không có dataset. Chỉ cần làm Phần 1, 2, 3, 6.
  - **Hướng 2:** có kết quả thực nghiệm. Làm đủ Phần 1 đến 6.
- [x] Nếu Hướng 2, chọn nguồn dữ liệu → **(b) HotelRec** (Fliggy chưa dùng):
  - **(a)** LI-5: dữ liệu tổng hợp do nhóm tự sinh
  - **(b)** Bộ công khai: Fliggy (chính) và/hoặc HotelRec
  - **(c)** Log thật của StayReco (chỉ khi đã có người dùng thật)

---

## PHẦN 1 — Thông tin hành chính (bắt buộc)

| Xong | Cần điền | Định dạng | Câu trả lời |
|---|---|---|---|
| [ ] | Tên hội nghị | Tên đầy đủ, ngày, thành phố, quốc gia | |
| [ ] | Sinh viên 1 | Họ tên, email trường | |
| [ ] | Sinh viên 2 | Họ tên, email trường | |
| [ ] | Sinh viên 3 | Họ tên, email trường | |
| [ ] | Sinh viên 4 | Họ tên, email trường | |
| [ ] | Sinh viên 5 | Họ tên, email trường | |
| [ ] | Giảng viên hướng dẫn | Họ tên, học hàm/học vị, email | |
| [ ] | Đơn vị | Khoa, trường, thành phố, Việt Nam (ghi riêng nếu GVHD khác khoa) | |
| [ ] | Lời cảm ơn | Tên GVHD; nguồn tài trợ và mã đề tài (nếu có) | |
| [ ] | Khai báo AI | Xác nhận nhóm khai đúng việc đã dùng AI để soạn bài và kiểm tra tài liệu (CEUR bắt buộc) | |
| [ ] | Quy định hội nghị | Giới hạn trang; một cột hay hai cột; ngôn ngữ; hạn nộp | |

---

## PHẦN 2 — Mô tả nền tảng StayReco (mục 3 của bài)

- [ ] Mô tả hiện tại có đúng không? Bài đang viết: trang web tiếng Việt; tìm theo điểm đến, ngày, ngân sách; partner portal gồm dashboard, booking, kho phòng, lịch tồn, luật giá, khuyến mãi, nhân viên, hồ sơ và xác minh; hiện là bản prototype.
  - Trả lời: 
- [ ] Nền tảng đang ở dạng demo hay đã có người dùng thật?
  - Trả lời: 
- [ ] Partner nhập những trường nào cho một khách sạn? (chép từ form trong partner portal)
  - Trả lời: 
- [ ] Thuật toán gợi ý đã lập trình chưa? (có / không / một phần)
  - Trả lời: 
- [ ] Đã ghi log hành vi người dùng chưa? Ghi những sự kiện nào (view / click / save / book)?
  - Trả lời: 

---

## PHẦN 3 — Chi tiết thuật toán (mục 4 của bài)

> Mục nào chưa quyết thì ghi "để giai đoạn cài đặt". Câu trong bài sẽ được sửa theo đúng tình trạng đó.

| Xong | Câu hỏi | Ví dụ | Câu trả lời |
|---|---|---|---|
| [x] | Content dùng đặc trưng nào, biểu diễn ra sao | Tiện ích (one-hot), hạng sao, giá, quận; mô tả dùng TF-IDF | HotelRec chỉ có tên + địa điểm từ URL: TF-IDF trên chuỗi đó, cosine với hồ sơ user (`protocol/sources.py`) |
| [x] | Hồ sơ Content của khách chưa khai báo sở thích | Trung bình đều, hay trung bình có trọng số 1/2/3/5 | Trung bình đều vector TF-IDF các hotel đã review (mỗi review n_u += 1, không có view/click) |
| [x] | Demographic chia nhóm theo gì | Nhóm tuổi × loại chuyến đi | Chỉ `home_region` (ghép username với tác giả TripAdvisor Mỹ), 53% user; còn lại nhóm "unknown" |
| [x] | Cửa sổ ngày ở bước R2 | ±3 ngày | Không áp dụng: HotelRec không có Stage 1 (phòng trống, ngày, ngân sách) |
| [x] | Tín hiệu x để ước lượng k | "Đồng ý đề xuất" (x = 1 nếu sự kiện thuộc khách sạn cuối cùng được đặt, ngược lại 0) | Dùng rating của review làm x; prior μ_u = trung bình rating trung bình của các hotel user đã review; k̂ = 6,94 |
| [x] | Mức ý nghĩa, số lần bootstrap | "Mặc định" (0,05; 10.000) | 0,05; 10.000 |

---

## PHẦN 4 — Dataset (chỉ khi chọn Hướng 2)

### 4.1 Cấu trúc dữ liệu cần có (cho cả nguồn a, b, c)

**File 1: `interactions.csv` (bắt buộc)**

| Cột | Ý nghĩa | Ví dụ |
|---|---|---|
| user_id | Mã người dùng | u0001 |
| hotel_id | Mã khách sạn | h0042 |
| event_type | view / click / save / book | click |
| timestamp | Thời điểm xảy ra | 2026-03-15 14:22:05 |

**File 2: `hotels.csv` (bắt buộc)**

| Cột | Ví dụ |
|---|---|
| hotel_id | h0042 |
| city | Đà Nẵng |
| star | 4 |
| price | 1200000 |
| amenities | wifi;pool;breakfast |
| description | "Khách sạn gần biển…" |
| created_at | Ngày khách sạn lên sàn (để xác định khách sạn mới) |

**File 3: `users.csv` (cần cho thành phần Demographic)**

| Cột | Ví dụ |
|---|---|
| user_id | u0001 |
| age_group | 25–34 |
| trip_type | gia đình |
| declared_preferences | wifi;pool (để trống nếu khách không khai báo) |

**File 4: `searches.csv` (không bắt buộc, có thì tốt hơn)**

| Cột | Ví dụ |
|---|---|
| search_id | s01 |
| user_id | u0001 |
| timestamp | 2026-03-15 14:20 |
| city | Đà Nẵng |
| checkin, checkout | 2026-04-01, 2026-04-03 |
| budget | 1500000 |
| booked_hotel_id | h0042 (để trống nếu không đặt) |

### 4.2 Quy mô tối thiểu (kinh nghiệm chung, không phải quy định)

- [x] Dữ liệu trải dài ít nhất vài tháng (2002-09 đến 2019-05; test từ 2018-03) (để chia train / validation / test theo thời gian)
- [x] Tập test có ít nhất khoảng **1.000 booking** (32.518 booking đánh giá được)
- [ ] Mỗi nhóm người dùng (n = 0, 1–4, 5–19, ≥ 20) có ít nhất vài trăm booking trong tập test — **chưa đạt**: n=0 có 288, 1–4 có 293 booking
- [ ] Có một phần khách sạn mới lên sàn trong giai đoạn test — **yếu**: chỉ 82 booking hotel mới

### 4.3 Theo từng nguồn

**Nguồn (a) — LI-5 tự sinh**
- [ ] 3–4 file theo cấu trúc ở mục 4.1
- [ ] Script sinh dữ liệu (kèm seed cố định)
- [ ] Đoạn mô tả quy tắc sinh: hành vi phụ thuộc vào gì; phân phối số sự kiện mỗi người; số user, khách sạn, thành phố, khoảng thời gian
- Lưu ý: tránh sinh hành vi chỉ từ chính các đặc trưng mà Content dùng, nếu không kết quả sẽ thiên về Content

**Nguồn (b) — Bộ công khai**
- [ ] **Fliggy** (tải từ Tianchi, cần đăng nhập): `user_item_behavior_history.csv`, `user_profile.csv`, `item_profile.csv`, kèm ảnh chụp trang điều khoản sử dụng
- [x] **HotelRec** (link GitHub trong paper): file đầy đủ, **không** lấy bản 5-core / 20-core (đã tải bản đầy đủ 14,38 GB, kiểm MD5/CRC)
- [x] Ghi lại: tải từ đâu, ngày tải, dung lượng từng file (`datasets_raw/hotelrec_*/source.json`, ngày tải 2026-11-16)
- Lưu ý: Fliggy là sản phẩm du lịch nói chung; HotelRec không có thông tin nhân khẩu học; cả hai không có file searches. Stage 1 (lọc phòng trống / giá) sẽ không đánh giá được trên hai bộ này
- Không dùng **Booking.com 515K** để đánh giá CF, vì bộ này không có mã người đánh giá

**Nguồn (c) — Log thật StayReco**
- [ ] Xuất dữ liệu từ database theo cấu trúc ở mục 4.1
- [ ] Ẩn danh hóa: không xuất tên, email, số điện thoại
- [ ] Ghi khoảng thời gian thu thập; người dùng có đồng ý cho dùng dữ liệu không

### 4.4 Cách gửi dữ liệu

- **Không** mở rồi lưu bằng Excel (Excel giới hạn khoảng 1 triệu dòng, có thể làm hỏng cột ID và ngày)
- **Không** đổi tên cột, xóa dòng, tự lọc hay đổi định dạng
- **Cách A (khuyên dùng):** đặt file vào một thư mục trên máy, ví dụ `Documents/StayReco_datasets/` với các thư mục con `fliggy/`, `hotelrec/`, `li5/`, rồi báo đường dẫn để cấp quyền truy cập
- **Cách B:** file nhỏ thì đính kèm thẳng vào chat (có thể nén `.zip`)

---

## PHẦN 5 — Kết quả thực nghiệm (chỉ khi chọn Hướng 2)

Chạy bằng (mẫu chọn chỉ từ thông tin trước mốc cắt, xem `prepare_protocol_sample.py`):

```
python prepare_protocol_sample.py
python prepare_hotelrec_demographics.py --folder datasets_raw/hotelrec_protocol_sample_v2
python run_protocol.py --sample datasets_raw/hotelrec_protocol_sample_v2 --tag v2_s42 --seed 42   # và seed 1, 2, 3, 4
```

File kết quả: `results/protocol_*_v2_s{42,1,2,3,4}.csv`. Chạy lại seed 42 (`v2_rep`) cho kết quả giống hệt (chênh 0,0).
**Mỗi review HotelRec là một booking; n_u đếm số review (không có view/click).** Các lần chạy trước (`hrec*`, `INVALID_*`) dùng mẫu chọn bằng dữ liệu tương lai và đã bị thay thế; không dùng.

- [x] Bảng thống kê dữ liệu (`protocol_stats_v2_s42.csv`): 32.448 user, 5.000 hotel, 192.817 review, 2002-10-01 đến 2019-05-13; 8.044 user (25%) có `home_region`
- [x] Mốc chia: train < 2017-08-01 ≤ validation < 2018-05-01 ≤ test (152.543 / 19.088 / 21.186 review)
- [x] NDCG@10 theo nhóm n_u, hotel mới, toàn bộ: `protocol_ndcg_v2_s*.csv` (bảng dưới, trung bình 5 seed)
- [x] Recall@10 tương tự: `protocol_recall_v2_s*.csv` (thực chất là tỷ lệ trúng theo từng booking, nên gọi HR@10 trong bài)
- [x] k = 25, k_c = 10 chọn trên validation (cả 5 seed cùng chọn); k̂ = 7,11 (tín hiệu = rating, prior leave-one-out)
- [x] n\*: **không xác định**: ở mọi nhóm n_u, CF only không vượt Content only có ý nghĩa (`protocol_nstar_v2_s*.csv`)
- [x] Kiểm định trên user (gộp booking của cùng user), paired t-test + bootstrap 10.000 lần, hiệu chỉnh Holm: `protocol_tests_v2_s*.csv`

NDCG@10 trên test, trung bình 5 seed (14.443 booking đánh giá được mỗi seed; nhóm: n=0 có 3.212, 1–4 có 3.579, 5–19 có 5.213, ≥20 có 2.439, hotel mới có 118 booking):

| Phương pháp | n=0 | 1–4 | 5–19 | ≥20 | Hotel mới | Tất cả |
|---|---|---|---|---|---|---|
| MostPop-City | 0,112 | 0,146 | 0,178 | 0,222 | 0,000 | 0,163 |
| CF only (BPR) | 0,084 | 0,120 | 0,147 | 0,183 | 0,036 | 0,132 |
| Content only | 0,084 | 0,126 | 0,150 | 0,177 | 0,095 | 0,134 |
| Demographic only | 0,087 | 0,102 | 0,123 | 0,156 | 0,077 | 0,115 |
| Best fixed α (α=0,3; k_c=10) | 0,087 | 0,141 | 0,181 | 0,218 | 0,070 | 0,157 |
| Dynamic α (k=25; k_c=10) | 0,087 | 0,146 | 0,180 | 0,223 | 0,077 | 0,158 |
| Dynamic + π_g (theo sai số) | 0,087 | 0,141 | 0,179 | 0,227 | 0,074 | 0,157 |
| Dynamic + RRF | 0,087 | 0,137 | 0,174 | 0,218 | 0,067 | 0,153 |
| FWLS (logistic) | 0,087 | 0,146 | 0,182 | 0,231 | 0,068 | 0,160 |

Độ lệch chuẩn giữa 5 seed của cột "Tất cả" là 0,002–0,003 (MostPop 0,0002); của cột "Hotel mới" là 0,03.

Đọc kết quả (kiểm định trên user, p đã hiệu chỉnh Holm, seed 42):
- H1: Dynamic cao hơn best fixed 0,0005 NDCG@10 toàn bộ, **không có ý nghĩa** (p = 0,61). Không nhóm nào có khác biệt có ý nghĩa sau Holm (nhóm ≥20 có p thô = 0,031, Holm = 0,78).
- H2 **không được ủng hộ**: ở n=0 và 1–4 Dynamic không hơn best fixed; ở hotel mới Dynamic hơn best fixed 0,008 (p = 0,21).
- Ở n=0, **MostPop-City tốt hơn Dynamic** 0,020 (Holm p = 0,008). Ở hotel mới Dynamic tốt hơn MostPop-City 0,070 (Holm p = 0,001). Toàn bộ: MostPop-City 0,163 so với Dynamic 0,158, không có ý nghĩa (p = 0,21).
- FWLS cao hơn Dynamic 0,0025 toàn bộ (p thô = 0,0008, Holm = 0,025), là cải thiện duy nhất có ý nghĩa; π_g và RRF không hơn Dynamic có ý nghĩa.
- Dynamic tốt hơn Content only ở mọi nhóm trừ hotel mới.
- H3: n\* không xác định; k̂ = 7,11 so với k = 25 chọn trên validation.

Giới hạn cần ghi vào bài: review thay cho booking; không có Stage 1; 32% booking test bị bỏ vì hotel không suy ra được vùng (đều quanh 30–33% ở mọi nhóm n_u, nên không lệch thành phần nhóm); URL HotelRec bị cắt ở 89 ký tự; Demographic chỉ có `home_region` cho 25% user (user còn lại không có điểm Demographic); mốc "hotel mới" dựa trên review đầu tiên (proxy cho ngày lên sàn); hotel mới chỉ 118 booking; seed chỉ đổi khởi tạo BPR, không đổi cách chia dữ liệu; không có SVD++/iALS; BPR chưa tune siêu tham số trên validation.

---

## PHẦN 6 — Việc cuối

- [x] Duyệt các quyết định thiết kế D1–D9 → **"đồng ý hết"** (người dùng xác nhận):
  - D1: khách sạn mới nhận điểm CF chuẩn hóa = 0
  - D2: điểm Demographic = tần suất được đặt trong cùng phân khúc
  - D3: baseline α cố định chia 1−α theo cùng quy tắc π
  - D4: thêm baseline "chỉ Demographic"
  - D5: các biện pháp dự phòng (τ̂² ≤ 0, trọng số FWLS âm, mở rộng lưới k)
  - D6: so sánh n* với k̂
  - D7: ghi nhãn "our adaptation" cho RRF có trọng số và FWLS logistic
  - D8: exploration (ví dụ 2/10 vị trí) chỉ nhắc như hướng phát triển
  - D9: thêm hạn chế về tính vòng của dữ liệu tổng hợp và về dữ liệu nhân khẩu học
- [ ] Tự tìm lại trên Google Scholar / ACM DL / IEEE Xplore để chắc khẳng định về tính mới vẫn đúng
- [ ] Kiểm tra liêm chính + phản biện mô phỏng + sửa bài (sau khi xong các phần trên)
- [ ] Biên dịch bản cuối trên Overleaf (`main_ceur.tex`, `refs.bib`, `ceurart.cls`, pdfLaTeX) và đọc lại toàn bộ

---

## Tóm tắt khối lượng

| Hướng | Phần cần làm |
|---|---|
| Hướng 1 (đề xuất) | Phần 1, 2, 3, 6 |
| Hướng 2 (có kết quả) | Phần 1, 2, 3, 4, 5, 6 |
