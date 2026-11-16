# HotelRec — mẫu sạch từ toàn bộ archive

> Đây là checkpoint chuẩn bị dữ liệu trước enrichment. Sau đó đã bổ sung
> hometown tự khai cho 693/1.000 user và chạy lại seed 42; xem
> [báo cáo demographic](hotelrec_demographic_report.md) cho trạng thái mới nhất.

Đã hoàn tất tải ZIP, quét toàn bộ 50.264.531 review, xác minh checksum/CRC,
làm sạch ID thiếu và dựng tập con. **Chưa huấn luyện mô hình trên mẫu mới.**

Nguồn: [HotelRec — Antognini & Faltings, LREC 2020](https://github.com/diegoantognini/HotelRec).
Chỉ dùng cho nghiên cứu học thuật.

## Audit dữ liệu gốc

| Bước | Số lượng |
|---|---:|
| Review đã quét | 50.264.531 |
| Loại do schema/rating/ngày/ID thiếu theo bộ lọc ban đầu | 23.890 |
| Loại thêm do author `/undefined` | 1.193.017 |
| Loại thêm do author `Undefined` | 12 |
| Review hợp lệ sau toàn bộ bộ lọc | 49.047.612 |
| Cặp user–hotel khác nhau | 47.731.590 |
| User ID còn lại | 21.891.400 |
| Hotel có tương tác hợp lệ | 365.011 |

`/undefined` là placeholder, không phải một tài khoản cá nhân. Nó đã gom review
của 38.820 hotel vào cùng một ID. Sau khi loại placeholder, user có nhiều hotel
nhất trong toàn dữ liệu còn 580 hotel. Có 174.620 user đạt ít nhất 20 hotel và
68.455 user đạt ít nhất 30 hotel; phần lớn user thật vẫn có lịch sử rất thưa.

**Kết quả cũ tag `hotelrec_subset_1m` không được dùng cho báo cáo.** Cache prefix
cũ có 26.695 review của `/undefined`; cả 8 rating warm trong test cũ đều thuộc
ID này. CSV cũ được giữ để truy vết và báo cáo cũ đã gắn cảnh báo. Loader hiện
loại placeholder kể cả khi đọc cache cũ.

ZIP gốc được giữ nguyên. Việc làm sạch SQLite được thực hiện trong transaction;
metadata ghi lại số review và cặp tương tác bị loại. Không tạo user giả để thay
thế các review thiếu ID.

## Cách chọn mẫu

1. Đếm số **user khác nhau** của mỗi hotel trên toàn bộ dữ liệu sạch.
2. Chọn tối đa 2.000 hotel có nhiều user nhất, đồng hạng sắp theo ID chuỗi.
3. Trong catalog đó, chọn 1.000 user có nhiều **hotel khác nhau** nhất,
   đồng hạng sắp theo user ID.
4. Mỗi cặp user–hotel giữ rating mới nhất theo timestamp; đồng thời điểm giữ
   review xuất hiện sau trong archive.
5. Lọc lặp đến khi user có ít nhất 3 hotel và hotel có ít nhất 5 user.

Catalog ban đầu 1.000 hotel chỉ cho 3–7 warm user trong 10 seed và vẫn chứa
placeholder trước audit. Sau audit, mở catalog lên 2.000 hotel để giữ thêm lịch
sử user. Việc chọn này chỉ dựa trên độ phủ tương tác, **chưa xem metric mô hình**.

| Dữ liệu mẫu cuối | Giá trị |
|---|---:|
| User | 1.000 |
| Hotel | 1.260 |
| Rating không trùng cặp | 21.700 |
| Hotel/user trung bình | 21,7 |
| Hotel/user: nhỏ nhất / trung vị / lớn nhất | 6 / 20 / 58 |
| Tứ phân vị 25% / 75% | 19 / 24 |

Các hotel không đủ tương tác trong nhóm user được chọn bị loại ở bước lọc lõi,
nên catalog cuối còn 1.260 hotel thay vì 2.000. Loader dùng giới hạn trong
`sampling_limits` của `source.json`, không vô tình cắt lại xuống 1.000 hotel.

## Kiểm tra 10 cách chia dữ liệu

Seeds: `42 1 2 3 4 5 6 7 8 9`. Dùng đúng cấu hình thí nghiệm hiện tại: 20% hotel
cold-start; 30% user được mô phỏng cắt lịch sử; warm có ít nhất 20 rating train,
cool có 5–19, cold có dưới 5. Chỉ tính user xếp hạng nếu test có rating >=4.

| Nhóm user xếp hạng | Ít nhất / nhiều nhất qua 10 seed |
|---|---:|
| Cold | 56 / 89 |
| Cool | 790 / 857 |
| Warm | 37 / 66 |

Tất cả 10 seed đều có đủ ba nhóm. Đây là kiểm tra độ phủ dữ liệu, **không phải**
kết quả chạy mô hình qua 10 seed hay bằng chứng α động tốt hơn.

Riêng seed 42:

| Phần | Số lượng |
|---|---:|
| Train / validation / test | 12.564 / 4.599 / 4.456 |
| Rating test: cold-item / cold-user / cool / warm | 2.053 / 564 / 1.707 / 132 |
| User xếp hạng: cold / cool / warm | 76 / 794 / 63 |
| Tổng user xếp hạng | 933 |
| Cặp ứng viên toàn catalog mẫu, sau loại train+validation | 1.159.548 |

Train+validation+test ít hơn 21.700 vì protocol hiện tại giới hạn tối đa 20
rating held-out cho mỗi user được mô phỏng cắt lịch sử; seed 42 bỏ qua 81 rating
ở bước này. Không thay đổi protocol, α, grid search hoặc thuật toán mô hình.

## Kiểm tra và khả năng chạy lại

- 176 test pass, gồm kiểm tra ID thiếu, làm sạch cache cũ, khử trùng rating mới
  nhất, áp dụng đúng giới hạn catalog và đối chiếu số ứng viên với hàm đánh giá.
- MD5 archive khớp nguồn: `671e0290514f8411fe780fdea236783a`.
- SHA256 archive: `160314db15080ecc15b547e9a235dfa971d388a88227e220147dafce3964aaff`.
- Đã đọc đến cuối `HotelRec.txt` và kiểm tra ZIP CRC. Các file slim có SHA256
  riêng trong `datasets_raw/hotelrec_full_sample/source.json`.
- Bộ quét dùng SQLite trên đĩa và `orjson 3.11.9` đã có sẵn. Không cài dependency
  mới. Bước index đầu tiên mất khoảng 31 phút, chưa tính tải và audit bổ sung.
- Dữ liệu chuẩn bị nằm trong `datasets_raw/hotelrec_full_sample/`; ZIP và SQLite
  hoàn chỉnh nằm trong `datasets_raw/hotelrec/`.

```powershell
conda activate stayreco
python -c "from prepare_hotelrec_full import profile_sample; profile_sample('datasets_raw/hotelrec_full_sample', 'results/hotelrec_full_sample_profile.json')"

# Bước tiếp theo: kiểm tra mô hình bằng một seed trên mẫu sạch.
python run_experiment.py 42 --dataset=hotelrec-full-sample --cold-item=0.2 --tag=hotelrec_full_sample
```

- [Profile JSON và provenance](hotelrec_full_sample_profile.json)
- [Độ phủ từng seed](hotelrec_full_sample_profile.csv)
- [Log profile](hotelrec_full_sample_profile.log)
- [Log kiểm tra](tests_hotelrec_full.log)

## Giới hạn còn lại

Mẫu thiên về hotel phổ biến và user hoạt động nhiều, dùng để mô phỏng cold-start;
không đại diện phân phối toàn dân số HotelRec. Nhóm warm đã có dữ liệu thực
nhưng vẫn nhỏ hơn nhóm cool. Content chỉ dùng tên/địa danh từ URL, không có
review text của test làm đặc trưng. HotelRec thiếu demographic nên thành phần
nhân khẩu vẫn là fallback `unknown`.

Chưa chạy mô hình, so sánh accuracy hoặc kiểm định thống kê trên mẫu mới.
