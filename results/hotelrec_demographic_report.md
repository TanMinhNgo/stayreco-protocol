# HotelRec — demographic địa lý và lần chạy seed 42

Đã bổ sung vùng cư trú tự khai cho HotelRec, tích hợp vào DemographicFilter
hiện có và chạy đủ 14 phương pháp trên mẫu sạch: 1.000 user, 1.260 hotel,
21.700 rating. Không thay đổi α, grid search hay thuật toán các model thành phần.

## Dữ liệu demographic thực sự có

HotelRec gốc không có hồ sơ nhân khẩu. Đã nối với file `authors.json.gz` của bộ
TripAdvisor US có sẵn trên máy, theo **username chính xác, phân biệt hoa/thường**.
Nguồn tham khảo bộ dữ liệu: [Mrázová & Behún, ITAT 2023](https://ceur-ws.org/Vol-3498/paper8.pdf).
Schema username/hometown được kiểm tra trực tiếp trong file nguồn.

| Độ phủ trong mẫu HotelRec | User |
|---|---:|
| Tổng | 1.000 |
| Khớp profile có hometown | 693 |
| Khớp profile nhưng thiếu hometown | 147 |
| Không khớp username | 160 |
| Tổng khớp profile | 840 |

Độ phủ vùng cư trú là **69,3%**, tạo **82 nhóm**. Không gặp trường hợp username
khớp nhiều profile ID hoặc hometown xung đột trong mẫu thật. Code loại những
trường hợp đó và đã có test riêng cho cả hai tình huống.

`home_region` là phần cuối sau dấu phẩy của hometown, chuẩn hóa khoảng trắng
và casefold. Ví dụ `New York City, New York` thành `new york`. Giá trị có thể
là bang, quốc gia hoặc thành phố; không diễn giải thành quốc tịch. Không có
dữ liệu tuổi/giới tính, không suy đoán các thuộc tính này từ tên hay review.

Các nhóm lớn nhất: United Kingdom 238 user; Australia 56; California 47;
Florida 28; Canada 23; New York 20. Có 307 user thiếu vùng cư trú.

## Hành vi của model

Loader đưa vùng cư trú vào cột `demographic` hiện có. Hồ sơ thiếu dùng giá trị
thiếu thật; DemographicFilter bỏ những profile này khỏi việc học nhóm.
User thiếu profile dùng trung bình hotel có shrinkage, rồi trung bình toàn
train nếu hotel chưa biết. Không học một cohort chung từ nhãn `unknown`.

User mới chưa có lịch sử nhưng có vùng cư trú vẫn có thể nhận dự đoán của nhóm
đã khai. Toàn bộ trung bình nhóm/hotel được học từ **train**, không dùng rating
của validation/test. Snapshot profile ngoài chỉ cung cấp hometown, không lấy
trip_type hoặc rating từ bộ review TripAdvisor US làm đặc trưng cho HotelRec.

Seed 42 học được **5.330 cặp vùng cư trú–hotel**. Train/validation/test lần lượt
là 12.564/4.599/4.456 rating. Xếp hạng trên 1.159.548 cặp ứng viên của 933 user,
toàn catalog mẫu sau khi loại item đã có trong train/validation.

## Kết quả test — một seed

RMSE thấp hơn là tốt; NDCG@10 cao hơn là tốt.

| Phương pháp | RMSE | NDCG@10 |
|---|---:|---:|
| Popularity | 0,9008 | 0,0045 |
| CF Only (SVD++) | 0,8711 | 0,0053 |
| Content Only | 0,8776 | **0,0112** |
| Demographic Only | 0,9039 | 0,0055 |
| Hybrid α=0.3 cố định | 0,8713 | 0,0074 |
| Hybrid α=0.5 cố định | 0,8693 | 0,0065 |
| Hybrid α=0.7 cố định | **0,8689** | 0,0059 |
| StayReco α thủ công | 0,8734 | 0,0093 |
| Tarek 2023 | 0,8801 | 0,0070 |
| Do & Le 2020, 2 thành phần | 0,8691 | 0,0086 |
| Do & Le 2020, mở rộng 3 thành phần | 0,8722 | 0,0065 |
| OHWSF 2022 | 0,8697 | 0,0059 |
| SL-ARec 2026 | 0,8758 | 0,0054 |
| StayReco α tuned | 0,8725 | 0,0084 |

Demographic đã có dữ liệu và hoạt động, nhưng RMSE riêng chưa tốt hơn Popularity
trên seed này. NDCG@10 tăng từ 0,0045 lên 0,0055 khi so hai dòng này. Đây không
phải bằng chứng về lợi ích của demographic trong hybrid; chưa chạy ablation
hybrid có/không có geography trên nhiều seed.

StayReco tuned chọn trên validation theo RMSE: `alpha_max=0.5`,
`n_threshold=50`, `gamma_base=0.2`, `alpha_min=0.3`. Không kết luận α động thắng
toàn diện; fixed α=0.7 tốt hơn về RMSE trong lần chạy này.

## Kiểm tra và provenance

- **182 test pass**: exact join, profile xung đột, checksum/target mismatch,
  thiếu metadata, cold user có vùng cư trú, thiếu profile fallback và batch
  prediction khớp scalar. Các test cũ vẫn qua.
- Kiểm tra đủ 14 dòng kết quả, seed/config, miền giá trị metric và không có
  NaN/Inf; manifest ghi 693 known / 307 missing / 82 nhóm.
- Đã xem cả hai biểu đồ; không có lỗi tạo chart. Các tiến trình chạy đã kết thúc.
- File profile nguồn local có 2.296.248 dòng; SHA256:
  `da741b8b3433aa80fc135912b5d7cdce4b79dfa18b04a7b1e32c56e6c3368c15`.
- Sidecar và sample rating đích có checksum riêng trong
  `datasets_raw/hotelrec_full_sample/demographics_source.json`.
- Enrichment không thay đổi rating, hotel hay tập user của mẫu sạch.
- Working tree còn thay đổi chưa commit; không thể tái lập chỉ từ commit hash.

```powershell
# Chuẩn bị một lần; script từ chối ghi đè file enrichment đã có.
python prepare_hotelrec_demographics.py
conda activate stayreco
python -m pytest tests/ -q
python run_experiment.py 42 --dataset=hotelrec-full-sample --cold-item=0.2 --tag=hotelrec_geo_seed42
```

- [Metrics đầy đủ](comparison_hotelrec_geo_seed42.csv)
- [Metrics theo nhóm](comparison_by_group_hotelrec_geo_seed42.csv)
- [Manifest cấu hình, nguồn và độ phủ profile](run_meta_hotelrec_geo_seed42.json)
- [Biểu đồ theo nhóm](comparison_hotelrec_geo_seed42.png)
- [Biểu đồ độ nhạy α](sensitivity_alpha_hotelrec_geo_seed42.png)
- [Log chạy](hotelrec_geo_seed42.log)
- [Log kiểm tra](tests_hotelrec_demographic.log)

## Giới hạn cần ghi khi sử dụng kết quả

Hồ sơ thuộc một đợt thu thập khác và không có ngày quan sát từng profile.
Không bảo đảm hometown đã có/giữ nguyên tại ngày review HotelRec lịch sử.
Username chính xác là khóa nối giữa hai bộ; HotelRec không có immutable profile
ID để chứng minh danh tính xuyên snapshot. Kết quả phù hợp để thăm dò enrichment,
chưa phải đánh giá temporal cold-start nghiêm ngặt.

Geography chỉ là một thuộc tính nhân khẩu; tuổi/giới tính vẫn thiếu, 30,7% user
không có vùng cư trú. Mẫu thiên về hotel phổ biến và user hoạt động nhiều.
Mới chạy model với một seed, chưa kiểm định độ ổn định qua nhiều seed.
Kết quả cũ `hotelrec_subset_1m` có placeholder `/undefined` vẫn không hợp lệ.
