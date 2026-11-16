# HotelRec — tập con thăm dò, seed 42

> **KHÔNG DÙNG KẾT QUẢ NÀY CHO BÁO CÁO.** Audit toàn archive phát hiện
> `/undefined` là ID thiếu bị gom thành một user. Cache prefix có 26.695 review
> thuộc placeholder này; cả 8 rating nhóm warm trong test dưới đây đều thuộc
> `/undefined`. Các CSV được giữ để truy vết, không phải bằng chứng hợp lệ về
> người dùng thật hay cold-start. Loader hiện đã loại placeholder. Dùng mẫu
> toàn archive sạch và chạy lại mô hình trước khi so sánh chất lượng.

Đã chạy đủ 14 phương pháp ngày 16/11/2026. Content Only đạt RMSE và NDCG@10
tốt nhất trên tập con này. Kết quả chưa đủ để khẳng định StayReco α động vượt
α cố định hoặc tổng quát hóa ra toàn bộ HotelRec.

Nguồn: [HotelRec, Antognini & Faltings, LREC 2020](https://github.com/diegoantognini/HotelRec).
Dữ liệu chỉ dùng cho nghiên cứu học thuật.

## Dữ liệu thực sự đã chạy

| Bước | Quy mô |
|---|---:|
| Review đầu tiên đã quét từ HotelRec.txt | 1.000.000 |
| Review hợp lệ | 999.566 |
| Khách sạn trong phần đã quét | 2.403 |
| User thật sau giới hạn và lọc | 439 |
| Khách sạn sau giới hạn và lọc | 190 |
| Cặp user–hotel sau khử trùng và lọc | 1.625 |
| Train / validation / test | 1.059 / 266 / 300 |
| Hotel cold-start, giấu toàn bộ rating khỏi train | 38 |
| User được chấm xếp hạng / cặp ứng viên | 182 / 34.074 |

Chọn tối đa 1.000 hotel nhiều review nhất, rồi tối đa 1.000 user nhiều tương tác
nhất, đồng hạng sắp theo ID. Mỗi cặp user–hotel giữ rating mới nhất. Lọc lặp đến
khi user có ít nhất 3 hotel và hotel có ít nhất 5 user. Các ngưỡng không được
chọn theo kết quả test. Xếp hạng trên toàn bộ **190 hotel của tập con**, loại
những hotel user đã đánh giá trong train/validation.

## Kết quả test

RMSE thấp hơn là tốt; P@10 và NDCG@10 cao hơn là tốt.

| Phương pháp | RMSE | P@10 | NDCG@10 |
|---|---:|---:|---:|
| Popularity | 0,8828 | 0,0044 | 0,0143 |
| CF Only (SVD++) | 0,8817 | 0,0044 | 0,0126 |
| **Content Only** | **0,8748** | 0,0044 | **0,0212** |
| Demographic Only (fallback unknown) | 0,8959 | 0,0033 | 0,0098 |
| Hybrid α=0.3 cố định | 0,8780 | 0,0044 | 0,0146 |
| Hybrid α=0.5 cố định | 0,8784 | 0,0044 | 0,0134 |
| Hybrid α=0.7 cố định | 0,8793 | 0,0049 | 0,0148 |
| StayReco α thủ công | 0,8778 | 0,0033 | 0,0135 |
| Tarek 2023 | 0,8762 | **0,0060** | 0,0188 |
| Do & Le 2020, 2 thành phần | 0,8750 | 0,0044 | 0,0171 |
| Do & Le 2020, mở rộng 3 thành phần | 0,8801 | 0,0038 | 0,0114 |
| OHWSF 2022 | 0,8788 | 0,0044 | 0,0169 |
| SL-ARec 2026 | 0,8867 | 0,0038 | 0,0152 |
| StayReco α tuned | 0,8789 | 0,0049 | 0,0157 |

StayReco tuned chọn trên validation: `alpha_max=0.9`, `n_threshold=50`,
`gamma_base=0.15`, `alpha_min=0.3`. Giữ nguyên grid search và cấu hình mô hình
hiện có: SVD++ 50 factors/30 epochs; LightGCN 32 factors/3 layers/300 epochs.

## Giới hạn khi diễn giải

- Đây là một triệu dòng **đầu file**, xếp theo hotel, rồi lọc theo hoạt động.
  Mẫu có thiên lệch; không được mô tả là đã chạy toàn bộ 50 triệu review.
- Một seed và chỉ 300 rating test; chưa có kiểm định độ ổn định qua nhiều seed.
  Test có 180 cold-item, 112 cold-user và 8 warm rating; không có rating nhóm cool.
  Xếp hạng có 178 cold user, 3 cool user và chỉ 1 warm user.
- User ID lấy từ `author` thật. Không có dữ liệu nhân khẩu nên toàn bộ user
  thuộc nhóm `unknown`; dòng Demographic chỉ là fallback, không kiểm chứng
  được giá trị của nhân khẩu học trong hybrid.
- Content chỉ dùng tên và địa danh trong URL hotel. Không dùng text/title,
  rating hay sub-rating của review làm đặc trưng, tránh rò rỉ validation/test.
  Đặc trưng này yếu hơn mô tả và tiện nghi khách sạn thực tế.
- `New@10` của StayReco tuned bằng 0: lần chạy này chưa cho thấy mô hình đưa
  hotel cold-start vào top 10.
- Không tải đầy đủ ZIP. File tải dở `datasets_raw/hotelrec/Full_HotelRec.zip.part`
  được giữ lại; các file slim chỉ khoảng 10 MB. Chưa xác minh CRC/checksum toàn
  ZIP; SHA256 của các dòng đầu và file slim nằm trong manifest.
- Working tree có thay đổi chưa commit trước và trong lần chạy. Manifest ghi
  `dirty=true`; không thể tái lập chỉ từ commit hash.

## Chạy lại và đối chiếu

```powershell
conda activate stayreco
python -m pytest tests/ -q
python run_experiment.py 42 --dataset=hotelrec --cold-item=0.2 --tag=hotelrec_subset_1m
```

Tập đã chuẩn bị nằm trong `datasets_raw/hotelrec/`. Khi dựng lại ở máy khác:

```powershell
python prepare_hotelrec.py --max-reviews=1000000
```

Đối chiếu SHA256 của các dòng nguồn:
`bda2642f9377b02098f4441dbc51db715fba9f10ea4d46cb32b1c7e12a5a7922`.

- [Metrics đầy đủ](comparison_hotelrec_subset_1m.csv)
- [Metrics theo nhóm](comparison_by_group_hotelrec_subset_1m.csv)
- [Cấu hình và nguồn dữ liệu](run_meta_hotelrec_subset_1m.json)
- [Tham số chọn trên validation](chosen_params_hotelrec_subset_1m.csv)
- [Biểu đồ theo nhóm](comparison_hotelrec_subset_1m.png)
- [Độ nhạy của α](sensitivity_alpha_hotelrec_subset_1m.png)
- [Log lần chạy](hotelrec_subset_1m.log)
- [Log kiểm tra và dựng lại biểu đồ](tests_hotelrec.log)

Đã kiểm tra: 170 test pass; metrics hữu hạn và nằm trong miền hợp lệ; manifest
ghi đúng dataset và số dòng. Đã sửa biểu đồ để xử lý nhóm cool thiếu rating
test và hiển thị đủ 5 giá trị alpha_max trong biểu đồ độ nhạy. Không thay đổi
thuật toán mô hình hay grid search để cải thiện kết quả.
