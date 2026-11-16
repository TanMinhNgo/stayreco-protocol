# StayReco Recommender — Thực nghiệm

Capstone FA26SE056 (FPT University HCMC). Chứng minh **α động tốt hơn α cố định**
cho bài toán cold-start hai phía của nền tảng đặt phòng mới.

```
Knowledge pre-filter → score = α(n)·CF + β(n)·Content + γ(n)·Demographic → Top-K
```

## Cài đặt

Thí nghiệm chạy trong conda env `stayreco`:

```bash
conda env create -f environment.yml      # lần đầu
conda activate stayreco                  # ⚠️ BẮT BUỘC, xem ghi chú dưới
pytest tests/ -q                          # kiểm tra trước khi chạy
```

> ⚠️ **Phải `conda activate` trước khi chạy.** Gọi thẳng
> `C:\...\envs\stayreco\python.exe` sẽ làm numpy crash ở phép nhân ma trận dày
> (`a @ b`), vì các DLL của MKL nằm trong `Library/bin` và chỉ được đưa vào
> đường dẫn tìm DLL khi env được activate. Lỗi hiện ra dưới dạng tiến trình chết
> không traceback (Windows exception `0xc06d007f`). Nếu cần chạy không activate,
> dùng `conda run -n stayreco python ...`.

## Cách chạy

```bash
# Thí nghiệm chính: 10 seed, 20% hotel bị giấu toàn bộ rating
python run_experiment.py 42 1 2 3 4 5 6 7 8 9 --cold-item=0.2 --tag=fullcat

# So cặp từng seed + kiểm định Wilcoxon
python analyze_seeds.py _fullcat              # trên RMSE
python analyze_seeds.py _fullcat ndcg@10      # trên NDCG@10

# SVD++ có thật sự hơn SVD không (không cần train lại)
python compare_svd_svdpp.py _fullcat
```

Tham số dòng lệnh:

| Cờ | Ý nghĩa |
|---|---|
| `42 1 2 ...` | danh sách seed; mỗi seed là một cách chia dữ liệu |
| `--cold-item=0.2` | % hotel bị giấu toàn bộ rating (Schein et al. 2002 dùng ~0.2) |
| `--tag=fullcat` | hậu tố file kết quả, để không ghi đè lần chạy trước |
| `--dataset=hotelrec` | tập con HotelRec đã chuẩn bị, có ID người dùng thật |

### Chạy tập con HotelRec

**Lưu ý audit:** kết quả cũ mang tag `hotelrec_subset_1m` có user placeholder
`/undefined`, nên không dùng làm bằng chứng thực nghiệm. Loader hiện loại
placeholder này kể cả khi đọc cache cũ. Ưu tiên mẫu toàn archive bên dưới.

Nguồn: [HotelRec, Antognini & Faltings (LREC 2020)](https://github.com/diegoantognini/HotelRec),
chỉ dùng cho nghiên cứu học thuật. ZIP gốc khoảng 14,4 GB, giải nén khoảng 50,4 GB.
Máy 16 GB RAM dùng tập con; pipeline hiện tại không phù hợp để chạy cả 50 triệu review.

```bash
conda activate stayreco
# Đọc luồng ZIP từ nguồn, dừng sau 1 triệu review; không tải/giải nén toàn bộ.
python prepare_hotelrec.py --max-reviews=1000000
# Hoặc thêm --archive=datasets_raw/hotelrec/Full_HotelRec.zip.part nếu đã tải prefix.
pytest tests/ -q
python run_experiment.py 42 --dataset=hotelrec --cold-item=0.2 --tag=hotelrec_subset_1m
```

Loader lấy tối đa 1.000 hotel nhiều review nhất, rồi tối đa 1.000 user hoạt động
nhiều nhất (đồng hạng sắp theo ID), giữ rating mới nhất cho mỗi cặp user–hotel,
lọc lặp đến khi mỗi user có ít nhất 3 hotel và mỗi hotel có ít nhất 5 user.
Xếp hạng trên toàn catalog **của tập con này**. Checksum, phạm vi mẫu, số dòng
thật và cấu hình được ghi vào `source.json` và `results/run_meta_<tag>.json`.

Giới hạn cần ghi trong báo cáo:

- Một triệu dòng đầu được xếp theo hotel; cộng thêm lọc theo hoạt động nên mẫu
  có thiên lệch, không đại diện toàn bộ HotelRec. Không gọi đây là kết quả trên 50M review.
- Content chỉ dùng tên/địa danh trong URL hotel, không dùng review text, rating
  hay sub-rating của validation/test làm đặc trưng.
- Bản nguồn không có hồ sơ nhân khẩu. Khi chưa bổ sung profile ở bước bên dưới,
  `Demographic` chỉ dùng fallback theo hotel; dữ liệu thiếu không tạo thành một cohort.
- Dừng đọc sớm nên chưa xác minh CRC/checksum toàn ZIP; có SHA256 cho các dòng
  đã đọc và các file đã chuẩn bị. `prepare_hotelrec.py` từ chối ghi đè tập đã có.

### Mẫu HotelRec từ toàn bộ archive

`prepare_hotelrec_full.py` quét hết ZIP, xác minh MD5 từ nguồn và CRC của
`HotelRec.txt`, lưu dữ liệu vào SQLite thay vì đưa 50 triệu dòng lên RAM.
Hai index trên đĩa phục vụ đếm cặp user–hotel khác nhau và truy xuất lịch sử.
Giữ nguyên tập prefix ở trên; tập mới nằm ở `datasets_raw/hotelrec_full_sample/`.
Nếu môi trường đã có `orjson`, trình quét dùng nó để tăng tốc và ghi phiên bản
vào metadata; nếu không, dùng JSON chuẩn. Không cần cài thêm dependency.

```powershell
# Nếu chưa có ZIP đầy đủ, tải tiếp từ nguồn chính thức.
New-Item -ItemType Directory -Force datasets_raw/hotelrec | Out-Null
curl.exe -fL -C - --retry 3 -o datasets_raw/hotelrec/Full_HotelRec.zip https://drive.switch.ch/public.php/dav/files/n48smsdufhRA7fR
python prepare_hotelrec_full.py

conda activate stayreco
python -c "from prepare_hotelrec_full import profile_sample; profile_sample('datasets_raw/hotelrec_full_sample', 'results/hotelrec_full_sample_profile.json')"
# Chạy mô hình sau khi đọc thống kê độ phủ trong profile.
python run_experiment.py 42 --dataset=hotelrec-full-sample --cold-item=0.2 --tag=hotelrec_full_sample
```

Mặc định chọn 2.000 hotel có nhiều user khác nhau nhất trên toàn bộ archive,
rồi 1.000 user có nhiều hotel khác nhau nhất trong catalog đó. Đồng hạng sắp
theo ID dạng chuỗi. Export giữ các review của những cặp được chọn; loader giữ
rating mới nhất, rồi lọc lặp user>=3 và hotel>=5 như tập prefix. Đây là mẫu thiên
về hoạt động để **mô phỏng cold-start**, không phải mẫu ngẫu nhiên của dân số.

Profile kiểm tra 10 seed chia dữ liệu mà không huấn luyện model: số rating theo
nhóm, user đủ điều kiện xếp hạng theo cold/cool/warm và số cặp ứng viên dự kiến.
Công thức α, grid search và thuật toán mô hình được giữ nguyên. Các giới hạn về
Content và Demographic của HotelRec vẫn áp dụng.

Loại các ID thiếu như `/undefined` trước khi đếm mức hoạt động. SQLite cache
được tạo bởi phiên bản cũ sẽ được làm sạch trong transaction và cập nhật lại
thống kê; ZIP gốc được giữ nguyên. Loader dùng giới hạn catalog ghi trong
`sampling_limits` của bộ dữ liệu đã chuẩn bị.

SQLite đã hoàn thành được tái sử dụng khi checksum archive khớp. Index chưa
hoàn thành hoặc output đã có sẽ báo lỗi để tránh trộn dữ liệu/ghi đè kết quả.

### Bổ sung demographic địa lý cho HotelRec

Dùng `authors.json.gz` của bộ TripAdvisor US đã có trong `datasets_raw/`, nối
`HotelRec.author = profile.username` chính xác. Trong mẫu 1.000 user hiện tại,
840 user khớp profile và 693 có hometown tự khai, tạo 82 nhóm vùng cư trú.

```powershell
python prepare_hotelrec_demographics.py
conda activate stayreco
python -m pytest tests/ -q
python run_experiment.py 42 --dataset=hotelrec-full-sample --cold-item=0.2 --tag=hotelrec_geo_seed42
```

Script tạo `users_demographics.csv.gz` và `demographics_source.json` cạnh dữ
liệu mẫu. Loader kiểm tra checksum cả file profile và tập rating đích, kiểm tra
ID không trùng, rồi đưa `home_region` vào cột `demographic` của model hiện có.
Các file cũ được giữ; script từ chối ghi đè enrichment đã tồn tại.

`home_region` là phần cuối sau dấu phẩy trong hometown, chuẩn hóa khoảng trắng
và chữ hoa/thường. Nó có thể là bang, quốc gia hoặc thành phố; không diễn giải
thành quốc tịch. Username khớp nhiều profile ID hoặc hometown xung đột bị loại.
307 user thiếu vùng cư trú dùng fallback item/global; model không học một nhóm
nhân khẩu chung cho các user thiếu thông tin.

Không có tuổi/giới tính trong nguồn profile này. Hồ sơ thuộc snapshot riêng,
không có ngày quan sát từng profile, nên không bảo đảm các thuộc tính đã có
tại thời điểm review HotelRec trong quá khứ. Ghi giới hạn này khi báo cáo kết quả.
Xem [báo cáo demographic](results/hotelrec_demographic_report.md).

## Hai cách đo — đọc kỹ chỗ này

Thí nghiệm báo cáo **hai nhóm metric riêng biệt**, vì chúng trả lời hai câu khác nhau:

| Nhóm | Metric | Câu hỏi | Rổ ứng viên |
|---|---|---|---|
| Sai số dự đoán | RMSE, MAE | đoán số sao chính xác đến đâu? | từng cặp (user, hotel) độc lập |
| Chất lượng xếp hạng | P@K, R@K, NDCG@K, Cov@K, New@K | chọn K hotel để hiển thị có đúng không? | **toàn bộ catalog** user chưa đặt |

Nhóm thứ hai **phải** xếp hạng trên toàn catalog. Cách làm sai phổ biến là xếp
hạng chỉ trong số item user đã rating trong test set — rổ đó chỉ ~20 item và ~55%
trong đó là item user thích, nên đoán bừa đã được NDCG@10 ≈ 0.84 và mọi phương
pháp dồn vào dải 0.88–0.89. Tệ hơn: hotel mới chưa ai rating không bao giờ có
trong rổ, nên cold-start phía hotel không được đo gì cả.

Cơ sở: Krichene & Rendle, *On Sampled Metrics for Item Recommendation* (KDD 2020).
Chi tiết và số liệu đối chứng: xem đầu file [`evaluation/metrics.py`](evaluation/metrics.py)
và [`evaluation/ranking.py`](evaluation/ranking.py).

`New@K` = tỷ lệ ô hiển thị dành cho hotel chưa ai rating. **Không phải càng cao
càng tốt** — đọc cùng P@K: tốt = giữ được Precision mà vẫn chia chỗ cho hotel mới.

## Các phương pháp được so sánh

Nhóm A — cùng 3 model thành phần, chỉ khác cách trộn:
Popularity · CF only (SVD++) · Content only · Demographic only ·
Hybrid α=0.3/0.5/0.7 cố định · **StayReco α động** (bản thủ công và bản tune)

Nhóm B — dựng lại 4 bài báo trong cùng khung thực nghiệm:
Tarek 2023 · Do & Le 2020 (2 và 3 thành phần) · OHWSF 2022 · SL-ARec 2026

> Đây là **bản thích nghi**, không phải bản sao nguyên gốc. Mỗi bài dùng dataset,
> cách chia và metric riêng, nên copy số trong bài báo rồi so với số của mình là
> sai. Chỗ nào phải sửa để chạy được đều ghi trong docstring
> [`models/paper_baselines.py`](models/paper_baselines.py).

Mọi phương pháp đều được grid search trên **validation** theo RMSE — như nhau cho
tất cả. Tune trên test là gian lận số liệu (Dacrema et al., RecSys 2019).

## Cấu trúc

```
stayreco-experiment/
├── config.py               # hằng số; giá trị THẬT của mỗi lần chạy ở run_meta_<tag>.json
├── run_experiment.py       # chạy toàn bộ thí nghiệm
├── analyze_seeds.py        # so cặp từng seed + Wilcoxon signed-rank
├── compare_svd_svdpp.py    # SVD++ vs SVD, tách theo nhóm cold-start
├── data/
│   ├── loader.py           # MovieLens; hook sẵn cho HotelRec/Expedia
│   └── preprocessor.py     # split train/val/test + mô phỏng cold-start
├── models/
│   ├── cf_svdpp.py         # SVD++ và SVD (Surprise)
│   ├── cf_itemknn.py       # Item-Item CF (cho Tarek 2023)
│   ├── lightgcn.py         # LightGCN (cho SL-ARec 2026)
│   ├── content_based.py    # TF-IDF + cosine
│   ├── demographic.py      # trung bình theo nhóm nhân khẩu
│   ├── popularity.py       # mốc tham chiếu thấp nhất
│   ├── knowledge.py        # pre-filter business rules
│   ├── hybrid.py           # trộn 3 thành phần
│   └── paper_baselines.py  # dựng lại 4 bài báo
├── evaluation/
│   ├── metrics.py          # RMSE, MAE (+ metric cũ, đã đánh dấu cảnh báo)
│   └── ranking.py          # P/R/NDCG/Cov@K trên TOÀN CATALOG  ← dùng cái này
├── utils/
│   ├── alpha_scheduler.py  # công thức α động (đóng góp chính)
│   ├── tuning.py           # grid search trên validation
│   ├── batch.py            # chấm điểm vectorized
│   └── manifest.py         # ghi run_meta_<tag>.json
├── tests/                  # pytest — chạy trước mỗi lần chạy thí nghiệm
└── results/                # CSV, PNG, run_meta_<tag>.json
```

## Output trong `results/`

| File | Nội dung |
|---|---|
| `run_meta_<tag>.json` | **truy xuất nguồn gốc**: commit, config thật, phiên bản thư viện |
| `comparison_<tag>.csv` | metrics tổng thể, mỗi dòng một (phương pháp, seed) |
| `comparison_by_group_<tag>.csv` | metrics theo nhóm cold-start |
| `chosen_params_<tag>.csv` | tham số grid search đã chọn |
| `score_table_seed*_<tag>.csv` | điểm từng model cho từng cặp test (phân tích lại không cần train) |
| `comparison_<tag>.png` | biểu đồ theo nhóm |
| `sensitivity_alpha_<tag>.png` | RMSE validation thay đổi theo α_max |

## Trạng thái

Đã có: MovieLens 100K, 14 phương pháp, 10 seed, xếp hạng toàn catalog, kiểm định
Wilcoxon, test cho vectorization / metric / công thức α.

Có thêm: loader và lệnh chạy tập con HotelRec có ID người dùng thật (xem giới hạn ở trên).

Còn thiếu: chạy HotelRec đầy đủ, Expedia;
implicit feedback thật (view/click) để biện minh cho việc chọn SVD++;
Knowledge pre-filter chưa vào vòng đánh giá (MovieLens không có budget/phòng trống).

## Giao thức của bài (HotelRec)

```bash
python prepare_protocol_sample.py   # chọn mẫu chỉ từ thông tin trước mốc cắt
python prepare_hotelrec_demographics.py --folder datasets_raw/hotelrec_protocol_sample_v2
python run_protocol.py --sample datasets_raw/hotelrec_protocol_sample_v2 --tag v2_s42 --seed 42
```

`run_protocol.py` ghi `results/protocol_{ndcg,recall,params,stats,tests}_<tag>.csv`. Mỗi review là một
booking, n_u đếm số review; vùng hotel suy ra từ chuỗi địa điểm (`--region-min-count`, mặc định 30 để
tập ứng viên đủ lớn so với K=10). Chi tiết: `docs/superpowers/specs/2026-11-16-stayreco-protocol-design.md`.
