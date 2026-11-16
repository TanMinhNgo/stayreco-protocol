# Thiết kế: giao thức đánh giá của bài StayReco (Hướng 2, dữ liệu HotelRec)

Ngày: 2026-11-16. Trạng thái: **chờ duyệt**.

## Ý định và lý do

Bài *Interaction-Dependent Hybrid Weighting for Cold-Start Hotel Recommendation* (`main_ceur.tex`) hiện là bài thiết kế + giao thức, Phần 5 (Results) còn trống. Mục tiêu: chạy đúng giao thức ở mục 5 của bài trên dữ liệu thật để điền Phần 5 và checklist Phần 5 bằng số liệu thật, có code chạy lại được. Không điền số giả định.

## Mục tiêu

- Chạy giao thức của bài trên HotelRec: chia train/val/test theo thời gian; NDCG@10 và Recall@10; nhóm n_u = 0, 1–4, 5–19, ≥ 20 và nhóm khách sạn mới; baseline MostPop-City, CF only, Content only, Demographic only, α cố định (tune trên validation), α động `n/(n+k)`.
- Chọn k, k_c trên validation (lưới k ∈ {5,10,25,50}, k_c ∈ {1,2,3,5,10}); mở rộng lưới lên trên nếu tối ưu nằm ở biên.
- Ước lượng k̂ bằng moment (mục 4.6 bước 1), n\* (mức n_u mà CF only vượt Content only), kiểm định paired t-test và paired bootstrap, kiểm tra H1–H3.
- Biến thể tùy chọn của bài: π_g theo sai số (Eq. bg), RRF có trọng số (Eq. rrf), FWLS logistic (Eq. fwls). Làm sau khi lõi chạy ổn.

## Không phải mục tiêu / chống mục tiêu

- Không đánh giá Stage 1 (phòng trống, ngân sách, relaxation R0–R3): HotelRec không có dữ liệu này. Ghi vào hạn chế.
- Không sửa pipeline cũ (`run_experiment.py`, đo rating/RMSE). Không làm iALS.
- Chống mục tiêu: kết quả trông đẹp nhờ rò rỉ dữ liệu (dùng thông tin sau mốc cắt thời gian), hoặc điền số không do code sinh ra.

## Ràng buộc và điều chỉnh vì dữ liệu (đã được người dùng chấp nhận)

- HotelRec chỉ có review (user, hotel, rating, ngày). Mỗi review là một "booking" (đối tượng liên quan). **Mỗi review tính n_u += 1** (không dùng trọng số 1/2/3/5, vì sẽ làm nhóm 1–4 rỗng). Ghi rõ là điều chỉnh vì dữ liệu.
- Không có ngữ cảnh tìm kiếm: dùng phương án dự phòng của bài, xếp hạng mọi hotel "đã duyệt" trong cùng vùng với hotel được đặt.
- Nhân khẩu học chỉ có `home_region` cho một phần user (ghép với tác giả TripAdvisor Mỹ, ~69% trong mẫu 1.000 user). Demographic yếu; user không có thuộc tính dùng nhóm "unknown".
- HotelRec dùng cho nghiên cứu học thuật. Dữ liệu chỉ có mô tả tên + địa điểm từ URL, không có text review.

## Cách tiếp cận đã cân nhắc

| | Mô tả | Ưu | Nhược |
|---|---|---|---|
| **A (chọn)** | Gói mới `protocol/`, script `run_protocol.py`, tái dùng `ContentBasedFilter`, `DemographicFilter`, `ItemItemCF` | Không đụng pipeline cũ | Thêm code mới |
| B | Thêm chế độ xếp hạng vào `run_experiment.py` | Ít file | Làm rối pipeline xoay quanh rating/RMSE |

Quy mô dữ liệu: mẫu lớn (chọn, các lệnh của mình) so với giữ mẫu 1.000 user (quá nhỏ cho kiểm định theo nhóm). CF: BPR bằng `torch`, giữ SVD++ làm baseline; bỏ iALS (bài cho phép chọn một trong hai).

## Các quyết định (calls của người thiết kế, chưa phải ý người dùng)

1. **Mẫu dữ liệu:** xuất mẫu lớn từ `datasets_raw/hotelrec/full.sqlite` bằng `prepare_hotelrec_full.py` (`--max-users`, `--max-items`), mục tiêu khoảng 20–50 nghìn user và vài nghìn hotel; đủ để mỗi nhóm n_u có hàng trăm booking test (ngưỡng kinh nghiệm của checklist 4.2). Chọn tham số cuối cùng sau khi đếm thực tế; **không** hạ ngưỡng để cho nhanh.
2. **Vùng ứng viên:** bảng `hotels` chỉ có chuỗi "tên + địa điểm", không có mã thành phố. Vùng = phần địa điểm suy ra từ chuỗi mô tả (thành phố/bang). Nếu suy ra không đáng tin, dùng bang; ghi cách suy ra trong bài. Không quét lại 14 GB để lấy mã geo trừ khi cần.
3. **Mốc thời gian:** hai mốc cắt toàn cục (train / validation / test) chọn sao cho test đủ lớn; ghi ngày cắt vào bài. Mọi n_u, trạng thái hotel mới tính từ dữ liệu trước mốc của giai đoạn đang đánh giá.
4. **Hotel mới:** hotel không có tương tác trong train.
5. **Điểm:** mỗi nguồn chuẩn hóa z-score theo từng yêu cầu trên tập ứng viên (Eq. zscore), α/β/γ theo Eq. alpha–betagamma. π: user không khai báo sở thích → `n/(n+k_c)`.
6. **CF:** BPR (torch) + SVD++ (đã có) làm baseline; chọn trên NDCG@10 validation. Hotel không có biểu diễn CF → điểm CF chuẩn hóa = 0.
7. **Kiểm định:** paired t-test và paired bootstrap trên NDCG@10 từng booking, mức 0,05, 10.000 lần bootstrap (khớp mặc định checklist, cần người dùng xác nhận).
8. **Khả năng tái lập:** seed cố định, manifest chạy (commit, phiên bản thư viện) như pipeline cũ.

## Đầu ra

- Bảng thống kê dữ liệu thật (user, hotel, sự kiện, booking, khoảng thời gian, mốc cắt).
- Bảng NDCG@10 và Recall@10 theo nhóm n_u, hotel mới, toàn bộ.
- k, k_c đã chọn; k̂; n\*; kết quả kiểm định (p-value).
- File CSV kết quả trong `results/` và đoạn số liệu để dán vào Phần 5 của bài; cập nhật checklist Phần 5 chỉ bằng số do code sinh ra.

## Việc để người xây dựng quyết

- Cấu trúc nội bộ của `protocol/` và test (nên có test nhỏ cho: chia theo thời gian không rò rỉ, n_u, chuẩn hóa z-score với σ = 0, nhóm n_u).
- Cách suy ra vùng từ chuỗi mô tả (xem quyết định 2) sau khi nhìn dữ liệu thật.
- Cách giới hạn bộ nhớ khi xếp hạng (không dựng ma trận user × toàn catalog; chỉ ứng viên trong vùng).
