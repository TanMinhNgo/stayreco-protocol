# StayReco Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chạy giao thức đánh giá của bài StayReco (NDCG@10/Recall@10 theo nhóm n_u, chọn k/k_c, k̂, n\*, kiểm định) trên một mẫu lớn của HotelRec và sinh số liệu thật cho Phần 5.

**Architecture:** Gói mới `protocol/` độc lập với pipeline rating cũ. Dữ liệu → chia theo thời gian → mỗi nguồn (CF/Content/Demographic) cho điểm trên tập ứng viên cùng vùng → z-score theo yêu cầu → trộn bằng α động hoặc cố định → NDCG/Recall từng booking → kiểm định. `run_protocol.py` ghép các bước và ghi CSV vào `results/`.

**Tech Stack:** Python 3.11 (env conda `stayreco`), pandas, numpy, scipy, torch (BPR), models có sẵn (`ContentBasedFilter`, `DemographicFilter`, `CollaborativeFilter`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-stayreco-protocol-design.md` (và `C:\Users\Admin\Downloads\main_ceur.tex`, mục 4–5).

## Global Constraints

- Chạy trong env: `C:\Users\Admin\miniconda3\Scripts\conda.exe run -n stayreco ...` (cần `conda activate`, xem README).
- Mỗi review = một booking; **n_u += 1 cho mỗi review** (không dùng trọng số 1/2/3/5).
- Nhóm n_u: `0`, `1–4`, `5–19`, `≥20`; thêm nhóm "hotel mới" (không có tương tác trong train).
- Metric: NDCG@10, Recall@10 (K=10). Không dùng RMSE.
- Lưới: `k ∈ {5,10,25,50}`, `k_c ∈ {1,2,3,5,10}`; α cố định `∈ {0,0.3,0.5,0.7,1.0}`; mở rộng lưới k lên trên nếu tối ưu nằm ở biên.
- Kiểm định: paired t-test + paired bootstrap, mức 0,05, 10.000 lần bootstrap.
- Z-score: trung bình và độ lệch chuẩn quần thể (ddof=0) trên tập ứng viên; `σ=0` → mọi điểm chuẩn hóa = 0.
- CF của hotel không có biểu diễn trong train: điểm CF chuẩn hóa = 0.
- Mọi tham số chọn trên validation; test chỉ đánh giá một lần.
- Không điền số không do code sinh ra. Không sửa `run_experiment.py`, không làm iALS.
- Không commit nếu người dùng chưa yêu cầu (bỏ qua các bước Commit nếu đang không được phép; commit message tiếng Anh, Conventional Commits).

## Review Focus

- User chỉ có trong test, không có trong train → nhóm n_u=0, không crash, Content/Demographic vẫn cho điểm.
- Hotel được đặt không nằm trong tập ứng viên của vùng → bỏ booking đó và đếm số bị bỏ (không âm thầm mất).
- Tất cả ứng viên cùng điểm (σ=0) → xếp hạng ổn định, không NaN.
- Hotel đã có trong train của chính user không được xuất hiện lại trong ứng viên.
- Chuỗi địa điểm hotel không suy ra được vùng → hotel đó bị loại khỏi mọi tập ứng viên và được đếm.
- Mốc cắt thời gian: không dòng train nào có `timestamp` ≥ mốc; n_u của giai đoạn test chỉ tính từ train+validation.

---

### Task 1: Dữ liệu — mẫu, vùng, chia thời gian, n_u

**Files:**
- Create: `protocol/__init__.py`, `protocol/data.py`
- Test: `tests/test_protocol_data.py`

**Interfaces:**
- Produces:
  - `region_of(description: str) -> str | None` — suy ra vùng từ chuỗi "tên + địa điểm" (ví dụ `"Hotel Finial Anniston Alabama"` → `"alabama"`); `None` nếu không suy ra được. Quyết định 2 của spec: dùng bang (từ cuối chuỗi khớp danh sách bang Mỹ/quốc gia thường gặp); ghi cách làm trong docstring.
  - `load_sample(path: str) -> tuple[pd.DataFrame, pd.DataFrame]` — `(reviews[user_id,item_id,rating,timestamp], hotels[item_id,description,region])`; đọc `reviews_slim.csv.gz` và `hotels.csv.gz` của thư mục mẫu; `item_id` kiểu str.
  - `temporal_split(reviews, t_val: int, t_test: int) -> tuple[DataFrame, DataFrame, DataFrame]` — `(train, val, test)` với `train.timestamp < t_val <= val.timestamp < t_test <= test.timestamp`.
  - `interaction_counts(history: DataFrame) -> pd.Series` — n_u theo `user_id` (một review = 1).
  - `user_group(n: int) -> str` trả `"0"`, `"1-4"`, `"5-19"`, `">=20"`.

- [ ] **Step 1: Viết test lỗi** `tests/test_protocol_data.py`: `test_region_of_extracts_state` (`"Hotel Finial Anniston Alabama"` → `"alabama"`), `test_region_of_unknown_returns_none`, `test_temporal_split_no_leak` (assert `train.timestamp.max() < t_val <= val.timestamp.min()` và tương tự val/test, tổng dòng giữ nguyên), `test_interaction_counts_one_per_review`, `test_user_group_boundaries` (0→"0", 1→"1-4", 4→"1-4", 5→"5-19", 19→"5-19", 20→">=20").
- [ ] **Step 2: Chạy** `pytest tests/test_protocol_data.py -v` → FAIL (chưa có module).
- [ ] **Step 3: Cài `protocol/data.py`** theo chữ ký trên. `region_of`: lấy các token cuối của chuỗi, so với tập tên bang/quốc gia nhỏ lấy từ chính cột mô tả của mẫu (xây từ tần suất token cuối; chỉ giữ token xuất hiện ở ≥ 5 hotel). `temporal_split` dùng lọc theo `timestamp`.
- [ ] **Step 4: Chạy lại** → PASS.
- [ ] **Step 5: Commit** `feat(protocol): add data loading, regions and temporal split`.

### Task 2: Chuẩn hóa và trộn điểm

**Files:**
- Create: `protocol/scoring.py`
- Test: `tests/test_protocol_scoring.py`

**Interfaces:**
- Produces:
  - `zscore(scores: np.ndarray) -> np.ndarray` — ddof=0; `σ==0` → mảng 0.
  - `dynamic_weights(n: float, k: float, k_c: float, declared: bool = False) -> tuple[float, float, float]` — `(α, β, γ)` theo Eq. alpha/pi/betagamma của bài; `α+β+γ=1`.
  - `fixed_weights(alpha: float, n: float, k_c: float, declared: bool = False) -> tuple[float, float, float]` — chỉ khác α, phần `1−α` chia theo cùng π.
  - `fuse(z_cf, z_c, z_d, weights) -> np.ndarray` — `α·z_cf + β·z_c + γ·z_d` (Eq. final).

- [ ] **Step 1: Viết test lỗi:** `test_zscore_constant_is_zero`; `test_dynamic_weights_match_paper_table` với `k=10, k_c=3`: `n=0` không khai báo → `(0,0,1)`; `n=0` khai báo → `(0,1,0)`; `n=10` không khai báo → `(0.5, 0.5·10/13, 0.5·3/13)` (≈ 0.50/0.38/0.12, khớp Bảng weights của bài); `n=30` → α=0.75; `test_weights_sum_to_one` cho nhiều n; `test_fixed_weights_alpha_one_is_cf_only`; `test_fuse_reproduces_paper_example` dùng 4 hotel của Bảng example (A..D, α=β=0.5): thứ hạng chuẩn hóa là B>A>D>C.
- [ ] **Step 2: Chạy** → FAIL.
- [ ] **Step 3: Cài `protocol/scoring.py`** theo chữ ký.
- [ ] **Step 4: Chạy** → PASS.
- [ ] **Step 5: Commit** `feat(protocol): add score standardization and weighting`.

### Task 3: Nguồn điểm — BPR, Content, Demographic, MostPop-City

**Files:**
- Create: `protocol/sources.py`
- Test: `tests/test_protocol_sources.py`

**Interfaces:**
- Consumes: `ContentBasedFilter(k_neighbors).fit(items_df, ratings_df)` / `.predict_many(users, items)`; `DemographicFilter(group_cols, shrinkage).fit(ratings_df, demo_df)` / `.predict_many`; `CollaborativeFilter` (SVD++, baseline).
- Produces:
  - `class BPR` với `fit(train: DataFrame, seed: int, n_factors=32, n_epochs=30, lr=0.01, reg=1e-4)` và `score(users, items) -> np.ndarray` (hotel/user chưa thấy trong train → `nan`).
  - `class MostPopCity` với `fit(train, hotels)` và `score(users, items)` = số booking train của hotel (dùng để xếp hạng trong vùng).
  - `build_sources(train, hotels, users_demo, seed) -> dict[str, object]` trả `{"cf": BPR, "svdpp": ..., "content": ..., "demo": ..., "pop": ...}` mỗi object có `score(users, items)`.

- [ ] **Step 1: Viết test lỗi:** `test_bpr_ranks_seen_pattern_above_random` (dữ liệu tí hon: hai nhóm user thích hai nhóm hotel; sau fit, điểm hotel đúng nhóm > hotel sai nhóm với cùng seed); `test_bpr_unseen_item_is_nan`; `test_mostpop_orders_by_train_count`; `test_sources_score_shape_matches_input`.
- [ ] **Step 2: Chạy** → FAIL.
- [ ] **Step 3: Cài `protocol/sources.py`.** BPR: embedding user/item bằng torch, mất mát BPR với một mẫu âm ngẫu nhiên mỗi cặp dương mỗi epoch, `torch.manual_seed(seed)`. Bọc các model có sẵn bằng `score = predict_many`.
- [ ] **Step 4: Chạy** → PASS.
- [ ] **Step 5: Commit** `feat(protocol): add BPR and component score sources`.

### Task 4: Đánh giá theo từng booking

**Files:**
- Create: `protocol/evaluate.py`
- Test: `tests/test_protocol_evaluate.py`

**Interfaces:**
- Consumes: `zscore`, `dynamic_weights`, `fixed_weights`, `fuse` (Task 2); đối tượng nguồn (Task 3); `user_group`, `interaction_counts` (Task 1).
- Produces:
  - `candidates(hotels, region: str, seen: set[str]) -> list[str]` — hotel có `region` giống và không nằm trong `seen`.
  - `ndcg_recall_at_k(ranked: list[str], booked: str, k=10) -> tuple[float, float]` — NDCG@10 và Recall@10 cho một booking; booked không nằm trong top-k → `(0.0, 0.0)`; rank 1 → NDCG 1, rank 3 → 0.5, rank 10 → ≈0.29 (khớp bài).
  - `evaluate_method(method, test, history, hotels, sources) -> pd.DataFrame` — mỗi dòng một booking: `user_id, item_id, group, is_new_hotel, ndcg, recall`; `method` là callable `(n_u, z_cf, z_c, z_d) -> scores`. Booking có hotel không trong tập ứng viên bị bỏ và số bị bỏ ghi vào `df.attrs["dropped"]`.

- [ ] **Step 1: Viết test lỗi:** `test_ndcg_known_ranks` (rank 1/3/10 → 1, 0.5, ≈0.2891); `test_recall_outside_topk_is_zero`; `test_candidates_exclude_seen_and_other_region`; `test_evaluate_drops_booking_outside_candidates` (kiểm `attrs["dropped"]`); `test_all_equal_scores_stable_no_nan`.
- [ ] **Step 2: Chạy** → FAIL.
- [ ] **Step 3: Cài `protocol/evaluate.py`.** Với mỗi booking: vùng của hotel được đặt → ứng viên → điểm từng nguồn (NaN → z-score tính trên giá trị hữu hạn, phần NaN gán 0 sau chuẩn hóa) → trộn → sắp xếp ổn định (`kind="stable"`).
- [ ] **Step 4: Chạy** → PASS.
- [ ] **Step 5: Commit** `feat(protocol): add per-booking NDCG and Recall evaluation`.

### Task 5: Thống kê — kiểm định, k̂, n\*

**Files:**
- Create: `protocol/stats.py`
- Test: `tests/test_protocol_stats.py`

**Interfaces:**
- Produces:
  - `paired_tests(a: np.ndarray, b: np.ndarray, n_boot=10_000, seed=0) -> dict` với `t_p`, `boot_p`, `mean_diff`.
  - `estimate_k(x_by_user: dict[str, np.ndarray], prior_by_user: dict[str, float], min_n=30) -> float | None` — moment estimate Eq. moments; `None` nếu `τ̂² ≤ 0`.
  - `crossover_n(ndcg_cf: pd.DataFrame, ndcg_c: pd.DataFrame, bins) -> int | None` — nhóm n_u thấp nhất mà CF only cao hơn Content only có ý nghĩa (p < 0,05); `None` nếu không có.

- [ ] **Step 1: Viết test lỗi:** `test_paired_tests_detect_shift` (a = b + 0.1 với nhiễu nhỏ → hai p < 0,05; a = b → p lớn, `mean_diff≈0`); `test_estimate_k_known_variances` (sinh dữ liệu với σ²=40, τ²=4 → k̂ gần 10 trong sai số 30%); `test_estimate_k_returns_none_when_tau_nonpositive`; `test_crossover_none_when_cf_never_wins`.
- [ ] **Step 2: Chạy** → FAIL.
- [ ] **Step 3: Cài `protocol/stats.py`** bằng `scipy.stats.ttest_rel` và bootstrap có seed cố định.
- [ ] **Step 4: Chạy** → PASS.
- [ ] **Step 5: Commit** `feat(protocol): add significance tests, k estimate and crossover`.

### Task 6: Script chạy và ghi kết quả

**Files:**
- Create: `run_protocol.py`
- Test: `tests/test_protocol_smoke.py`

**Interfaces:**
- Consumes: tất cả các task trên.
- Produces: CLI `python run_protocol.py --sample PATH --t-val TS --t-test TS --seed 42 --tag NAME`; ghi `results/protocol_ndcg_<tag>.csv`, `results/protocol_recall_<tag>.csv` (hàng = phương pháp, cột = nhóm `0, 1-4, 5-19, >=20, new, all`), `results/protocol_params_<tag>.csv` (k, k_c, α cố định tốt nhất, k̂, n\*), `results/protocol_stats_<tag>.csv` (thống kê dữ liệu, số booking bị bỏ), `results/protocol_tests_<tag>.csv` (p-value từng cặp), và manifest chạy qua `utils/manifest.py` như `run_experiment.py`.

- [ ] **Step 1: Viết test lỗi** `test_protocol_smoke`: sinh dữ liệu tổng hợp nhỏ vào thư mục tạm, gọi hàm chính, assert 5 file kết quả xuất hiện, cột nhóm đủ 6, mọi giá trị NDCG nằm trong [0,1], và hai lần chạy cùng seed cho đúng cùng kết quả.
- [ ] **Step 2: Chạy** → FAIL.
- [ ] **Step 3: Cài `run_protocol.py`.** Luồng: load → split → build_sources trên train → duyệt lưới `(k, k_c)` và α cố định trên validation (NDCG@10 toàn bộ) → chọn → đánh giá một lần trên test (train+val làm lịch sử) → thống kê → ghi CSV. Hàm chính nhận tham số để test gọi được không qua CLI.
- [ ] **Step 4: Chạy** `pytest tests/ -q` → toàn bộ PASS, kể cả 166 test cũ.
- [ ] **Step 5: Commit** `feat(protocol): add run_protocol entry point`.

### Task 7: Chạy thật, thống kê dữ liệu, cập nhật tài liệu

**Files:**
- Modify: `StayReco_checklist_hoan_thanh_paper.md` (chỉ Phần 5), `README.md` (mục cách chạy giao thức)
- Create: mẫu dữ liệu mới trong `datasets_raw/hotelrec_protocol_sample/` (không commit, đã bị `.gitignore`)

**Interfaces:**
- Consumes: `prepare_hotelrec_full.py` (`--max-users`, `--max-items`, `--output`) và `run_protocol.py`.

- [ ] **Step 1:** Xuất mẫu lớn: `python prepare_hotelrec_full.py --max-users 30000 --max-items 5000 --output datasets_raw/hotelrec_protocol_sample` (con số khởi điểm; điều chỉnh theo bước 2).
- [ ] **Step 2:** Đếm booking test theo nhóm n_u. Nếu nhóm nào dưới vài trăm booking (ngưỡng checklist 4.2), tăng `--max-users` hoặc lùi mốc cắt; **không hạ ngưỡng**. Ghi mốc cắt và thống kê vào `protocol_stats_<tag>.csv`.
- [ ] **Step 3:** Chạy `run_protocol.py` với 1 seed trước, rồi 10 seed (`--seed` khác nhau) nếu thời gian chạy hợp lý; gom p-value.
- [ ] **Step 4:** Điền Phần 5 của checklist và README **chỉ** bằng số lấy từ CSV; ghi rõ giới hạn (review thay booking, n_u đếm số review, không có Stage 1, nhân khẩu học thưa).
- [ ] **Step 5: Commit** `docs: record protocol results and run instructions`.

### Task 8 (sau khi lõi chạy ổn): biến thể tùy chọn

**Files:** Modify `protocol/scoring.py`, `protocol/stats.py`; test thêm trong các file test cũ.

- [ ] Thêm `pi_group(e_c, e_d) -> float` (Eq. bg, ví dụ `e_c=0.9, e_d=0.6 → 0.4`), `rrf(scores_by_source, weights, kappa=60) -> np.ndarray` (Eq. rrf, nguồn không có thông tin bị bỏ khỏi tổng), và FWLS logistic (Eq. fwls) với dự phòng `τ̂²≤0` / trọng số âm như bài mô tả. Mỗi hàm có một test với giá trị số của bài. Chỉ làm khi người dùng đồng ý sau khi có kết quả lõi.

---

## Self-review

- **Phủ spec:** mục tiêu (Task 1–6), đầu ra (Task 6–7), biến thể (Task 8), hạn chế (Task 7 bước 4). Stage 1 và iALS nằm ngoài phạm vi theo spec.
- **Nhất quán chữ ký:** `zscore/dynamic_weights/fixed_weights/fuse` (Task 2) được dùng đúng tên ở Task 4; `user_group/interaction_counts` (Task 1) ở Task 4; `build_sources` (Task 3) ở Task 4 và 6.
- **Tỷ lệ:** plan ngắn hơn khối code tương ứng; không có thân hàm trừ thuật toán không suy ra được từ chữ ký và test.
- **Rủi ro đã biết:** `region_of` là phỏng đoán chưa kiểm trên dữ liệu thật (kiểm ở Task 1 bước 3 bằng cách in vài chục hotel không suy ra được vùng); mẫu lớn có thể chưa đủ booking cho nhóm n_u=0 (kiểm ở Task 7 bước 2).
