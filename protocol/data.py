"""Dữ liệu cho giao thức StayReco: vùng hotel, chia thời gian, n_u, nhóm user."""

import collections
import os

import pandas as pd


def assign_regions(hotels, min_count=5, max_tokens=4):
    """Gán `region` cho mỗi hotel từ chuỗi "tên + địa điểm" của HotelRec.

    Chuỗi gốc đã mất mã thành phố, nên vùng = hậu tố 1-4 từ dài nhất được >= min_count
    hotel cùng chia sẻ (vd "london england"). Không có hậu tố nào đủ phổ biến -> None.
    shortcut: có thể gộp thô (chỉ "england") hoặc nhặt nhầm phần tên hotel; nâng cấp bằng mã geo từ URL.
    """
    toks = [str(d).lower().split() for d in hotels["description"]]
    support = collections.Counter(
        " ".join(t[-n:]) for t in toks for n in range(1, min(max_tokens, len(t)) + 1))

    def pick(t):
        for n in range(min(max_tokens, len(t)), 0, -1):
            s = " ".join(t[-n:])
            if support[s] >= min_count:
                return s
        return None

    out = hotels.copy()
    out["region"] = [pick(t) for t in toks]
    # hậu tố ngắn có thể "đủ phổ biến" nhưng phần lớn hotel đã được gán hậu tố dài hơn: kiểm lại kích thước thật
    size = out.groupby("region")["item_id"].transform("size")
    out.loc[size < min_count, "region"] = None
    return out


def load_sample(path, min_count=30):
    """Đọc thư mục mẫu HotelRec -> (reviews, hotels có cột region)."""
    reviews = pd.read_csv(os.path.join(path, "reviews_slim.csv.gz"),
                          dtype={"user_id": str, "item_id": str})
    hotels = pd.read_csv(os.path.join(path, "hotels.csv.gz"), dtype={"item_id": str})
    return reviews, assign_regions(hotels, min_count=min_count)


def segments_of(sample_dir):
    """{user: home_region} chỉ cho user CÓ vùng; user không có vùng thì vắng mặt (Demographic không có ý kiến)."""
    path = os.path.join(sample_dir, "users_demographics.csv.gz")
    if not os.path.exists(path):
        return {}
    demo = pd.read_csv(path, dtype=str).dropna(subset=["home_region"])
    return dict(zip(demo["user_id"], demo["home_region"]))


def temporal_split(reviews, t_val, t_test):
    """train: ts < t_val; val: t_val <= ts < t_test; test: ts >= t_test."""
    ts = reviews["timestamp"]
    return reviews[ts < t_val], reviews[(ts >= t_val) & (ts < t_test)], reviews[ts >= t_test]


def interaction_counts(history):
    """n_u: mỗi review = 1 tương tác (HotelRec không có view/click)."""
    return history.groupby("user_id").size()


def user_group(n):
    return "0" if n == 0 else "1-4" if n < 5 else "5-19" if n < 20 else ">=20"
