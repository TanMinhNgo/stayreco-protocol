"""Load dataset cho thực nghiệm.

Dùng Surprise library để load. Hai lựa chọn:
1. movielens-100k: nhỏ, test nhanh (100K ratings)
2. File CSV tự chuẩn bị: user_id, item_id, rating, [timestamp]

Mọi ID đều được đổi sang string để CF / Content / Demo dùng chung một kiểu khóa.
"""

import os
import re
from functools import partial
import pandas as pd
from surprise import Dataset, Reader
from surprise.builtin_datasets import get_dataset_dir


def load_movielens():
    """Load MoviLens 100K từ Surprise (có sẵn, không cần download)."""
    data = Dataset.load_builtin("ml-100k", prompt=False)
    print("✓ Loaded MovieLens 100K")
    return data


def load_movielens_full():
    """Load MovieLens 100K dạng DataFrame, đủ dữ liệu cho cả 3 model.

    MovieLens đóng vai "hotel" để test pipeline trước khi có dataset hotel:
      - ratings  → CF (SVD++)
      - items    → Content: thể loại phim + thập niên  (thay cho mô tả hotel)
      - users    → Demographic: tuổi, giới tính, nghề   (thay cho tuổi, quốc tịch, loại chuyến)

    Returns:
        ratings_df: user_id, item_id, rating, timestamp
        items_df:   item_id, description
        users_df:   user_id, age_group, gender, occupation
    """
    load_movielens()  # đảm bảo đã download
    folder = os.path.join(get_dataset_dir(), "ml-100k", "ml-100k")

    ratings_df = pd.read_csv(
        os.path.join(folder, "u.data"), sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": str, "item_id": str},
    )

    genres = pd.read_csv(os.path.join(folder, "u.genre"), sep="|", names=["genre", "idx"])["genre"].tolist()
    items_df = pd.read_csv(
        os.path.join(folder, "u.item"), sep="|", encoding="latin-1",
        names=["item_id", "title", "release_date", "video_date", "url"] + genres,
        dtype={"item_id": str},
    )
    items_df["description"] = items_df.apply(lambda row: _movie_description(row, genres), axis=1)

    users_df = pd.read_csv(
        os.path.join(folder, "u.user"), sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        dtype={"user_id": str},
    )
    users_df["age_group"] = pd.cut(
        users_df["age"], bins=[0, 17, 24, 34, 44, 54, 200],
        labels=["<18", "18-24", "25-34", "35-44", "45-54", "55+"],
    ).astype(str)

    print(f"  Users: {len(users_df)}, Items: {len(items_df)}, Ratings: {len(ratings_df)}")
    return (
        ratings_df,
        items_df[["item_id", "description"]],
        users_df[["user_id", "age_group", "gender", "occupation"]],
    )


def _movie_description(row, genres):
    """Ghép thể loại + thập niên thành 1 đoạn text, vd: "genre_action genre_scifi decade_1990"."""
    tokens = ["genre_" + re.sub(r"[^a-z]", "", g.lower()) for g in genres if row[g] == 1]
    year = re.search(r"(\d{4})", str(row["release_date"]))
    if year:
        tokens.append(f"decade_{int(year.group(1)) // 10 * 10}")
    return " ".join(tokens)


def load_csv(filepath, rating_scale=(1, 5)):
    """Load dataset từ file CSV.

    CSV cần có ít nhất 3 cột: user_id, item_id, rating
    Có thể thêm: timestamp, user_age, user_country, trip_type,
                  hotel_description, hotel_amenities, ...

    Args:
        filepath: đường dẫn tới file CSV
        rating_scale: (min, max) rating

    Returns:
        surprise.Dataset
    """
    df = pd.read_csv(filepath)

    # Cột tối thiểu cho Surprise
    required = ["user_id", "item_id", "rating"]
    for col in required:
        assert col in df.columns, f"Thiếu cột '{col}' trong CSV"

    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[required], reader)

    print(f"✓ Loaded {filepath}")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Items: {df['item_id'].nunique()}")
    print(f"  Ratings: {len(df)}")
    print(f"  Sparsity: {1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.2%}")

    return data, df


def load_hotel_features(filepath):
    """Load features khách sạn cho Content-Based.

    CSV cần có: item_id, description, amenities, star_rating, location, ...
    """
    df = pd.read_csv(filepath)
    assert "item_id" in df.columns, "Thiếu cột 'item_id'"
    print(f"✓ Loaded hotel features: {len(df)} hotels")
    return df


def load_user_demographics(filepath):
    """Load thông tin nhân khẩu cho Demographic filtering.

    CSV cần có: user_id, age_group, country, trip_type
    """
    df = pd.read_csv(filepath)
    assert "user_id" in df.columns, "Thiếu cột 'user_id'"
    print(f"✓ Loaded user demographics: {len(df)} users")
    return df


def load_movielens_1m(folder="datasets_raw/ml-1m"):
    """MovieLens 1M (1M rating, 6K user, 3.9K phim) cùng dạng (ratings, items, users) với 100K."""
    ratings_df = pd.read_csv(os.path.join(folder, "ratings.dat"), sep="::", engine="python",
                             names=["user_id", "item_id", "rating", "timestamp"],
                             dtype={"user_id": str, "item_id": str})
    genres = None  # ml-1m movies.dat chỉ có "A|B|C" nên không cần danh sách thể loại cố định
    movies = pd.read_csv(os.path.join(folder, "movies.dat"), sep="::", engine="python",
                         names=["item_id", "title", "genres"], encoding="latin-1", dtype={"item_id": str})
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    movies["description"] = (movies["genres"].str.lower().str.replace(r"[^a-z|]", "", regex=True)
                             .str.split("|").apply(lambda g: " ".join("genre_" + x for x in g))
                             + " decade_" + (year // 10 * 10).fillna(0).astype(int).astype(str))
    users = pd.read_csv(os.path.join(folder, "users.dat"), sep="::", engine="python",
                        names=["user_id", "gender", "age", "occupation", "zip"], dtype={"user_id": str})
    # ml-1m đã mã hóa tuổi theo nhóm (1,18,25,35,45,50,56) → dùng thẳng làm age_group
    users["age_group"] = users["age"].astype(str)
    users["occupation"] = users["occupation"].astype(str)
    print(f"  Users: {len(users)}, Items: {len(movies)}, Ratings: {len(ratings_df)}")
    return ratings_df, movies[["item_id", "description"]], users[["user_id", "age_group", "gender", "occupation"]]


def load_tripadvisor_vn(folder="datasets_raw", min_user=3, min_item=5):
    """TripAdvisor Vietnam hotel reviews (Zenodo 7967494): Reviews.csv.

    user = userId, item = locationId, rating 1-5, tripType/language là thuộc tính user.
    Dữ liệu thưa (~1.1 review/user) nên lọc user>=min_user, item>=min_item;
    shortcut: chỉ lọc một lượt (không lặp tới hội tụ), đủ cho thực nghiệm.
    """
    r = pd.read_csv(os.path.join(folder, "Reviews.csv"),
                    dtype={"userId": str, "locationId": str}, parse_dates=["createdDate"])
    r = r.rename(columns={"userId": "user_id", "locationId": "item_id"})
    r["timestamp"] = r["createdDate"].astype("int64") // 10**9
    r = r.drop_duplicates(["user_id", "item_id"])
    r = r[r["item_id"].map(r["item_id"].value_counts()) >= min_item]
    r = r[r["user_id"].map(r["user_id"].value_counts()) >= min_user]
    ratings_df = r[["user_id", "item_id", "rating", "timestamp"]].reset_index(drop=True)

    text = (r["title"].fillna("") + " " + r["text"].fillna("")).groupby(r["item_id"]).apply(
        lambda s: " ".join(s)[:20000])
    names = r.groupby("item_id")["hotelName"].first()
    items_df = pd.DataFrame({"item_id": text.index, "description": (names + " " + text).values})

    g = r.groupby("user_id")
    users_df = pd.DataFrame({
        "trip_type": g["tripType"].agg(lambda s: s.mode().iat[0]),
        "language": g["language"].agg(lambda s: s.mode().iat[0]),
    }).reset_index()
    print(f"  Users: {len(users_df)}, Items: {len(items_df)}, Ratings: {len(ratings_df)}")
    return ratings_df, items_df, users_df


def load_booking_515k(folder="datasets_raw", min_user=3):
    """Booking.com 515K Europe hotel reviews. Không có user ID.

    shortcut: pseudo-user = (quốc tịch, loại chuyến, loại khách) lấy từ Tags, nên một
    "user" là cả một nhóm khách; CF học nhóm chứ không học cá nhân. Hết lỗi này khi có ID thật.
    rating = Reviewer_Score (2.5-10) chia 2, làm tròn, kẹp 1-5.
    """
    r = pd.read_csv(os.path.join(folder, "Hotel_Reviews.csv"), usecols=[
        "Hotel_Name", "Hotel_Address", "Reviewer_Nationality", "Reviewer_Score",
        "Positive_Review", "Negative_Review", "Tags", "Review_Date"])
    tags = r["Tags"].str.lower()
    r["trip_type"] = tags.str.extract(r"(leisure|business)\s+trip")[0].fillna("unknown")
    r["traveler_type"] = tags.str.extract(r"(solo traveler|couple|family with (?:young|older) children|group|travelers with friends)")[0].fillna("unknown")
    r["country"] = r["Reviewer_Nationality"].str.strip()
    r["user_id"] = r["country"] + "|" + r["trip_type"] + "|" + r["traveler_type"]
    r["item_id"] = r["Hotel_Name"]
    r["rating"] = (r["Reviewer_Score"] / 2).round().clip(1, 5)
    r["timestamp"] = pd.to_datetime(r["Review_Date"]).astype("int64") // 10**9

    items_df = r.groupby("item_id").apply(lambda g: pd.Series({
        "description": g["Hotel_Address"].iat[0] + " " + " ".join(
            (g["Positive_Review"] + " " + g["Negative_Review"]).head(40))[:20000]})).reset_index()
    users_df = r.drop_duplicates("user_id")[["user_id", "country", "trip_type", "traveler_type"]]
    # mỗi (user, item) giữ 1 rating: trung bình, vì nhiều khách cùng nhóm review cùng hotel
    ratings_df = (r.groupby(["user_id", "item_id"], as_index=False)
                  .agg(rating=("rating", "mean"), timestamp=("timestamp", "min")))
    ratings_df = ratings_df[ratings_df["user_id"].map(ratings_df["user_id"].value_counts()) >= min_user]
    users_df = users_df[users_df["user_id"].isin(ratings_df["user_id"])]
    print(f"  Users: {len(users_df)}, Items: {len(items_df)}, Ratings: {len(ratings_df)}")
    return ratings_df, items_df, users_df


def load_tripadvisor_us(folder="datasets_raw/tripadvisor-us", min_user=10, min_item=20):
    """TripAdvisor US scraped (Mrazova & Behun, ITAT 2023): 3.1M review, 3.260 hotel, user thật.

    Quét reviews.json.gz một lượt (bỏ phần text), cache ra reviews_slim.csv.gz lần đầu.
    Content = mô tả + tiện nghi của hotel (hotels.json.gz); Demographic = trip_type + hometown.
    shortcut: hometown lấy phần cuối sau dấu phẩy (bang/quốc gia), ~70% author không khai hometown → "unknown".
    """
    import gzip, json
    slim = os.path.join(folder, "reviews_slim.csv.gz")
    if not os.path.exists(slim):
        rows = []
        with gzip.open(os.path.join(folder, "reviews.json.gz"), "rt", encoding="utf8") as g:
            for line in g:
                r = json.loads(line)
                rows.append((r["author_id"], r["hotel_id"], r["ratings"].get("Overall"),
                             r["created"], r["trip_type"]))
        pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "created", "trip_type"]).to_csv(slim, index=False)
    r = pd.read_csv(slim, dtype={"user_id": str, "item_id": str}).dropna(subset=["rating"])
    r["timestamp"] = pd.to_datetime(r["created"]).astype("int64") // 10**9
    r["trip_type"] = r["trip_type"].fillna("unknown")
    r = r.drop_duplicates(["user_id", "item_id"])
    r = r[r["item_id"].map(r["item_id"].value_counts()) >= min_item]
    r = r[r["user_id"].map(r["user_id"].value_counts()) >= min_user]
    ratings_df = r[["user_id", "item_id", "rating", "timestamp"]].reset_index(drop=True)

    hotels = pd.read_json(os.path.join(folder, "hotels.json.gz"), lines=True, dtype={"id": str})
    hotels = hotels[hotels["id"].isin(ratings_df["item_id"])]
    items_df = pd.DataFrame({"item_id": hotels["id"], "description": (
        hotels["name"].fillna("") + " " + hotels["description"].fillna("") + " "
        + hotels["amenities"].apply(lambda a: " ".join(a) if isinstance(a, list) else str(a or "")))})

    authors = pd.read_json(os.path.join(folder, "authors.json.gz"), lines=True, dtype={"id": str})
    authors = authors[authors["id"].isin(ratings_df["user_id"])]
    users_df = pd.DataFrame({"user_id": authors["id"], "hometown": authors["hometown"].fillna("unknown")
                             .str.split(",").str[-1].str.strip()})
    mode = r.groupby("user_id")["trip_type"].agg(lambda s: s.mode().iat[0])
    users_df["trip_type"] = users_df["user_id"].map(mode)
    print(f"  Users: {len(users_df)}, Items: {len(items_df)}, Ratings: {len(ratings_df)}")
    return ratings_df, items_df, users_df


def load_hotelrec(folder="datasets_raw/hotelrec", min_user=3, min_item=5,
                  max_users=None, max_items=None):
    """Bounded HotelRec experiment with real author IDs, no review-text leakage.

    shortcut: activity-capped sample is not representative of all 50M reviews;
    use a population sampling protocol for population-level conclusions.
    Optional exact-username profile enrichment supplies self-reported home regions.
    """
    import hashlib
    import json
    from pathlib import Path
    from prepare_hotelrec import INVALID_AUTHORS

    folder = Path(folder)
    source_path = folder / "source.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Prepared HotelRec data missing: {source_path}. See preparation commands in README.md.")
    source = json.loads(source_path.read_text(encoding="utf8"))
    limits = source.get("sampling_limits", {})
    max_users = limits.get("max_users", 1000) if max_users is None else max_users
    max_items = limits.get("max_items", 1000) if max_items is None else max_items
    if min(min_user, min_item, max_users, max_items) < 1:
        raise ValueError("HotelRec limits must be positive")
    for name in ("reviews_slim.csv.gz", "hotels.csv.gz"):
        with (folder / name).open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != source["prepared_sha256"][name]:
            raise ValueError(f"HotelRec checksum mismatch: {name}")
    ratings = pd.read_csv(folder / "reviews_slim.csv.gz",
                          dtype={"user_id": str, "item_id": str}, keep_default_na=False)
    if (ratings[["user_id", "item_id"]].eq("").any().any()
            or not ratings["rating"].between(1, 5).all()
            or ratings["timestamp"].isna().any()):
        raise ValueError("Invalid HotelRec IDs, ratings or timestamps")
    invalid_author_rows = ratings["user_id"].str.lower().isin(INVALID_AUTHORS)
    excluded_author_rows = int(invalid_author_rows.sum())
    ratings = ratings[~invalid_author_rows]
    ratings = ratings.sort_values("timestamp", kind="stable").drop_duplicates(
        ["user_id", "item_id"], keep="last")

    # Explicit tie-break on ID makes the activity cap independent of input order.
    def active_ids(column, limit):
        counts = ratings.groupby(column).size().rename("count").reset_index()
        return counts.sort_values(["count", column], ascending=[False, True])[column].head(limit)

    ratings = ratings[ratings["item_id"].isin(active_ids("item_id", max_items))]
    ratings = ratings[ratings["user_id"].isin(active_ids("user_id", max_users))]
    while len(ratings):
        previous = len(ratings)
        ratings = ratings[ratings["item_id"].map(ratings["item_id"].value_counts()) >= min_item]
        ratings = ratings[ratings["user_id"].map(ratings["user_id"].value_counts()) >= min_user]
        if len(ratings) == previous:
            break
    if ratings.empty:
        raise ValueError("HotelRec subset is empty after activity/core filters")
    ratings = ratings.reset_index(drop=True)
    hotels = pd.read_csv(folder / "hotels.csv.gz", dtype={"item_id": str}, keep_default_na=False)
    hotels = hotels[hotels["item_id"].isin(ratings["item_id"])].reset_index(drop=True)
    if (hotels["item_id"].duplicated().any() or hotels["description"].eq("").any()
            or set(hotels["item_id"]) != set(ratings["item_id"])):
        raise ValueError("HotelRec hotel features are missing, empty or duplicated")
    users = ratings[["user_id"]].drop_duplicates().assign(demographic=pd.NA)
    profiles_path, metadata_path = folder / "users_demographics.csv.gz", folder / "demographics_source.json"
    demographics = {"available": False, "known_users": 0, "missing_users": len(users)}
    if profiles_path.exists() != metadata_path.exists():
        raise ValueError("HotelRec demographic enrichment is incomplete")
    if profiles_path.exists():
        demographics_source = json.loads(metadata_path.read_text(encoding="utf8"))
        with profiles_path.open("rb") as stream:
            profiles_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if (profiles_sha != demographics_source["profiles_sha256"]
                or source["prepared_sha256"]["reviews_slim.csv.gz"] != demographics_source["target_reviews_sha256"]):
            raise ValueError("HotelRec demographic checksum or target sample mismatch")
        profiles = pd.read_csv(profiles_path, dtype=str, keep_default_na=False)
        required = {"user_id", "profile_id", "hometown", "home_region", "match_status"}
        if (not required.issubset(profiles.columns) or profiles["user_id"].duplicated().any()
                or profiles["user_id"].eq("").any() or not set(users["user_id"]).issubset(profiles["user_id"])
                or len(profiles) != demographics_source["counts"]["target_users"]
                or (profiles["home_region"].ne("") & profiles["match_status"].ne("matched")).any()):
            raise ValueError("Invalid HotelRec demographic profiles")
        regions = profiles.set_index("user_id")["home_region"].replace("", pd.NA)
        users["demographic"] = users["user_id"].map(regions)
        known_users = int(users["demographic"].notna().sum())
        demographics = {"available": True, "feature": "home_region", "known_users": known_users,
                        "missing_users": len(users) - known_users,
                        "groups": int(users["demographic"].nunique()), "source": demographics_source}
    ratings.attrs["dataset_details"] = {
        "source": source,
        "demographics": demographics,
        "invalid_author_rows_removed": excluded_author_rows,
        "subset": {"max_users": max_users, "max_items": max_items,
                   "min_user": min_user, "min_item": min_item,
                   "selection": "Most reviewed hotels, then most active users; iterative core filtering",
                   "duplicate_pairs": "Keep latest timestamp"},
    }
    print(f"  HotelRec subset: {len(users):,} real users, {len(hotels):,} hotels, {len(ratings):,} ratings")
    print(f"  Content = URL name/location; demographic home regions: {demographics['known_users']}/{len(users)} users.")
    return ratings, hotels, users


# tên dataset → (hàm load, cột demographic dùng cho dataset đó)
DATASETS = {
    "ml-100k": (load_movielens_full, ("age_group", "gender", "occupation")),
    "ml-1m": (load_movielens_1m, ("age_group", "gender", "occupation")),
    "booking-515k": (load_booking_515k, ("country", "trip_type", "traveler_type")),
    "tripadvisor-us": (load_tripadvisor_us, ("hometown", "trip_type")),
    "tripadvisor-vn": (load_tripadvisor_vn, ("trip_type", "language")),
    "hotelrec": (load_hotelrec, ("demographic",)),
    "hotelrec-full-sample": (partial(load_hotelrec, folder="datasets_raw/hotelrec_full_sample"), ("demographic",)),
}
