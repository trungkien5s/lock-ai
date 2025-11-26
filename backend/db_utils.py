# backend/db_utils.py
import os
from datetime import datetime, timezone

import numpy as np
from fastapi import HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv  # pip install python-dotenv
from bson import ObjectId

# 🔹 Load biến môi trường từ file .env (nếu có)
load_dotenv()

# ⚠️ Nhớ set đúng tên biến trên Render: MONGODB_URI, MONGODB_DB_NAME, ...
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "face_recognition_db")

# collection tên gì thì set env, không có thì dùng default
FACE_COLLECTION_NAME = os.getenv("MONGODB_FACE_COLLECTION", "faces")
LOCKER_COLLECTION_NAME = os.getenv("MONGODB_LOCKER_COLLECTION", "lockers")
SESSION_COLLECTION_NAME = os.getenv("MONGODB_SESSION_COLLECTION", "locker_sessions")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set. Please configure it in .env or Render env vars")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

face_collection = db[FACE_COLLECTION_NAME]
lockers_collection = db[LOCKER_COLLECTION_NAME]
locker_sessions_collection = db[SESSION_COLLECTION_NAME]

# ✅ Indexes
# Mỗi user_id chỉ có 1 mặt (1 embedding) – dùng cho mode "enroll" nếu cần
face_collection.create_index("user_id", unique=True)

# Lockers: locker_id là duy nhất
lockers_collection.create_index("locker_id", unique=True)
# Sessions: lọc nhanh theo trạng thái
locker_sessions_collection.create_index("status")
locker_sessions_collection.create_index("locker_id")


# ========== COMMON ==========
def _to_unit_vector(vec) -> list[float]:
    """
    Chuẩn hóa embedding về vector đơn vị (norm = 1).
    Nếu vector toàn 0 thì báo lỗi.
    """
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Embedding vector has zero norm")
    arr = arr / norm
    return arr.astype(float).tolist()


# ========== FACES (MODE USER/ADMIN – GIỮ LẠI CHO /process_frame DEBUG) ==========
def get_face_by_user_id(user_id: str):
    """Tìm document khuôn mặt theo user_id. Trả về None nếu không có."""
    return face_collection.find_one({"user_id": user_id})


def store_face_data(user_id: str, name: str, face_embedding):
    """
    Lưu trữ dữ liệu khuôn mặt vào MongoDB.
    - Chuẩn hóa embedding thành vector đơn vị trước khi lưu.
    - Mỗi user_id chỉ có 1 bản ghi (1 khuôn mặt đại diện).
    """
    try:
        if not isinstance(user_id, str):
            raise ValueError(f"user_id must be a string, got {type(user_id)}")
        if not isinstance(name, str):
            raise ValueError(f"name must be a string, got {type(name)}")

        existing = get_face_by_user_id(user_id)
        if existing:
            raise ValueError(f"Face for user_id={user_id} already exists")

        unit_vec = _to_unit_vector(face_embedding)

        now = datetime.now(timezone.utc)
        face_data = {
            "user_id": user_id,
            "name": name,
            "face_embedding": unit_vec,
            "created_at": now,
            "updated_at": now,
        }

        result = face_collection.insert_one(face_data)
        print(
            f"[MongoDB] Stored face data for user_id={user_id}, inserted_id={result.inserted_id}"
        )
        return True

    except Exception as e:
        print(f"[MongoDB] Error storing face data: {e}")
        return False


def find_similar_faces(query_embedding, top_k: int = 1):
    """
    Tìm kiếm các khuôn mặt tương đồng bằng COSINE SIMILARITY trong collection faces.

    Trả về list:
    [
        {
            "user_id": "...",
            "name": "...",
            "cosineSim": 0.97,
        },
        ...
    ]
    """
    try:
        query_vec = _to_unit_vector(query_embedding)
        dim = len(query_vec)

        pipeline = [
            {
                "$addFields": {
                    "cosineSim": {
                        "$reduce": {
                            "input": {
                                "$map": {
                                    "input": {"$range": [0, dim]},
                                    "as": "i",
                                    "in": {
                                        "$multiply": [
                                            {"$arrayElemAt": ["$face_embedding", "$$i"]},
                                            {"$arrayElemAt": [query_vec, "$$i"]},
                                        ]
                                    },
                                }
                            },
                            "initialValue": 0,
                            "in": {"$add": ["$$value", "$$this"]},
                        }
                    }
                }
            },
            {"$sort": {"cosineSim": -1}},
            {"$limit": top_k},
            {"$project": {"user_id": 1, "name": 1, "cosineSim": 1, "_id": 0}},
        ]

        results = list(face_collection.aggregate(pipeline))
        print(f"[MongoDB] find_similar_faces -> {len(results)} result(s)")
        for r in results:
            print(f" - {r.get('user_id')} / {r.get('name')} / cosineSim={r['cosineSim']:.4f}")
        return results

    except Exception as e:
        print(f"[MongoDB] Error finding similar faces: {e}")
        raise HTTPException(status_code=500, detail="Failed to find similar faces")


# ========== LOCKERS + SESSIONS (FLOW LƯU / LẤY ĐỒ) ==========

def init_lockers_if_empty(num_lockers: int = 10):
    """
    Khởi tạo danh sách tủ mặc định nếu collection lockers đang trống.
    Locker_id sẽ là L01, L02, ... theo num_lockers.
    Hàm này có thể được gọi 1 lần khi start server (hoặc bạn chạy script riêng).
    """
    count = lockers_collection.count_documents({})
    if count > 0:
        print(f"[MongoDB] Lockers already initialized ({count} lockers).")
        return

    now = datetime.now(timezone.utc)
    bulk = []
    for i in range(1, num_lockers + 1):
        locker_id = f"L{i:02d}"
        bulk.append(
            {
                "locker_id": locker_id,
                "status": "free",             # "free" | "occupied"
                "current_session_id": None,
                "created_at": now,
                "updated_at": now,
            }
        )

    if bulk:
        lockers_collection.insert_many(bulk)
        print(f"[MongoDB] Initialized {num_lockers} lockers.")


def find_free_locker():
    """
    Tìm 1 tủ đang free.
    Trả về document locker hoặc None nếu hết tủ.
    """
    locker = lockers_collection.find_one({"status": "free"})
    if locker:
        print(f"[MongoDB] find_free_locker -> {locker['locker_id']}")
    else:
        print("[MongoDB] find_free_locker -> no free locker")
    return locker


def mark_locker_occupied(locker_id: str, session_id: str):
    """
    Đánh dấu tủ đang bị chiếm (occupied) và lưu session hiện tại.
    """
    now = datetime.now(timezone.utc)
    result = lockers_collection.update_one(
        {"locker_id": locker_id},
        {
            "$set": {
                "status": "occupied",
                "current_session_id": session_id,
                "updated_at": now,
            }
        },
    )
    print(
        f"[MongoDB] mark_locker_occupied({locker_id}) -> matched={result.matched_count}, modified={result.modified_count}"
    )


def mark_locker_free(locker_id: str):
    """
    Đánh dấu tủ đang trống (free) và reset session hiện tại.
    """
    now = datetime.now(timezone.utc)
    result = lockers_collection.update_one(
        {"locker_id": locker_id},
        {
            "$set": {
                "status": "free",
                "current_session_id": None,
                "updated_at": now,
            }
        },
    )
    print(
        f"[MongoDB] mark_locker_free({locker_id}) -> matched={result.matched_count}, modified={result.modified_count}"
    )


def create_locker_session(locker_id: str, face_embedding):
    """
    Tạo 1 phiên gửi đồ (session) gắn với locker_id.
    Lưu face_embedding đã chuẩn hóa.
    Trả về session_id (string).
    """
    try:
        unit_vec = _to_unit_vector(face_embedding)
        now = datetime.now(timezone.utc)

        doc = {
            "locker_id": locker_id,
            "face_embedding": unit_vec,
            "status": "active",      # "active" = đang có đồ, "closed" = đã lấy xong
            "created_at": now,
            "closed_at": None,
        }

        result = locker_sessions_collection.insert_one(doc)
        session_id = str(result.inserted_id)
        print(
            f"[MongoDB] create_locker_session -> locker_id={locker_id}, session_id={session_id}"
        )
        return session_id

    except Exception as e:
        print(f"[MongoDB] Error creating locker session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create locker session")


def close_locker_session(session_id: str):
    """
    Đóng session (khi khách đã lấy đồ).
    """
    now = datetime.now(timezone.utc)
    try:
        result = locker_sessions_collection.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$set": {
                    "status": "closed",
                    "closed_at": now,
                }
            },
        )
        print(
            f"[MongoDB] close_locker_session({session_id}) -> matched={result.matched_count}, modified={result.modified_count}"
        )
    except Exception as e:
        print(f"[MongoDB] Error closing locker session: {e}")
        raise HTTPException(status_code=500, detail="Failed to close locker session")


# ----- LOCKERS -----
locker_collection = db["lockers"]
locker_collection.create_index("locker_id", unique=True)


def create_lockers(count: int):
    """
    Khởi tạo 'count' tủ nếu chưa tồn tại.
    locker_id = L01, L02, ...
    """
    created = 0
    for i in range(1, count + 1):
        locker_id = f"L{i:02d}"

        exists = locker_collection.find_one({"locker_id": locker_id})
        if exists:
            continue

        now = datetime.now(timezone.utc)
        locker_collection.insert_one({
            "locker_id": locker_id,
            "status": "free",
            "current_user_id": None,
            "created_at": now,
            "updated_at": now,
        })
        created += 1

    return created


def get_free_locker():
    """Lấy tủ trống đầu tiên."""
    return locker_collection.find_one({"status": "free"})


def occupy_locker(locker_id, user_id):
    """Đánh dấu tủ đã được dùng."""
    locker_collection.update_one(
        {"locker_id": locker_id},
        {
            "$set": {
                "status": "occupied",
                "current_user_id": user_id,
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )


def release_locker(locker_id):
    """Trả tủ khi người dùng lấy đồ."""
    locker_collection.update_one(
        {"locker_id": locker_id},
        {
            "$set": {
                "status": "free",
                "current_user_id": None,
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )

def find_active_session_by_face(query_embedding):
    """
    Tìm session đang active có khuôn mặt giống nhất với query_embedding.

    Trả về dict:
    {
        "session_id": "...",
        "locker_id": "L01",
        "cosineSim": 0.95,
    }
    hoặc None nếu không tìm thấy.
    """
    try:
        query_vec = _to_unit_vector(query_embedding)
        dim = len(query_vec)

        pipeline = [
            {"$match": {"status": "active"}},
            {
                "$addFields": {
                    "cosineSim": {
                        "$reduce": {
                            "input": {
                                "$map": {
                                    "input": {"$range": [0, dim]},
                                    "as": "i",
                                    "in": {
                                        "$multiply": [
                                            {"$arrayElemAt": ["$face_embedding", "$$i"]},
                                            {"$arrayElemAt": [query_vec, "$$i"]},
                                        ]
                                    },
                                }
                            },
                            "initialValue": 0,
                            "in": {"$add": ["$$value", "$$this"]},
                        }
                    }
                }
            },
            {"$sort": {"cosineSim": -1}},
            {"$limit": 1},
        ]

        results = list(locker_sessions_collection.aggregate(pipeline))
        if not results:
            print("[MongoDB] find_active_session_by_face -> no active session")
            return None

        doc = results[0]
        session_id = str(doc["_id"])
        locker_id = doc["locker_id"]
        cosineSim = float(doc["cosineSim"])
        print(
            f"[MongoDB] find_active_session_by_face -> session_id={session_id}, locker_id={locker_id}, cosineSim={cosineSim:.4f}"
        )

        return {
            "session_id": session_id,
            "locker_id": locker_id,
            "cosineSim": cosineSim,
        }

    except Exception as e:
        print(f"[MongoDB] Error finding active session by face: {e}")
        raise HTTPException(status_code=500, detail="Failed to find active session by face")
