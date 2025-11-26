# backend/db_utils.py
import os
from datetime import datetime, timezone

import numpy as np
from fastapi import HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv  # pip install python-dotenv

# 🔹 Load biến môi trường từ file .env (nếu có)
load_dotenv()

# ⚠️ Nhớ set đúng tên biến trên Render: MONGODB_URI, MONGODB_DB_NAME, MONGODB_FACE_COLLECTION
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "face_recognition_db")
FACE_COLLECTION_NAME = os.getenv("MONGODB_FACE_COLLECTION", "faces")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set. Please configure it in .env or Render env vars")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
face_collection = db[FACE_COLLECTION_NAME]

# ✅ Mỗi user_id chỉ có 1 mặt (1 embedding)
face_collection.create_index("user_id", unique=True)


def _to_unit_vector(vec) -> list[float]:
    """
    Chuyển embedding bất kỳ thành vector đơn vị (chuẩn hóa về độ dài = 1).
    Nếu vector toàn 0 thì báo lỗi.
    """
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Embedding vector has zero norm")
    arr = arr / norm
    return arr.astype(float).tolist()


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

        # ✅ Chuẩn hóa embedding
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
    Tìm kiếm các khuôn mặt tương đồng bằng COSINE SIMILARITY.

    Vì tất cả embedding (trong DB và query) đều đã được chuẩn hóa về vector đơn vị,
    nên tích vô hướng (dot product) chính là cosine similarity.

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
        # ✅ Chuẩn hóa query embedding
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
            print(f" - {r['user_id']} / {r['name']} / cosineSim={r['cosineSim']:.4f}")
        return results

    except Exception as e:
        print(f"[MongoDB] Error finding similar faces: {e}")
        raise HTTPException(status_code=500, detail="Failed to find similar faces")
