# backend/db_utils.py
import os
from datetime import datetime, timezone

import numpy as np
from fastapi import HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv  # pip install python-dotenv

# 🔹 Load biến môi trường từ file .env (nếu có)
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "face_recognition_db")
FACE_COLLECTION_NAME = os.getenv("MONGODB_FACE_COLLECTION", "faces")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set. Please configure it in .env")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
face_collection = db[FACE_COLLECTION_NAME]

# ✅ Tạo index unique cho user_id để đảm bảo 1 user chỉ có 1 bản ghi
face_collection.create_index("user_id", unique=True)


def get_face_by_user_id(user_id: str):
    """Tìm document khuôn mặt theo user_id. Trả về None nếu không có."""
    return face_collection.find_one({"user_id": user_id})


def store_face_data(user_id: str, name: str, face_embedding):
    """
    Lưu trữ dữ liệu khuôn mặt vào MongoDB.
    Mỗi document tương ứng với 1 user/locker.
    KHÔNG cho phép một user_id có nhiều bản ghi.
    """
    try:
        if not isinstance(user_id, str):
            raise ValueError(f"user_id must be a string, got {type(user_id)}")
        if not isinstance(name, str):
            raise ValueError(f"name must be a string, got {type(name)}")

        # ⚠️ Check: nếu user đã có khuôn mặt thì không cho đăng ký nữa
        existing = get_face_by_user_id(user_id)
        if existing:
            raise ValueError(f"Face for user_id={user_id} already exists")

        # Đảm bảo embedding là list số
        if isinstance(face_embedding, np.ndarray):
            face_embedding = face_embedding.astype(float).tolist()
        if not isinstance(face_embedding, list) or not all(
            isinstance(x, (int, float)) for x in face_embedding
        ):
            raise ValueError(
                f"face_embedding must be a list of numbers, got {type(face_embedding)}"
            )

        now = datetime.now(timezone.utc)
        face_data = {
            "user_id": user_id,
            "name": name,
            "face_embedding": face_embedding,
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
    Tìm kiếm các khuôn mặt tương đồng bằng tích vô hướng (dot product).

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
        if isinstance(query_embedding, np.ndarray):
            query_vec = query_embedding.astype(float).tolist()
        else:
            query_vec = np.array(query_embedding, dtype=np.float32).tolist()

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
        return results

    except Exception as e:
        print(f"[MongoDB] Error finding similar faces: {e}")
        raise HTTPException(status_code=500, detail="Failed to find similar faces")
