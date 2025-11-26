# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import numpy as np
import cv2
from typing import List
from backend.db_utils import lockers_collection  # hoặc export hàm trong db_utils
from app.box_detector import Detector
from backend import db_utils  # Import các hàm từ db_utils.py

app = FastAPI(
    title="Smart Locker System using Facial Recognition",
    description="FastAPI backend for real-time face recognition and smart locker control.",
    version="1.0.0",
)

detector = Detector()

# Khởi tạo danh sách tủ nếu cần (ví dụ 10 tủ: L01..L10)
# Bạn có thể comment dòng này nếu muốn tự init bằng script riêng
db_utils.init_lockers_if_empty(num_lockers=10)

# ----------------- CORS -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể siết lại origin khi deploy
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- Pydantic Models -----------------
class StoreResponse(BaseModel):
    status: str              # "granted" | "denied"
    locker_id: str | None
    confidence: float | None
    message: str


class RetrieveResponse(BaseModel):
    status: str              # "granted" | "denied"
    locker_id: str | None
    confidence: float | None
    message: str


# Ngưỡng để quyết định cho mở tủ khi LẤY ĐỒ
UNLOCK_THRESHOLD = 0.93


# ----------------- API: process_frame (giữ để debug/thống kê) -----------------
@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    """
    Nhận 1 frame từ camera, detect person & face,
    trả về bounding boxes + tên khuôn mặt (nếu match DB faces).
    Dùng cho debug / demo.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh từ file upload")

    person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)

    face_boxes_for_response = []
    for (coords, conf, emotion, embedding) in face_boxes:
        if embedding is not None:
            similar_faces = db_utils.find_similar_faces(embedding, top_k=3)
            face_names = [face.get("name") for face in similar_faces]
            face_boxes_for_response.append(
                {
                    "coords": coords,
                    "confidence": conf,
                    "emotion": emotion,
                    "similar_faces": face_names,
                }
            )
        else:
            face_boxes_for_response.append(
                {
                    "coords": coords,
                    "confidence": conf,
                    "emotion": emotion,
                    "similar_faces": [],
                }
            )

    return {
        "persons": person_count,
        "faces": face_count,
        "person_boxes": [
            {"coords": coords, "confidence": conf}
            for (coords, conf, action) in person_boxes
        ],
        "face_boxes": face_boxes_for_response,
    }


# ----------------- API: LƯU ĐỒ (STORE) -----------------
@app.post("/store", response_model=StoreResponse)
async def store_item(
    files: List[UploadFile] = File(
        None, description="Danh sách frame chụp khuôn mặt (tùy chọn nhiều frame)"
    ),
    file: UploadFile = File(
        None, description="1 frame chụp khuôn mặt (fallback, tương thích đơn giản)"
    ),
):
    """
    Flow LƯU ĐỒ:
    - Frontend bấm 'Lưu đồ' -> bật camera -> gửi 1 hoặc nhiều frame lên endpoint này.
    - Backend:
      + Trích embedding khuôn mặt (có thể lấy trung bình nhiều frame cho ổn định).
      + Tìm 1 tủ đang 'free'.
      + Tạo session mới (locker_sessions) gắn với locker đó.
      + Đánh dấu tủ 'occupied'.
      + Trả về locker_id để bật tủ & hiển thị cho người dùng.
    """

    uploads: List[UploadFile] = []

    if files:
        uploads.extend(files)
    if file is not None:
        uploads.append(file)

    if not uploads:
        raise HTTPException(status_code=400, detail="Không nhận được file ảnh nào để lưu đồ")

    embeddings: list[np.ndarray] = []

    # Lấy embedding từ các frame
    for upload in uploads:
        contents = await upload.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("[STORE] Bỏ qua frame: không đọc được ảnh")
            continue

        person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)

        if face_count == 0:
            print("[STORE] Frame không có khuôn mặt, bỏ qua")
            continue

        coords, conf, emotion, embedding = face_boxes[0]

        if embedding is None:
            print("[STORE] Không trích xuất được embedding, bỏ qua frame này")
            continue

        embeddings.append(np.array(embedding, dtype=np.float32))

    if len(embeddings) == 0:
        raise HTTPException(
            status_code=400,
            detail="Không thu được khuôn mặt hợp lệ nào. Hãy thử lại và đảm bảo mặt rõ, đủ sáng.",
        )

    # Trung bình embedding để có vector đại diện ổn định hơn
    embed_stack = np.stack(embeddings, axis=0)  # shape: (N, D)
    avg_embedding = np.mean(embed_stack, axis=0)
    print(f"[STORE] Collected {len(embeddings)} embeddings, using averaged template.")

    # Tìm 1 tủ đang free
    locker = db_utils.find_free_locker()
    if not locker:
        return StoreResponse(
            status="denied",
            locker_id=None,
            confidence=None,
            message="Hiện không còn tủ trống, vui lòng thử lại sau.",
        )

    locker_id = locker["locker_id"]

    # Tạo session mới
    session_id = db_utils.create_locker_session(locker_id=locker_id, face_embedding=avg_embedding)

    # Đánh dấu tủ đang bị chiếm
    db_utils.mark_locker_occupied(locker_id=locker_id, session_id=session_id)

    # (Sau này) tại đây có thể gửi lệnh mở tủ locker_id ra phần cứng

    return StoreResponse(
        status="granted",
        locker_id=locker_id,
        confidence=None,  # có thể thêm confidence nếu bạn muốn
        message=f"Tủ {locker_id} đã được cấp. Vui lòng gửi đồ vào tủ.",
    )


# ----------------- API: LẤY ĐỒ (RETRIEVE) -----------------
@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_item(file: UploadFile = File(...)):
    """
    Flow LẤY ĐỒ:
    - Frontend bấm 'Lấy đồ' -> bật camera -> gửi 1 frame khuôn mặt hiện tại.
    - Backend:
      + Trích embedding khuôn mặt.
      + Tìm session đang 'active' có cosine similarity cao nhất.
      + Nếu cosineSim >= UNLOCK_THRESHOLD -> mở tủ đó, đóng session, giải phóng tủ.
      + Ngược lại -> từ chối.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh từ file upload")

    person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)

    if face_count == 0:
        raise HTTPException(status_code=400, detail="Không phát hiện khuôn mặt nào trong ảnh")

    coords, conf, emotion, embedding = face_boxes[0]

    if embedding is None:
        raise HTTPException(
            status_code=400, detail="Không trích xuất được embedding khuôn mặt"
        )

    # Tìm session active giống nhất
    best_session = db_utils.find_active_session_by_face(embedding)

    if not best_session:
        return RetrieveResponse(
            status="denied",
            locker_id=None,
            confidence=None,
            message="Không tìm thấy tủ tương ứng với khuôn mặt này.",
        )

    locker_id = best_session["locker_id"]
    cosineSim = float(best_session["cosineSim"])

    if cosineSim < UNLOCK_THRESHOLD:
        return RetrieveResponse(
            status="denied",
            locker_id=locker_id,
            confidence=cosineSim,
            message="Độ tương đồng khuôn mặt chưa đủ để mở tủ.",
        )

    # Nếu đạt ngưỡng => cho mở tủ, đóng session & free locker
    session_id = best_session["session_id"]
    db_utils.close_locker_session(session_id=session_id)
    db_utils.mark_locker_free(locker_id=locker_id)

    # TODO: gửi lệnh mở tủ locker_id ra phần cứng tại đây

    return RetrieveResponse(
        status="granted",
        locker_id=locker_id,
        confidence=cosineSim,
        message=f"Đã mở tủ {locker_id}. Vui lòng lấy đồ.",
    )


@app.get("/lockers/summary")
async def lockers_summary():
    total = lockers_collection.count_documents({})
    free = lockers_collection.count_documents({"status": "free"})
    occupied = lockers_collection.count_documents({"status": "occupied"})
    return {
        "total_lockers": total,
        "free_lockers": free,
        "occupied_lockers": occupied,
    }

@app.post("/init_lockers")
async def init_lockers(count: int = 12):
    created = db_utils.create_lockers(count)
    return {
        "requested": count,
        "created": created,
        "message": f"Đã tạo {created} tủ mới"
    }


# ----------------- HEALTH CHECK -----------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ----------------- STATIC FRONTEND -----------------
# Giả định cấu trúc: project_root/frontend/...
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
