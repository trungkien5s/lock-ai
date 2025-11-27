# Smart Locker System using Facial Recognition  
_Mở tủ thông minh bằng khuôn mặt – FastAPI + YOLO + TFLite_

Hệ thống **Smart Locker** cho phép người dùng gửi và lấy đồ bằng **nhận diện khuôn mặt**.  
Không cần tài khoản, không nhập thông tin cá nhân – hệ thống chỉ lưu **embedding khuôn mặt gắn với một ngăn tủ** trong thời gian gửi đồ.

---

## **1. Tính năng chính**

### **🎯 Luồng sử dụng**

#### **Gửi đồ (Store)**
- Giao diện hiển thị:
  - **Số tủ trống** / **Tổng số tủ**
  - Nút **"Lưu đồ"**
- Khi nhấn **Lưu đồ**:
  - Camera bật lên → người dùng nhìn vào camera.
  - Hệ thống thu nhiều frame khuôn mặt, trích xuất embedding.
  - Backend tìm một **tủ trống**, tạo **locker session** và gắn embedding đó với tủ.
  - Tủ tương ứng được **mở ra** cho người dùng gửi đồ.
  - UI hiển thị: **mã tủ**, trạng thái, độ tự tin (confidence).

#### **Lấy đồ (Retrieve)**
- Giao diện hiển thì:
  - Số tủ trống / tổng số tủ
  - Nút **"Lấy đồ"**
- Khi nhấn **Lấy đồ**:
  - Camera bật lên → người dùng nhìn vào camera.
  - Hệ thống chụp khuôn mặt hiện tại, trích xuất embedding.
  - Backend so khớp với các **session đang active** trong DB.
  - Nếu tìm được session có độ tương đồng (cosine similarity) đủ cao:
    - Mở **đúng tủ** mà người đó đã gửi đồ.
    - Đánh dấu session là **closed**, cập nhật tủ về trạng thái **free** sau khi lấy đồ.
  - Nếu không tìm thấy/matching thấp → trả về lỗi "khuôn mặt không khớp".

### **🔢 Thống kê & giao diện**

- Hiển thị:
  - **Tổng số tủ** và **số tủ đang trống** (lấy từ API summary).
  - Danh sách khuôn mặt được detect (nếu bật).
- Camera:
  - Vẽ **bounding box người** và **bounding box khuôn mặt** trên canvas.
  - Hiển thị tên/ID tạm thời hoặc "Unknown face" tuỳ cấu hình.
- Camera sẽ **tự tắt** sau khi:
  - Hoàn thành lưu đồ.
  - Hoàn thành lấy đồ (kể cả success hay thất bại).

---

## **2. Công nghệ sử dụng**

### **Backend**

- **FastAPI** – Web framework chính.
- **Uvicorn** – ASGI server.
- **OpenCV** – Xử lý ảnh cơ bản.
- **YOLO** – Phát hiện người/khuôn mặt (model `best.pt` trong repo).
- **TensorFlow Lite** – Model embedding khuôn mặt (`emotion_model.h5` / TFLite model tương ứng).
- **MongoDB Atlas** – Lưu:
  - Thông tin tủ (**lockers**)
  - Phiên gửi đồ (**locker_sessions**)
- **python-dotenv** – Đọc cấu hình từ `.env`.

### **Frontend**

- **HTML / CSS / JavaScript (vanilla)**.
- **WebRTC / getUserMedia** – Lấy camera từ trình duyệt.
- **Canvas API** – Vẽ bounding box, label.
- **Fetch API** – Gửi frame/ảnh tới backend:
  - API xử lý frame (YOLO, face detect)
  - API lưu đồ
  - API lấy đồ
  - API thống kê tủ

---

## **3. Kiến trúc hệ thống**

```text
┌─────────────────────┐        ┌──────────────────────┐        ┌───────────────────────┐
│  Frontend (Web)     │  HTTP  │  FastAPI Backend     │        │   AI Models            │
│  - Camera/WebRTC    │ <────> │  - API REST          │  <───> │  - YOLO (detect)       │
│  - Canvas overlay   │        │  - Xử lý embedding   │        │  - TFLite embedding    │
└─────────────────────┘        └──────────────────────┘        └───────────────────────┘
                                           │
                                           ▼
                               MongoDB Atlas (lockers, locker_sessions)
```

---

## **4. Cấu trúc thư mục**

```
lock-ai/
├── app/                     # (Nếu dùng thêm, ví dụ cho training / utils)
├── backend/
│   ├── main.py              # FastAPI app, mount static & định nghĩa API
│   ├── db_utils.py          # Hàm thao tác MongoDB (lockers + locker_sessions)
│
├── frontend/
│   ├── index.html           # Trang web chính (UI Smart Locker)
│   ├── static/
│   │   ├── css/
│   │   │   ├── style.css
│   │   │   ├── base.css
│   │   │   ├── layout.css
│   │   │   └── components/*.css
│   │   └── js/
│   │       ├── main.js      # Khởi động app, event handler Store/Retrieve
│   │       ├── camera.js    # Xử lý bật/tắt camera 
│   │       ├── detection.js # Gửi frame → backend, vẽ bounding box
│   │       ├── stats.js     # Cập nhật thống kê
│   │       ├── ui.js        # Hàm UI helper (reset canvas, cập nhật text)
│   │       ├── state.js     # Trạng thái app (isRunning, stream, fps, ...)
│   │       └── config.js    # URL API, FPS, màu sắc, flag hiển thị,...
│
├── dataset/
│   └── widerface-yolo/      # Dataset dùng trong quá trình train YOLO (tham khảo)
│
├── models/
│   ├── best.pt              # YOLO model đã train
│
├── scripts/                 # Script train / convert model, hỗ trợ dev
├── ssl/                     # Chứng chỉ SSL (nếu chạy HTTPS local)
├── .env.example             # Ví dụ file cấu hình môi trường
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT license
├── README.md                # (File này)
└── screenshot.png           # Screenshot giao diện
```

---

## **5. Mô hình dữ liệu (MongoDB)**

### **Collection `lockers`**

Mỗi document đại diện cho 1 tủ:

```json
{
  "locker_id": "L01",
  "status": "free",              // "free" | "occupied"
  "current_session_id": "65f...",// id của session đang active (nếu có)
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

### **Collection `locker_sessions`**

Mỗi document là 1 lần gửi đồ:

```json
{
  "_id": "65f...",
  "locker_id": "L01",
  "face_embedding": [0.123, -0.045, ...],   // vector đã chuẩn hóa (norm = 1)
  "status": "active",                       // "active" (đang gửi) | "closed" (đã lấy đồ)
  "created_at": "2025-01-01T00:00:00Z",
  "closed_at": null
}
```

**Lưu ý:** Hệ thống không cần bảng `faces` riêng vì không đăng ký user, chỉ cần biết "embedding này đang giữ đồ ở tủ nào?".

---

## **6. Cài đặt & chạy hệ thống**

### **6.1. Yêu cầu**

- Python 3.11+
- MongoDB (khuyến nghị MongoDB Atlas)
- (Tuỳ chọn) Virtualenv

### **6.2. Clone project**

```bash
git clone https://github.com/trungkien5s/lock-ai.git
cd lock-ai
```

### **6.3. Tạo môi trường ảo & cài dependency**

```bash
# Tạo venv
python -m venv venv311

# Kích hoạt (Windows)
venv311\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv311/bin/activate

# Cài thư viện Python
pip install -r requirements.txt
```

### **6.4. Tạo file `.env`**

Tạo file `.env` ở root (cùng cấp `backend/`, `frontend/`), dựa trên `.env.example` và cập nhật:

```env
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>/        # URI MongoDB Atlas
MONGODB_DB_NAME=face_recognition_db

# Tuỳ chọn, nếu có dùng trong code:
MONGODB_LOCKER_COLLECTION=lockers
MONGODB_SESSION_COLLECTION=locker_sessions
```

### **6.5. Chạy backend**

Có 2 cách thường dùng:

#### **Cách 1 – Dùng script `run_server.py`** (nếu có sẵn trong repo)

```bash
python run_server.py
```

#### **Cách 2 – Chạy trực tiếp Uvicorn**

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## **7. Sử dụng hệ thống**

1. Mở trình duyệt và truy cập: `http://localhost:8000`
2. Cho phép trình duyệt truy cập camera
3. Sử dụng các chức năng:
   - **Lưu đồ**: Nhấn nút, nhìn vào camera, hệ thống sẽ gán khuôn mặt với tủ trống
   - **Lấy đồ**: Nhấn nút, nhìn vào camera, hệ thống sẽ mở tủ đã gửi đồ nếu khớp khuôn mặt

---

## **8. Troubleshooting**

- **Camera không bật**: Kiểm tra quyền truy cập camera trong trình duyệt
- **Không kết nối được MongoDB**: Kiểm tra lại `MONGODB_URI` trong file `.env`
- **Model không load được**: Đảm bảo file `best.pt` và các model TFLite có trong thư mục `models/`

---

## **9. License**

MIT License - Xem file `LICENSE` để biết thêm chi tiết.