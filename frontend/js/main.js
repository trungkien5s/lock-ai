// static/js/main.js
import config from "./config.js";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const personCountEl = document.getElementById("personCount");
const freeLockerCountEl = document.getElementById("freeLockerCount");
const totalLockerCountEl = document.getElementById("totalLockerCount");
const peopleListContainer = document.querySelector(
  "#detectedPeopleList .people-list-content"
);

const cameraSelect = document.getElementById("cameraSelect");
const toggleRearCamera = document.getElementById("toggleRearCamera");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");

const storeButton = document.getElementById("storeButton");
const retrieveButton = document.getElementById("retrieveButton");

const lockerStatusText = document.getElementById("lockerStatusText");
const lockerIdText = document.getElementById("lockerIdText");
const recognizedConfidenceText = document.getElementById(
  "recognizedConfidenceText"
);

let currentStream = null;
let isStreaming = false;
let isProcessing = false;
let lastFrameTime = 0;
const frameInterval = 1000 / config.frameRate;

// ================== Helper: Bật camera tự động nếu chưa bật ==================
async function ensureCameraStarted() {
  if (!isStreaming) {
    console.log("📷 Đang bật camera...");
    await startCamera();
    // Đợi camera ổn định
    await new Promise(res => setTimeout(res, 800));
  }
}

// ================== Helper: Resize canvas khớp với video ==================
function resizeCanvasToVideo() {
  if (!video.videoWidth || !video.videoHeight) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// ================== Gọi API lấy số tủ trống ==================
async function fetchLockerSummary() {
  try {
    const res = await fetch(config.lockersSummaryUrl, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });

    if (!res.ok) {
      console.warn("⚠️ Không lấy được thống kê tủ. Status:", res.status);
      return;
    }

    const data = await res.json();
    console.log("✅ Locker summary:", data);

    // Cập nhật UI
    if (data.free_lockers !== undefined && freeLockerCountEl) {
      freeLockerCountEl.textContent = data.free_lockers;
    }

    if (data.total_lockers !== undefined && totalLockerCountEl) {
      totalLockerCountEl.textContent = data.total_lockers;
    }
  } catch (err) {
    console.error("❌ Lỗi khi fetch locker summary:", err);
  }
}

// ================== Lấy danh sách camera ==================
async function loadCameraDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === "videoinput");

    if (!cameraSelect) return;

    cameraSelect.innerHTML = "";

    videoDevices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = device.label || `Camera ${index + 1}`;
      cameraSelect.appendChild(option);
    });

    if (videoDevices.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "Không tìm thấy camera";
      cameraSelect.appendChild(option);
    }
  } catch (err) {
    console.error("Lỗi tải danh sách camera:", err);
    if (cameraSelect) {
      cameraSelect.innerHTML = "";
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "Không truy cập được thiết bị camera";
      cameraSelect.appendChild(option);
    }
  }
}

// ================== Bắt đầu camera ==================
async function startCamera() {
  try {
    stopCamera();

    const selectedDeviceId = cameraSelect ? cameraSelect.value : "";

    let constraints;
    if (config.isMobile) {
      constraints = {
        video: {
          facingMode:
            toggleRearCamera && toggleRearCamera.checked
              ? "environment"
              : "user",
        },
        audio: false,
      };
    } else {
      constraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId } }
          : true,
        audio: false,
      };
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    currentStream = stream;
    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        resizeCanvasToVideo();
        resolve();
      };
    });

    isStreaming = true;
    lastFrameTime = 0;
    requestAnimationFrame(processLoop);
    
    console.log("✅ Camera đã bật");
  } catch (err) {
    console.error("❌ Không thể bật camera:", err);
    alert("Không thể bật camera. Hãy kiểm tra quyền truy cập camera.");
  }
}

// ================== Dừng camera ==================
function stopCamera() {
  isStreaming = false;
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
    currentStream = null;
  }
  if (ctx && canvas) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  console.log("🛑 Camera đã dừng");
}

// ================== Capture frame hiện tại thành Blob ==================
function captureFrameAsBlob() {
  return new Promise((resolve) => {
    if (!video.videoWidth || !video.videoHeight) {
      resolve(null);
      return;
    }

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    tempCanvas.toBlob(
      (blob) => {
        resolve(blob);
      },
      "image/jpeg",
      0.9
    );
  });
}

// ================== Vẽ bounding box & cập nhật UI thống kê ==================
function drawDetections(data) {
  if (!ctx || !canvas) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!data) return;

  const { persons, person_boxes = [], face_boxes = [] } = data;

  // Cập nhật thống kê người
  if (personCountEl) {
    personCountEl.textContent = persons ?? 0;
  }

  if (peopleListContainer) {
    peopleListContainer.innerHTML = "";
  }

  ctx.lineWidth = config.borderWidth;
  ctx.font = `${
    config.isMobile
      ? config.mobileLabelFontSize
      : config.desktopLabelFontSize
  }px sans-serif`;
  ctx.textBaseline = "top";

  // Vẽ khung person
  if (config.showPersons && Array.isArray(person_boxes)) {
    ctx.strokeStyle = config.personColor;
    ctx.fillStyle = "rgba(0, 0, 0, 0.5)";

    person_boxes.forEach((box) => {
      const { coords, confidence } = box;
      if (!coords || coords.length < 4) return;
      const [x1, y1, x2, y2] = coords;
      const w = x2 - x1;
      const h = y2 - y1;

      ctx.strokeRect(x1, y1, w, h);

      let label = "Person";
      if (config.showConfidence && typeof confidence === "number") {
        label += ` ${(confidence * 100).toFixed(1)}%`;
      }

      const textWidth = ctx.measureText(label).width;
      const textHeight =
        (config.isMobile
          ? config.mobileLabelFontSize
          : config.desktopLabelFontSize) +
        config.labelPadding * 2;

      ctx.fillRect(
        x1,
        y1 - textHeight - config.labelMargin,
        textWidth + config.labelPadding * 2,
        textHeight
      );
      ctx.fillStyle = "#ffffff";
      ctx.fillText(
        label,
        x1 + config.labelPadding,
        y1 - textHeight - config.labelMargin + config.labelPadding
      );
      ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    });
  }

  // Vẽ khung face
  if (config.showFaces && Array.isArray(face_boxes)) {
    ctx.strokeStyle = config.faceColor;
    ctx.fillStyle = "rgba(0, 0, 0, 0.5)";

    face_boxes.forEach((box) => {
      const { coords, confidence, similar_faces } = box;
      if (!coords || coords.length < 4) return;
      const [x1, y1, x2, y2] = coords;
      const w = x2 - x1;
      const h = y2 - y1;

      ctx.strokeRect(x1, y1, w, h);

      let faceName = null;
      if (
        config.showFaceNames &&
        Array.isArray(similar_faces) &&
        similar_faces.length > 0
      ) {
        faceName = similar_faces[0];
      }

      if (peopleListContainer) {
        const item = document.createElement("div");
        item.className = "person-item";
        item.textContent = faceName || "Unknown face";
        peopleListContainer.appendChild(item);
      }

      let labelParts = [];
      if (faceName) labelParts.push(faceName);
      if (config.showConfidence && typeof confidence === "number") {
        labelParts.push(`${(confidence * 100).toFixed(1)}%`);
      }
      if (emotion) labelParts.push(emotion);

      const label = labelParts.join(" | ") || "Face";

      const textWidth = ctx.measureText(label).width;
      const textHeight =
        (config.isMobile
          ? config.mobileLabelFontSize
          : config.desktopLabelFontSize) +
        config.labelPadding * 2;

      ctx.fillRect(
        x1,
        y1 - textHeight - config.labelMargin,
        textWidth + config.labelPadding * 2,
        textHeight
      );
      ctx.fillStyle = "#ffffff";
      ctx.fillText(
        label,
        x1 + config.labelPadding,
        y1 - textHeight - config.labelMargin + config.labelPadding
      );
      ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    });
  }
}

// ================== Loop gửi frame lên /process_frame ==================
async function processLoop(timestamp) {
  if (!isStreaming) return;

  if (timestamp - lastFrameTime < frameInterval || isProcessing) {
    requestAnimationFrame(processLoop);
    return;
  }

  lastFrameTime = timestamp;
  isProcessing = true;

  try {
    const blob = await captureFrameAsBlob();
    if (!blob) {
      isProcessing = false;
      requestAnimationFrame(processLoop);
      return;
    }

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    const response = await fetch(config.serverUrl, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      console.error("Lỗi từ /process_frame:", await response.text());
    } else {
      const data = await response.json();
      drawDetections(data);
    }
  } catch (err) {
    console.error("Lỗi trong processLoop:", err);
  }

  isProcessing = false;
  requestAnimationFrame(processLoop);
}

// ================== Handler: Lưu đồ (STORE) ==================
async function handleStoreItem() {
  try {
    // Vô hiệu hóa nút
    storeButton.disabled = true;
    
    // Bật camera tự động
    await ensureCameraStarted();
    
    // Cập nhật trạng thái
    lockerStatusText.textContent = "📷 Đang thu thập khuôn mặt... Vui lòng nhìn thẳng vào camera!";
    lockerIdText.textContent = "-";
    recognizedConfidenceText.textContent = "-";

    const NUM_FRAMES = 5;
    const FRAME_DELAY = 800; // 800ms giữa mỗi frame = ~4-5s tổng
    const formData = new FormData();

    // Thu thập 5 frame trong 4-5 giây
    for (let i = 0; i < NUM_FRAMES; i++) {
      lockerStatusText.textContent = `📷 Thu thập ảnh ${i + 1}/${NUM_FRAMES}... Giữ nguyên tư thế!`;
      
      const blob = await captureFrameAsBlob();
      if (blob) {
        formData.append("files", blob, `store_${i}.jpg`);
      }
      
      if (i < NUM_FRAMES - 1) {
        await new Promise(resolve => setTimeout(resolve, FRAME_DELAY));
      }
    }

    // Gửi lên server
    lockerStatusText.textContent = "⏳ Đang xử lý và phân bổ tủ...";
    
    const res = await fetch(config.storeUrl, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    // Cập nhật kết quả
    lockerStatusText.textContent = data.message;
    lockerIdText.textContent = data.locker_id || "-";
    recognizedConfidenceText.textContent =
      data.confidence ? data.confidence.toFixed(3) : "-";

    // Cập nhật số tủ trống
    await fetchLockerSummary();

    // Thông báo thành công
    if (data.status === "success") {
      alert(`✅ ${data.message}\n🔑 Tủ số: ${data.locker_id}\n\nVui lòng ghi nhớ số tủ để lấy đồ sau!`);
    } else {
      alert(`⚠️ ${data.message}`);
    }
  } catch (err) {
    console.error("❌ Lỗi khi lưu đồ:", err);
    lockerStatusText.textContent = "❌ Lỗi khi lưu đồ";
    alert("❌ Lỗi khi lưu đồ. Vui lòng thử lại!");
  } finally {
    storeButton.disabled = false;
  }
}

// ================== Handler: Lấy đồ (RETRIEVE) ==================
async function handleRetrieveItem() {
  try {
    // Vô hiệu hóa nút
    retrieveButton.disabled = true;
    
    // Bật camera tự động
    await ensureCameraStarted();
    
    // Cập nhật trạng thái
    lockerStatusText.textContent = "📷 Đang xác thực khuôn mặt... Vui lòng nhìn thẳng vào camera!";
    lockerIdText.textContent = "-";
    recognizedConfidenceText.textContent = "-";

    // Đợi thêm 1 giây để người dùng chuẩn bị
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Chụp ảnh xác thực
    lockerStatusText.textContent = "📸 Đang chụp và xác thực...";
    const blob = await captureFrameAsBlob();
    
    if (!blob) {
      throw new Error("Không thể chụp ảnh từ camera");
    }

    const formData = new FormData();
    formData.append("file", blob, "retrieve.jpg");

    // Gửi lên server
    lockerStatusText.textContent = "⏳ Đang tìm kiếm tủ của bạn...";
    
    const res = await fetch(config.retrieveUrl, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    // Cập nhật kết quả
    lockerStatusText.textContent = data.message;
    lockerIdText.textContent = data.locker_id || "-";
    recognizedConfidenceText.textContent =
      typeof data.confidence === "number"
        ? data.confidence.toFixed(3)
        : "-";

    // Cập nhật số tủ trống
    await fetchLockerSummary();

    // Thông báo kết quả
    if (data.status === "success") {
      alert(`✅ ${data.message}\n🔓 Tủ số ${data.locker_id} đã được mở!\n🎯 Độ chính xác: ${(data.confidence * 100).toFixed(1)}%`);
    } else {
      alert(`⚠️ ${data.message}`);
    }
  } catch (err) {
    console.error("❌ Lỗi khi lấy đồ:", err);
    lockerStatusText.textContent = "❌ Lỗi khi lấy đồ";
    alert("❌ Lỗi khi lấy đồ. Vui lòng thử lại!");
  } finally {
    retrieveButton.disabled = false;
  }
}

// ================== Gắn event listeners ==================
function setupEventListeners() {
  if (startButton) {
    startButton.addEventListener("click", () => {
      startCamera();
    });
  }

  if (stopButton) {
    stopButton.addEventListener("click", () => {
      stopCamera();
    });
  }

  if (cameraSelect) {
    cameraSelect.addEventListener("change", () => {
      if (!config.isMobile && isStreaming) {
        startCamera();
      }
    });
  }

  if (toggleRearCamera) {
    toggleRearCamera.addEventListener("change", () => {
      if (config.isMobile && isStreaming) {
        startCamera();
      }
    });
  }

  if (storeButton) {
    storeButton.addEventListener("click", handleStoreItem);
  }

  if (retrieveButton) {
    retrieveButton.addEventListener("click", handleRetrieveItem);
  }

  window.addEventListener("resize", () => {
    resizeCanvasToVideo();
  });
}

// ================== Khởi động ==================
async function init() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("❌ Trình duyệt của bạn không hỗ trợ camera (getUserMedia).");
    return;
  }

  console.log("🚀 Khởi động Smart Locker System...");

  await loadCameraDevices();
  setupEventListeners();
  
  // Gọi fetchLockerSummary ngay khi trang load
  await fetchLockerSummary();
  
  // Cập nhật định kỳ mỗi 5 giây
  setInterval(fetchLockerSummary, 5000);
  
  console.log("✅ Hệ thống đã sẵn sàng!");
}

init();