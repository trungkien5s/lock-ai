// static/js/main.js
import config from "./config.js";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const personCountEl = document.getElementById("personCount");
const peopleListContainer = document.querySelector(
  "#detectedPeopleList .people-list-content"
);

const cameraSelect = document.getElementById("cameraSelect");
const toggleRearCamera = document.getElementById("toggleRearCamera");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");

const enrollButton = document.getElementById("enrollButton");
const unlockButton = document.getElementById("unlockButton");
const userIdInput = document.getElementById("userIdInput");
const userNameInput = document.getElementById("userNameInput");

const lockerStatusText = document.getElementById("lockerStatusText");
const recognizedUserText = document.getElementById("recognizedUserText");
const recognizedConfidenceText = document.getElementById(
  "recognizedConfidenceText"
);
const enrollStatusText = document.getElementById("enrollStatusText");
const enrollProgressText = document.getElementById("enrollProgressText");


let currentStream = null;
let isStreaming = false;
let isProcessing = false;
let lastFrameTime = 0;
const frameInterval = 1000 / config.frameRate;

// ============ Helper: Resize canvas khớp với video ============
function resizeCanvasToVideo() {
  if (!video.videoWidth || !video.videoHeight) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// ============ Lấy danh sách camera ============
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

// ============ Bắt đầu camera ============
async function startCamera() {
  try {
    // Dừng stream cũ nếu có
    stopCamera();

    const selectedDeviceId = cameraSelect ? cameraSelect.value : "";

    let constraints;
    if (config.isMobile) {
      // Trên mobile, dùng facingMode
      constraints = {
        video: {
          facingMode: toggleRearCamera && toggleRearCamera.checked ? "environment" : "user",
        },
        audio: false,
      };
    } else {
      // Desktop: dùng deviceId nếu có
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
  } catch (err) {
    console.error("Không thể bật camera:", err);
    alert("Không thể bật camera. Hãy kiểm tra quyền truy cập camera.");
  }
}

// ============ Dừng camera ============
function stopCamera() {
  isStreaming = false;
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
    currentStream = null;
  }
  if (ctx && canvas) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

// ============ Capture frame hiện tại thành Blob ============
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

// ============ Vẽ bounding box & cập nhật UI thống kê ============
function drawDetections(data) {
  if (!ctx || !canvas) return;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!data) return;

  const { persons, person_boxes = [], face_boxes = [] } = data;

  // Cập nhật thống kê
  if (personCountEl) {
    personCountEl.textContent = persons ?? 0;
  }

  if (peopleListContainer) {
    peopleListContainer.innerHTML = "";
  }

  // Cài đặt style chung
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
      const { coords, confidence, } = box;
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
        faceName = similar_faces[0]; // lấy tên giống nhất
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

// ============ Loop gửi frame lên /process_frame ============
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


// ============ Handler: Đăng ký khuôn mặt (nhiều frame) ============
// ============ Handler: Đăng ký khuôn mặt (nhiều frame + hướng dẫn UI) ============
async function handleEnrollFace() {
  const userId = userIdInput ? userIdInput.value.trim() : "";
  const userName = userNameInput ? userNameInput.value.trim() : "";

  if (!userId || !userName) {
    alert("Vui lòng nhập đầy đủ Mã người dùng / Mã tủ và Tên người dùng.");
    return;
  }

  if (!isStreaming) {
    alert("Hãy bật camera trước khi đăng ký khuôn mặt.");
    return;
  }

  const NUM_FRAMES = 5;      // số khung hình sẽ chụp
  const FRAME_DELAY = 500;   // mỗi 500ms chụp 1 lần

  try {
    // Cập nhật UI trạng thái
    if (enrollStatusText) {
      enrollStatusText.textContent =
        "Đang thu thập khuôn mặt, vui lòng nhìn thẳng, quay trái/phải và mỉm cười nhẹ...";
    }
    if (enrollProgressText) {
      enrollProgressText.textContent = `0/${NUM_FRAMES} khung hình`;
    }

    // Khoá nút đăng ký trong lúc đang chạy để tránh spam
    if (enrollButton) {
      enrollButton.disabled = true;
    }

    const formData = new FormData();
    let capturedCount = 0;

    for (let i = 0; i < NUM_FRAMES; i++) {
      const blob = await captureFrameAsBlob();
      if (blob) {
        formData.append("files", blob, `enroll_${i}.jpg`);
        capturedCount++;

        if (enrollProgressText) {
          enrollProgressText.textContent = `${capturedCount}/${NUM_FRAMES} khung hình`;
        }
      }
      // Chờ một chút để user có thời gian đổi biểu cảm / góc mặt
      await new Promise((resolve) => setTimeout(resolve, FRAME_DELAY));
    }

    if (capturedCount === 0) {
      if (enrollStatusText) {
        enrollStatusText.textContent =
          "Đăng ký thất bại: không chụp được khung hình nào. Hãy thử lại.";
      }
      alert("Không thể chụp được khung hình nào từ camera.");
      return;
    }

    const url =
      `${config.enrollUrl}?user_id=${encodeURIComponent(
        userId
      )}&name=${encodeURIComponent(userName)}`;

    const res = await fetch(url, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      if (enrollStatusText) {
        enrollStatusText.textContent =
          `Đăng ký thất bại: ${data.detail || "Lỗi không xác định"}`;
      }
      alert(`Đăng ký thất bại: ${data.detail || "Lỗi không xác định"}`);
      return;
    }

    if (enrollStatusText) {
      enrollStatusText.textContent =
        `Đăng ký thành công cho ${data.name} (${data.user_id}). Bạn có thể dùng khuôn mặt để mở tủ.`;
    }
    if (enrollProgressText) {
      enrollProgressText.textContent =
        `${capturedCount}/${NUM_FRAMES} khung hình (hoàn tất)`;
    }

    alert(
      `Đăng ký khuôn mặt thành công cho ${data.name} (${data.user_id}) với ${capturedCount} khung hình.`
    );
  } catch (err) {
    console.error("Lỗi khi đăng ký khuôn mặt:", err);
    if (enrollStatusText) {
      enrollStatusText.textContent =
        "Có lỗi xảy ra khi đăng ký khuôn mặt. Hãy thử lại.";
    }
    alert("Có lỗi xảy ra khi đăng ký khuôn mặt.");
  } finally {
    // Mở lại nút đăng ký
    if (enrollButton) {
      enrollButton.disabled = false;
    }
  }
}



// ============ Handler: Mở tủ bằng khuôn mặt ============
async function handleUnlockLocker() {
  if (!isStreaming) {
    alert("Hãy bật camera trước khi mở tủ.");
    return;
  }

  try {
    const blob = await captureFrameAsBlob();
    if (!blob) {
      alert("Không thể chụp hình từ camera.");
      return;
    }

    const formData = new FormData();
    formData.append("file", blob, "unlock.jpg");

    const res = await fetch(config.unlockUrl, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      alert(`Lỗi khi mở tủ: ${data.detail || "Lỗi không xác định"}`);
      return;
    }

    if (lockerStatusText) {
      lockerStatusText.textContent =
        data.status === "granted" ? "ĐÃ MỞ" : "TỪ CHỐI";
    }

    if (recognizedUserText) {
      if (data.name && data.user_id) {
        recognizedUserText.textContent = `${data.name} (${data.user_id})`;
      } else if (data.name) {
        recognizedUserText.textContent = data.name;
      } else {
        recognizedUserText.textContent = "-";
      }
    }

    if (recognizedConfidenceText) {
      if (typeof data.confidence === "number") {
        recognizedConfidenceText.textContent = data.confidence.toFixed(3);
      } else {
        recognizedConfidenceText.textContent = "-";
      }
    }

    if (data.message) {
      alert(data.message);
    }
  } catch (err) {
    console.error("Lỗi khi mở tủ:", err);
    alert("Có lỗi xảy ra khi mở tủ.");
  }
}

// ============ Gắn event listeners ============
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
      if (!config.isMobile) {
        // Desktop: đổi camera thì bật lại
        if (isStreaming) {
          startCamera();
        }
      }
    });
  }

  if (toggleRearCamera) {
    toggleRearCamera.addEventListener("change", () => {
      if (config.isMobile) {
        if (isStreaming) {
          startCamera();
        }
      }
    });
  }

  if (enrollButton) {
    enrollButton.addEventListener("click", handleEnrollFace);
  }

  if (unlockButton) {
    unlockButton.addEventListener("click", handleUnlockLocker);
  }

  window.addEventListener("resize", () => {
    resizeCanvasToVideo();
  });
}

// ============ Khởi động ============
async function init() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Trình duyệt của bạn không hỗ trợ camera (getUserMedia).");
    return;
  }

  await loadCameraDevices();
  setupEventListeners();
}

init();