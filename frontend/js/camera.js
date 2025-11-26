/**
 * camera.js
 * Handle camera initialization, start/stop
 */

import state from './state.js';
import config from './config.js';
import { startProcessing, stopProcessing } from './detection.js';
import { resetCanvas } from './ui.js';

// DOM elements
let video, cameraSelect, startButton, stopButton;

/**
 * Initialize camera module
 * @param {Object} elements - DOM elements
 */
export function initCamera(elements) {
    video = elements.video;
    cameraSelect = elements.cameraSelect;
    startButton = elements.startButton;
    stopButton = elements.stopButton;
}

/**
 * Load available cameras and populate select dropdown
 */
export async function loadCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        if (videoDevices.length === 0) {
            cameraSelect.innerHTML = '<option value="">Không tìm thấy camera</option>';
            return;
        }
        
        cameraSelect.innerHTML = '';
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Lỗi khi tải danh sách camera:', error);
        cameraSelect.innerHTML = '<option value="">Lỗi khi tải camera</option>';
    }
}

/**
 * Start camera with selected settings
 */
export async function startCamera() {
    try {
        const deviceId = cameraSelect.value;
        const useRearCamera = document.getElementById('toggleRearCamera').checked;
        
        // Camera configuration
        let constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        };
        
        // If specific deviceId is provided and rear camera toggle is not checked
        if (deviceId && !useRearCamera) {
            constraints.video.deviceId = { exact: deviceId };
        } 
        // Use rear camera if selected
        else if (useRearCamera) {
            constraints.video.facingMode = { exact: "environment" };
        }
        // Default to front camera
        else {
            constraints.video.facingMode = "user";
        }
        
        // Stop existing stream if running
        if (state.stream) {
            stopCamera();
        }
        
        // Start new stream
        state.stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = state.stream;
        
        // Wait for video to load
        await new Promise(resolve => {
            video.onloadedmetadata = resolve;
        });
        
        await video.play();
        
        // Update canvas dimensions to match video
        const overlay = document.getElementById('overlay');
        overlay.width = video.videoWidth || 640;
        overlay.height = video.videoHeight || 480;
        
        // Update application state
        state.isRunning = true;
        
        // Start processing frames
        startProcessing();
        
        // Update UI
        startButton.disabled = true;
        stopButton.disabled = false;
    } catch (error) {
        console.error('Lỗi khi bắt đầu camera:', error);
        alert('Không thể bắt đầu camera. Vui lòng kiểm tra quyền truy cập.');
    }
}

/**
 * Stop camera and processing
 */

export async function ensureCameraStarted() {
    if (!state.stream) {
        await startCamera();
        await new Promise(resolve => setTimeout(resolve, 500));
    }
}

export function stopCamera() {
    // Stop video stream
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
        video.srcObject = null;
    }
    
    // Stop frame processing
    stopProcessing();
    
    // Stop FPS counter
    if (state.fpsTimerId) {
        clearInterval(state.fpsTimerId);
        state.fpsTimerId = null;
    }
    
    // Reset canvas and state
    resetCanvas();
    state.isRunning = false;
    
    // Update UI
    startButton.disabled = false;
    stopButton.disabled = true;
}