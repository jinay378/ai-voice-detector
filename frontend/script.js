/* ============================================
   AI Voice Detector — Frontend Logic
   ============================================ */

(function () {
    "use strict";

    // ---------- DOM Elements ----------
    const tabUpload = document.getElementById("tabUpload");
    const tabRecord = document.getElementById("tabRecord");
    const panelUpload = document.getElementById("panelUpload");
    const panelRecord = document.getElementById("panelRecord");
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const fileInfo = document.getElementById("fileInfo");
    const fileName = document.getElementById("fileName");
    const fileSize = document.getElementById("fileSize");
    const fileRemove = document.getElementById("fileRemove");
    const recordBtn = document.getElementById("recordBtn");
    const recordHint = document.getElementById("recordHint");
    const recordTimer = document.getElementById("recordTimer");
    const waveformCanvas = document.getElementById("waveformCanvas");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const btnLoader = document.getElementById("btnLoader");
    const resultsCard = document.getElementById("resultsCard");
    const resultBadge = document.getElementById("resultBadge");
    const resultIcon = document.getElementById("resultIcon");
    const resultLabel = document.getElementById("resultLabel");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceFill = document.getElementById("confidenceFill");
    const resultExplanation = document.getElementById("resultExplanation");
    const resetBtn = document.getElementById("resetBtn");

    // ---------- State ----------
    let selectedFile = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let isRecording = false;
    let timerInterval = null;
    let timerSeconds = 0;
    let audioContext = null;
    let analyserNode = null;
    let animFrameId = null;

    // ---------- Tab Switching ----------
    function switchTab(tab) {
        const isUpload = tab === "upload";
        tabUpload.classList.toggle("active", isUpload);
        tabRecord.classList.toggle("active", !isUpload);
        panelUpload.classList.toggle("active", isUpload);
        panelRecord.classList.toggle("active", !isUpload);

        // Reset file selection when switching to record
        if (!isUpload && !isRecording) {
            clearFile();
        }
    }

    tabUpload.addEventListener("click", () => switchTab("upload"));
    tabRecord.addEventListener("click", () => {
        if (!isRecording) switchTab("record");
    });

    // ---------- File Upload ----------
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("audio/")) {
            setFile(file);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files[0]) setFile(fileInput.files[0]);
    });

    fileRemove.addEventListener("click", clearFile);

    function setFile(file) {
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatBytes(file.size);
        fileInfo.classList.remove("hidden");
        dropZone.classList.add("hidden");
        analyzeBtn.disabled = false;
        resultsCard.classList.add("hidden");
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = "";
        fileInfo.classList.add("hidden");
        dropZone.classList.remove("hidden");
        analyzeBtn.disabled = true;
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / 1048576).toFixed(1) + " MB";
    }

    // ---------- Microphone Recording ----------
    recordBtn.addEventListener("click", toggleRecording);

    async function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Set up audio context for waveform visualization
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            analyserNode = audioContext.createAnalyser();
            analyserNode.fftSize = 256;
            source.connect(analyserNode);

            // Start MediaRecorder
            mediaRecorder = new MediaRecorder(stream, { mimeType: getSupportedMime() });
            recordedChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) recordedChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                stream.getTracks().forEach((t) => t.stop());
                const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
                const ext = mediaRecorder.mimeType.includes("webm") ? ".webm" : ".ogg";
                selectedFile = new File([blob], "recording" + ext, { type: blob.type });
                analyzeBtn.disabled = false;
                recordHint.textContent = "Recording saved — click Analyze";
            };

            mediaRecorder.start(200);
            isRecording = true;
            recordBtn.classList.add("recording");
            recordHint.textContent = "Recording… click to stop";
            startTimer();
            drawWaveform();
        } catch (err) {
            recordHint.textContent = "Microphone access denied";
            console.error("Mic error:", err);
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop();
        }
        isRecording = false;
        recordBtn.classList.remove("recording");
        stopTimer();
        cancelAnimationFrame(animFrameId);
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
    }

    function getSupportedMime() {
        const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg"];
        for (const t of types) {
            if (MediaRecorder.isTypeSupported(t)) return t;
        }
        return "";
    }

    // ---------- Timer ----------
    function startTimer() {
        timerSeconds = 0;
        updateTimerDisplay();
        timerInterval = setInterval(() => {
            timerSeconds++;
            updateTimerDisplay();
        }, 1000);
    }

    function stopTimer() {
        clearInterval(timerInterval);
    }

    function updateTimerDisplay() {
        const m = String(Math.floor(timerSeconds / 60)).padStart(2, "0");
        const s = String(timerSeconds % 60).padStart(2, "0");
        recordTimer.textContent = m + ":" + s;
    }

    // ---------- Waveform Visualization ----------
    function drawWaveform() {
        if (!analyserNode) return;
        const canvas = waveformCanvas;
        const ctx = canvas.getContext("2d");
        const bufferLength = analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function draw() {
            animFrameId = requestAnimationFrame(draw);
            analyserNode.getByteTimeDomainData(dataArray);

            ctx.fillStyle = "rgba(10, 10, 15, 0.3)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.lineWidth = 2;
            ctx.strokeStyle = "#a78bfa";
            ctx.beginPath();

            const sliceWidth = canvas.width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * canvas.height) / 2;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                x += sliceWidth;
            }

            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();
        }

        draw();
    }

    // ---------- Analyze ----------
    analyzeBtn.addEventListener("click", analyze);

    async function analyze() {
        if (!selectedFile) return;

        // Show loader
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector(".btn-text").classList.add("hidden");
        btnLoader.classList.remove("hidden");
        resultsCard.classList.add("hidden");

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            const data = await res.json();

            if (data.status === "success") {
                showResult(data);
            } else {
                alert("Error: " + (data.message || "Unknown error"));
            }
        } catch (err) {
            alert("Network error — is the server running?");
            console.error(err);
        } finally {
            analyzeBtn.querySelector(".btn-text").classList.remove("hidden");
            btnLoader.classList.add("hidden");
            analyzeBtn.disabled = false;
        }
    }

    // ---------- Show Result ----------
    function showResult(data) {
        const isHuman = data.classification === "HUMAN";
        const pct = Math.round(data.confidenceScore * 100);

        // Badge
        resultBadge.className = "result-badge " + (isHuman ? "human" : "ai");
        resultIcon.textContent = isHuman ? "✅" : "⚠️";
        resultLabel.textContent = isHuman ? "Human Voice" : "AI-Generated";

        // Confidence
        confidenceValue.textContent = pct + "%";
        confidenceFill.className = "confidence-fill " + (isHuman ? "human" : "ai");
        // Trigger animation
        confidenceFill.style.width = "0%";
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                confidenceFill.style.width = pct + "%";
            });
        });

        // Explanation
        resultExplanation.textContent = data.explanation || "";

        // Show card
        resultsCard.classList.remove("hidden");
        resultsCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // ---------- Reset ----------
    resetBtn.addEventListener("click", () => {
        resultsCard.classList.add("hidden");
        clearFile();
        recordHint.textContent = "Click to start recording";
        recordTimer.textContent = "00:00";
        // Clear canvas
        const ctx = waveformCanvas.getContext("2d");
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    });
})();
