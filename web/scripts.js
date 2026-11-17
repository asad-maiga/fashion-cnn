"use strict";

const MODEL_PATH = "./models/fashion_cnn.onnx";
const SAMPLE_IMAGE_PATH = "./sample_ankle_boot.png";
const ORT_WASM_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

const IMAGE_SIZE = 28;
const PIXEL_COUNT = IMAGE_SIZE * IMAGE_SIZE;
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

const ALLOWED_MIME_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
  "image/bmp",
]);

const ALLOWED_EXTENSIONS = /\.(png|jpe?g|webp|gif|bmp)$/i;

const FASHION_CLASSES = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
];

const state = {
  session: null,
  modelLoading: false,
  modelError: null,
  inputTensorData: null,
  currentObjectUrl: null,
  previewMode: "original",
  isInferring: false,
  requestToken: 0,
};

const elements = {
  statusDot: document.querySelector(".status-dot"),
  statusText: document.getElementById("model-status-text"),
  dropZone: document.getElementById("drop-zone"),
  dropTrigger: document.getElementById("drop-trigger"),
  fileInput: document.getElementById("file-input"),
  browseBtn: document.getElementById("browse-btn"),
  sampleBtn: document.getElementById("sample-btn"),
  inputError: document.getElementById("input-error"),
  previewSection: document.getElementById("preview-section"),
  viewOriginalBtn: document.getElementById("view-original-btn"),
  viewTensorBtn: document.getElementById("view-tensor-btn"),
  previewMeta: document.getElementById("preview-meta"),
  sourcePreview: document.getElementById("source-preview"),
  tensorPreview: document.getElementById("tensor-preview"),
  clearBtn: document.getElementById("clear-btn"),
  classifyBtn: document.getElementById("classify-btn"),
  inferenceTime: document.getElementById("inference-time"),
  resultEmpty: document.getElementById("result-empty"),
  resultReady: document.getElementById("result-ready"),
  resultLoading: document.getElementById("result-loading"),
  resultError: document.getElementById("result-error"),
  resultErrorText: document.getElementById("result-error-text"),
  retryModelBtn: document.getElementById("retry-model-btn"),
  resultOutput: document.getElementById("result-output"),
  predictedLabel: document.getElementById("predicted-label"),
  predictedConfidence: document.getElementById("predicted-confidence"),
  probabilityList: document.getElementById("probability-list"),
};

function setModelStatus(status, message) {
  elements.statusDot.dataset.state = status;
  elements.statusText.textContent = message;
}

function showInputError(message) {
  elements.inputError.textContent = message;
  elements.inputError.hidden = false;
}

function clearInputError() {
  elements.inputError.textContent = "";
  elements.inputError.hidden = true;
}

function setPreviewMode(mode) {
  state.previewMode = mode;
  const showOriginal = mode === "original";

  elements.viewOriginalBtn.setAttribute("aria-pressed", String(showOriginal));
  elements.viewTensorBtn.setAttribute("aria-pressed", String(!showOriginal));

  elements.sourcePreview.hidden = !showOriginal;
  elements.tensorPreview.hidden = showOriginal;
}

function setResultView(view, errorMessage = "", showRetry = false) {
  elements.resultEmpty.hidden = view !== "empty";
  elements.resultReady.hidden = view !== "ready";
  elements.resultLoading.hidden = view !== "loading";
  elements.resultError.hidden = view !== "error";
  elements.resultOutput.hidden = view !== "output";

  if (view !== "output") {
    elements.inferenceTime.hidden = true;
    elements.inferenceTime.textContent = "";
  }

  if (view === "error") {
    elements.resultErrorText.textContent = errorMessage;
    elements.retryModelBtn.hidden = !showRetry;
  }
}

function syncIdleOutputState() {
  if (state.modelError) {
    setResultView("error", state.modelError, true);
  } else if (state.inputTensorData !== null) {
    setResultView("ready");
  } else {
    setResultView("empty");
  }
}

function updateActionButtons() {
  const hasImage = state.inputTensorData !== null;
  elements.clearBtn.disabled = !hasImage || state.isInferring;
  elements.classifyBtn.disabled =
    !hasImage || !state.session || state.isInferring;
  elements.browseBtn.disabled = state.isInferring;
  elements.sampleBtn.disabled = state.isInferring;
}

function revokeCurrentObjectUrl() {
  if (state.currentObjectUrl) {
    URL.revokeObjectURL(state.currentObjectUrl);
    state.currentObjectUrl = null;
  }
}

function validateImageFile(file) {
  if (!file) {
    return "No file was selected.";
  }

  const hasValidMime = file.type && ALLOWED_MIME_TYPES.has(file.type);
  const hasValidExtension = !file.type && ALLOWED_EXTENSIONS.test(file.name);

  if (!hasValidMime && !hasValidExtension) {
    return "Unsupported file type. Please choose a PNG, JPEG, WebP, GIF, or BMP image.";
  }

  if (file.size === 0) {
    return "The selected file is empty.";
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return "File exceeds the 10 MB limit. Please choose a smaller image.";
  }

  return null;
}

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () =>
      reject(new Error("The selected file could not be decoded as an image."));
    img.src = src;
  });
}

function preprocessImageToTensor(img) {
  const canvas = elements.tensorPreview;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  if (!ctx) {
    throw new Error("2D canvas context is unavailable in this browser.");
  }

  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const isExactSampleSize =
    img.naturalWidth === IMAGE_SIZE && img.naturalHeight === IMAGE_SIZE;

  ctx.imageSmoothingEnabled = !isExactSampleSize;
  if (!isExactSampleSize) {
    ctx.imageSmoothingQuality = "high";
  }

  ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const rgba = imageData.data;
  const floatData = new Float32Array(PIXEL_COUNT);

  for (let i = 0; i < PIXEL_COUNT; i += 1) {
    const offset = i * 4;
    const r = rgba[offset];
    const g = rgba[offset + 1];
    const b = rgba[offset + 2];

    const gray =
      r === g && g === b
        ? r
        : Math.round(0.299 * r + 0.587 * g + 0.114 * b);

    rgba[offset] = gray;
    rgba[offset + 1] = gray;
    rgba[offset + 2] = gray;
    rgba[offset + 3] = 255;

    floatData[i] = gray / 255.0;
  }

  ctx.putImageData(imageData, 0, 0);
  return floatData;
}

function computeSoftmax(logits) {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
    }
  }

  const exps = new Float64Array(logits.length);
  let sum = 0;

  for (let i = 0; i < logits.length; i += 1) {
    const val = Math.exp(logits[i] - maxLogit);
    exps[i] = val;
    sum += val;
  }

  const probabilities = new Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) {
    probabilities[i] = exps[i] / sum;
  }

  return probabilities;
}

function getTopClassIndex(probabilities) {
  let bestIndex = 0;
  for (let i = 1; i < probabilities.length; i += 1) {
    if (probabilities[i] > probabilities[bestIndex]) {
      bestIndex = i;
    }
  }
  return bestIndex;
}

function formatPercent(probability) {
  return `${(probability * 100).toFixed(1)}%`;
}

function renderPrediction(probabilities, durationMs) {
  const topIndex = getTopClassIndex(probabilities);
  const topLabel = FASHION_CLASSES[topIndex];
  const topProb = probabilities[topIndex];

  elements.predictedLabel.textContent = topLabel;
  elements.predictedConfidence.textContent = formatPercent(topProb);
  elements.inferenceTime.textContent = `${durationMs.toFixed(1)} ms`;
  elements.inferenceTime.hidden = false;

  const fragment = document.createDocumentFragment();

  FASHION_CLASSES.forEach((className, index) => {
    const probability = probabilities[index];
    const isTop = index === topIndex;

    const item = document.createElement("li");
    item.className = isTop ? "prob-row is-top" : "prob-row";

    const nameSpan = document.createElement("span");
    nameSpan.className = "prob-name";
    nameSpan.textContent = className;

    const bar = document.createElement("progress");
    bar.className = "prob-bar";
    bar.max = 100;
    bar.value = Number((probability * 100).toFixed(2));
    bar.setAttribute("aria-label", `${className} probability`);

    const valueSpan = document.createElement("span");
    valueSpan.className = "prob-value";
    valueSpan.textContent = formatPercent(probability);

    item.append(nameSpan, bar, valueSpan);
    fragment.appendChild(item);
  });

  elements.probabilityList.replaceChildren(fragment);
  setResultView("output");
}

async function applySelectedImage(objectUrl, displayName) {
  const token = ++state.requestToken;
  clearInputError();

  try {
    const img = await loadImageElement(objectUrl);

    if (token !== state.requestToken) {
      URL.revokeObjectURL(objectUrl);
      return;
    }

    const tensorData = preprocessImageToTensor(img);

    revokeCurrentObjectUrl();
    state.currentObjectUrl = objectUrl;
    state.inputTensorData = tensorData;

    elements.sourcePreview.src = objectUrl;
    elements.sourcePreview.alt = `Preview of ${displayName}`;
    elements.sourcePreview.classList.toggle(
      "is-pixelated",
      img.naturalWidth <= 64 && img.naturalHeight <= 64
    );

    elements.previewMeta.textContent = `${displayName} (${img.naturalWidth}\u00d7${img.naturalHeight})`;
    elements.previewSection.hidden = false;
    setPreviewMode(state.previewMode);

    syncIdleOutputState();
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    showInputError(
      error instanceof Error ? error.message : "Failed to load image."
    );
  } finally {
    updateActionButtons();
  }
}

async function handleFileSelection(file) {
  if (state.isInferring) {
    return;
  }

  const validationError = validateImageFile(file);
  if (validationError) {
    showInputError(validationError);
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  await applySelectedImage(objectUrl, file.name);
}

async function handleLoadSample() {
  if (state.isInferring) {
    return;
  }

  clearInputError();
  elements.sampleBtn.disabled = true;

  try {
    const response = await fetch(SAMPLE_IMAGE_PATH);
    if (!response.ok) {
      throw new Error(`Could not load sample image (HTTP ${response.status}).`);
    }

    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    await applySelectedImage(objectUrl, "sample_ankle_boot.png");
  } catch (error) {
    showInputError(
      error instanceof Error ? error.message : "Failed to fetch the sample image."
    );
  } finally {
    updateActionButtons();
  }
}

function handleClear() {
  if (state.isInferring) {
    return;
  }

  state.requestToken += 1;
  state.inputTensorData = null;
  revokeCurrentObjectUrl();

  elements.fileInput.value = "";
  elements.sourcePreview.removeAttribute("src");
  elements.sourcePreview.classList.remove("is-pixelated");
  elements.previewMeta.textContent = "\u2014";
  elements.previewSection.hidden = true;
  setPreviewMode("original");
  clearInputError();

  syncIdleOutputState();
  updateActionButtons();
}

async function loadOnnxModel() {
  if (state.modelLoading) {
    return;
  }

  state.modelLoading = true;
  state.modelError = null;
  setModelStatus("loading", "Loading model\u2026");

  try {
    if (typeof window.ort === "undefined") {
      throw new Error(
        "ONNX Runtime Web failed to load from the CDN. Check your connection and reload the page."
      );
    }

    window.ort.env.wasm.wasmPaths = ORT_WASM_CDN;
    window.ort.env.wasm.numThreads = 1;

    state.session = await window.ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });

    setModelStatus("ready", "Model ready");
    syncIdleOutputState();
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Unable to initialize ONNX Runtime Web session.";
    state.session = null;
    state.modelError = `Failed to load ${MODEL_PATH}: ${message}`;
    setModelStatus("error", "Model unavailable");
    setResultView("error", state.modelError, true);
  } finally {
    state.modelLoading = false;
    updateActionButtons();
  }
}

async function runInference() {
  if (!state.session || !state.inputTensorData || state.isInferring) {
    return;
  }

  const token = state.requestToken;
  state.isInferring = true;
  updateActionButtons();
  setResultView("loading");

  try {
    const inputName = state.session.inputNames[0] || "images";
    const outputName = state.session.outputNames[0] || "logits";

    const inputTensor = new window.ort.Tensor(
      "float32",
      state.inputTensorData,
      [1, 1, IMAGE_SIZE, IMAGE_SIZE]
    );

    const start = performance.now();
    const outputs = await state.session.run({ [inputName]: inputTensor });
    const durationMs = performance.now() - start;

    if (token !== state.requestToken) {
      return;
    }

    const outputTensor = outputs[outputName];
    if (
      !outputTensor ||
      !outputTensor.data ||
      outputTensor.data.length !== FASHION_CLASSES.length
    ) {
      throw new Error("Unexpected model output shape.");
    }

    const logits = Array.from(outputTensor.data);
    const probabilities = computeSoftmax(logits);
    renderPrediction(probabilities, durationMs);
  } catch (error) {
    if (token !== state.requestToken) {
      return;
    }

    const message =
      error instanceof Error
        ? error.message
        : "An unexpected inference error occurred.";
    setResultView("error", `Inference failed: ${message}`, false);
  } finally {
    state.isInferring = false;
    updateActionButtons();
  }
}

function bindEvents() {
  elements.browseBtn.addEventListener("click", () => {
    elements.fileInput.click();
  });

  elements.dropTrigger.addEventListener("click", () => {
    if (!state.isInferring) {
      elements.fileInput.click();
    }
  });

  elements.fileInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileSelection(file);
    }
  });

  elements.dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (!state.isInferring) {
      elements.dropZone.classList.add("is-dragover");
    }
  });

  elements.dropZone.addEventListener("dragleave", (event) => {
    if (!elements.dropZone.contains(event.relatedTarget)) {
      elements.dropZone.classList.remove("is-dragover");
    }
  });

  elements.dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("is-dragover");

    const file = event.dataTransfer?.files?.[0];
    if (file) {
      handleFileSelection(file);
    }
  });

  elements.viewOriginalBtn.addEventListener("click", () => {
    setPreviewMode("original");
  });

  elements.viewTensorBtn.addEventListener("click", () => {
    setPreviewMode("tensor");
  });

  elements.sampleBtn.addEventListener("click", handleLoadSample);
  elements.clearBtn.addEventListener("click", handleClear);
  elements.classifyBtn.addEventListener("click", runInference);
  elements.retryModelBtn.addEventListener("click", loadOnnxModel);
}

bindEvents();
loadOnnxModel();
