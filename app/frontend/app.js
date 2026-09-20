// IRONSIGHT Vision QC — frontend logic
// Talks to the FastAPI backend at /api/predict, /api/stats, /api/chat

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const scanSweep = document.getElementById('scanSweep');
const stampPass = document.getElementById('stampPass');
const stampFail = document.getElementById('stampFail');
const verdictLabel = document.getElementById('verdictLabel');
const confidenceText = document.getElementById('confidenceText');
const latencyText = document.getElementById('latencyText');
const backendText = document.getElementById('backendText');
const sampleBtn = document.getElementById('sampleBtn');
const statusPill = document.getElementById('statusPill');
const statusText = document.getElementById('statusText');

const statInspected = document.getElementById('statInspected');
const statDefectRate = document.getElementById('statDefectRate');
const statTimeSaved = document.getElementById('statTimeSaved');
const historyList = document.getElementById('historyList');

const chatLog = document.getElementById('chatLog');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const chatQuick = document.getElementById('chatQuick');

// ---------------- Upload / dropzone ----------------

function openFileDialog() { fileInput.click(); }
dropzone.addEventListener('click', openFileDialog);
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openFileDialog(); }
});

['dragenter', 'dragover'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
});
['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
});
dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

let lastLabel = null;

async function handleFile(file) {
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  dropzone.classList.add('has-image');
  stampPass.classList.remove('slam');
  stampFail.classList.remove('slam');
  stampPass.style.opacity = 0;
  stampFail.style.opacity = 0;

  verdictLabel.textContent = 'SCANNING…';
  verdictLabel.className = 'verdict-label';
  scanSweep.classList.remove('active');
  void scanSweep.offsetWidth; // restart animation
  scanSweep.classList.add('active');

  const formData = new FormData();
  formData.append('file', file);

  const start = performance.now();
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    // Let the ~1.1s scan animation read before revealing the verdict
    setTimeout(() => showVerdict(data), 750);
  } catch (err) {
    verdictLabel.textContent = 'ERROR';
    verdictLabel.classList.add('fail');
    confidenceText.textContent = 'Could not reach the inspection model.';
    console.error(err);
  }
}

function showVerdict(data) {
  const isPass = data.verdict === 'PASS';
  verdictLabel.textContent = isPass ? 'PASS' : 'DEFECT DETECTED';
  verdictLabel.className = 'verdict-label ' + (isPass ? 'pass' : 'fail');
  confidenceText.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;
  latencyText.textContent = `${data.inference_ms.toFixed(1)} ms inference`;
  backendText.textContent = data.model_backend === 'keras' ? 'Keras backend' : (data.model_backend === 'tflite' ? 'TFLite (edge) backend' : 'no model');

  const stamp = isPass ? stampPass : stampFail;
  stamp.classList.add('slam');

  lastLabel = data.verdict;
  refreshStats();
}

// Pull a real, random image from the dataset's test split (def_front or
// ok_front) via the backend, instead of using a fabricated placeholder.
// A cache-busting timestamp query param + { cache: 'no-store' } ensure
// the browser actually re-fetches on every click, rather than reusing
// the very first response it ever got for this URL.
sampleBtn.addEventListener('click', async () => {
  try {
    const res = await fetch('/api/sample_image?t=' + Date.now(), { cache: 'no-store' });
    if (!res.ok) throw new Error('No sample image available.');
    const blob = await res.blob();
    const file = new File([blob], 'sample.jpeg', { type: blob.type || 'image/jpeg' });
    handleFile(file);
  } catch (err) {
    console.error('sample image fetch failed', err);
  }
});

// ---------------- Stats polling ----------------

async function refreshStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();

    animateCount(statInspected, parseInt(statInspected.textContent) || 0, data.inspected);
    statDefectRate.textContent = `${(data.defect_rate * 100).toFixed(1)}%`;
    statTimeSaved.textContent = `${data.time_saved_pct}%`;

    if (data.history.length) {
      historyList.innerHTML = '';
      data.history.slice().reverse().forEach(h => {
        const li = document.createElement('li');
        li.className = h.label === 'ok_front' ? 'pass' : 'fail';
        const t = new Date(h.ts);
        li.innerHTML = `<span>${h.label === 'ok_front' ? 'PASS' : 'DEFECT'}</span><span>${(h.confidence*100).toFixed(1)}%</span>`;
        historyList.appendChild(li);
      });
    }

    if (statusPill && data.model_backend && data.model_backend !== 'none') {
      statusPill.classList.add('online');
      statusText.textContent = data.model_backend === 'tflite' ? 'edge model online (TFLite)' : 'model online';
    }
  } catch (err) {
    console.error('stats fetch failed', err);
  }
}

function animateCount(el, from, to) {
  const duration = 400;
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const value = Math.round(from + (to - from) * progress);
    el.textContent = value;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.model_backend && data.model_backend !== 'none') {
      statusPill.classList.add('online');
      statusText.textContent = data.model_backend === 'tflite' ? 'edge model online (TFLite)' : 'model online';
    } else {
      statusText.textContent = 'model not trained yet';
    }
  } catch (err) {
    statusText.textContent = 'backend unreachable';
  }
}

async function loadModelReport() {
  try {
    const res = await fetch('/api/model_report');
    const data = await res.json();
    if (data.available && data.test_metrics) {
      const acc = (data.test_metrics.accuracy * 100).toFixed(1);
      document.getElementById('statAccuracy').textContent = `${acc}%`;
    }
  } catch (err) {
    console.error('model report fetch failed', err);
  }
}

checkHealth();
refreshStats();
loadModelReport();
setInterval(refreshStats, 8000);

// ---------------- Chat ----------------

function appendMessage(text, who) {
  const wrap = document.createElement('div');
  wrap.className = `msg msg-${who}`;
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return wrap;
}

function appendTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg-bot';
  wrap.innerHTML = `<div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return wrap;
}

async function sendChat(message) {
  appendMessage(message, 'user');
  const typing = appendTyping();
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    typing.remove();
    appendMessage(data.reply, 'bot');
  } catch (err) {
    typing.remove();
    appendMessage("I couldn't reach the backend just now — check that the API server is running.", 'bot');
  }
}

chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const val = chatInput.value.trim();
  if (!val) return;
  chatInput.value = '';
  sendChat(val);
});

chatQuick.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-q]');
  if (btn) sendChat(btn.getAttribute('data-q'));
});
