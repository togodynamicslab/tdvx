'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let jobId        = null;
let segments     = [];       // [{id, start, end, text}]
let ws           = null;     // WaveSurfer instance
let wsRegions    = null;     // RegionsPlugin instance
let regMap       = {};       // id → Region object
let activeId     = null;
let playingSegId = null;     // id do segmento tocando agora

// ── DOM refs ──────────────────────────────────────────────────────────────────
const uploadPanel   = document.getElementById('upload-panel');
const progressPanel = document.getElementById('progress-panel');
const editorPanel   = document.getElementById('editor-panel');
const progressMsg   = document.getElementById('progress-msg');
const fileInput     = document.getElementById('file-input');
const dropZone      = document.getElementById('drop-zone');
const btnPlay       = document.getElementById('btn-play');
const btnExport     = document.getElementById('btn-export');
const btnAddSeg     = document.getElementById('btn-add-seg');
const timeDisplay   = document.getElementById('time-display');
const zoomRange     = document.getElementById('zoom-range');
const fileName      = document.getElementById('file-name');
const segCount      = document.getElementById('seg-count');
const segmentsList  = document.getElementById('segments-list');
const exportToast   = document.getElementById('export-toast');
const exportCount   = document.getElementById('export-count');
const exportPath    = document.getElementById('export-path');

// ── Upload ────────────────────────────────────────────────────────────────────
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

const STATUS_MSG = {
  pending:      'Aguardando...',
  extracting:   'Extraindo áudio (ffmpeg)...',
  transcribing: 'Transcrevendo com TDvX...',
};

async function handleFile(file) {
  showPanel('progress');
  progressMsg.textContent = `Enviando "${file.name}"...`;

  const form = new FormData();
  form.append('file', file);

  let data;
  try {
    const res = await fetch('/upload', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    data = await res.json();
  } catch (e) {
    progressMsg.textContent = `Erro: ${e.message}`;
    return;
  }

  jobId = data.job_id;

  // Polling até o job terminar
  const segs = await pollStatus(jobId, data.filename);
  if (!segs) return; // erro já mostrado

  segments = segs.map(normalizeSegment);
  fileName.textContent = data.filename;
  renderSegments();
  showPanel('editor');
  btnExport.disabled = false;

  initWaveform().catch(e => showWaveformError(e.message));
}

async function pollStatus(id, name) {
  const start = Date.now();
  while (true) {
    await new Promise(r => setTimeout(r, 1500));
    let s;
    try {
      const res = await fetch(`/status/${id}`);
      s = await res.json();
    } catch (e) {
      progressMsg.textContent = `Erro de rede: ${e.message}`;
      return null;
    }
    if (s.status === 'error') {
      progressMsg.textContent = `Erro: ${s.error}`;
      return null;
    }
    if (s.status === 'done') return s.segments;
    const elapsed = Math.round((Date.now() - start) / 1000);
    progressMsg.textContent = `${STATUS_MSG[s.status] || s.status} (${elapsed}s)`;
  }
}

// ── WaveSurfer v7 ─────────────────────────────────────────────────────────────
async function initWaveform() {
  if (ws) { ws.destroy(); ws = null; wsRegions = null; regMap = {}; }

  if (typeof WaveSurfer === 'undefined') throw new Error('WaveSurfer não carregou (verifique conexão)');

  // Regions plugin — v7 UMD expõe via WaveSurfer.Regions ou window.RegionsPlugin
  const RegPlugin = (WaveSurfer.Regions || window.RegionsPlugin);
  if (!RegPlugin) throw new Error('RegionsPlugin não encontrado');

  wsRegions = RegPlugin.create();

  ws = WaveSurfer.create({
    container:    '#waveform',
    waveColor:    '#4a9eff',
    progressColor:'#1a6fcc',
    cursorColor:  '#fff',
    height:       100,
    normalize:    true,
    plugins:      [wsRegions],
  });

  ws.on('timeupdate', t => {
    timeDisplay.textContent = `${fmt(t)} / ${fmt(ws.getDuration())}`;
  });
  ws.on('finish',  () => { btnPlay.textContent = '▶'; });
  ws.on('seeking', t => {
    const seg = segments.find(s => t >= s.start && t <= s.end);
    if (seg) focusSegment(seg.id, false);
  });

  wsRegions.on('region-clicked', (r, e) => {
    e.stopPropagation();
    focusSegment(Number(r.id), false);
    ws.setTime(r.start);
  });

  wsRegions.on('region-updated', r => {
    const seg = segments.find(s => s.id === Number(r.id));
    if (!seg) return;
    seg.start = round(r.start);
    seg.end   = round(r.end);
    updateSegmentRow(seg.id);
  });

  zoomRange.addEventListener('input', () => ws.zoom(Number(zoomRange.value)));

  await ws.load(`/audio/${jobId}`);
  drawRegions();
}

function showWaveformError(msg) {
  const wrap = document.getElementById('waveform-wrap');
  if (wrap) wrap.innerHTML = `<p style="color:#e05252;padding:16px;font-size:13px">⚠ Waveform: ${msg}</p>`;
}

function drawRegions() {
  if (!wsRegions) return;
  wsRegions.clearRegions();
  regMap = {};
  const colors = ['rgba(74,158,255,0.25)', 'rgba(80,220,120,0.25)', 'rgba(255,180,50,0.25)'];
  segments.forEach((seg, i) => {
    const r = wsRegions.addRegion({
      id:      String(seg.id),
      start:   seg.start,
      end:     seg.end,
      content: seg.text.slice(0, 40) || '—',
      color:   colors[i % colors.length],
      drag:    true,
      resize:  true,
    });
    regMap[seg.id] = r;
  });
}

// ── Player controls ───────────────────────────────────────────────────────────
btnPlay.addEventListener('click', () => {
  if (!ws) return;
  ws.playPause();
  btnPlay.textContent = ws.isPlaying() ? '⏸' : '▶';
});

// ── Segments rendering ────────────────────────────────────────────────────────
function renderSegments() {
  segmentsList.innerHTML = '';
  segments.forEach(seg => segmentsList.appendChild(buildRow(seg)));
  segCount.textContent = `${segments.length} segmento${segments.length !== 1 ? 's' : ''}`;
}

function buildRow(seg) {
  const row = document.createElement('div');
  row.className = 'seg-row';
  row.dataset.id = seg.id;

  row.innerHTML = `
    <div class="seg-time">
      <input class="time-input" data-field="start" value="${fmt(seg.start)}" title="Início" />
      <span>→</span>
      <input class="time-input" data-field="end"   value="${fmt(seg.end)}"   title="Fim" />
    </div>
    <textarea class="seg-text" rows="2">${esc(seg.text)}</textarea>
    <div class="seg-actions">
      <button class="btn-icon-sm" data-action="play"   title="Ouvir">▶</button>
      <button class="btn-icon-sm" data-action="delete" title="Remover">🗑</button>
    </div>
  `;

  row.querySelector('[data-action="play"]').addEventListener('click', e => {
    if (!ws) return;
    const btn = e.currentTarget;
    if (playingSegId === seg.id && ws.isPlaying()) {
      ws.pause();
      btn.textContent = '▶';
      playingSegId = null;
      return;
    }
    // Para qualquer segmento tocando antes
    if (playingSegId !== null) {
      const prev = segmentsList.querySelector(`[data-id="${playingSegId}"] [data-action="play"]`);
      if (prev) prev.textContent = '▶';
    }
    playingSegId = seg.id;
    btn.textContent = '⏸';
    ws.setTime(seg.start);
    ws.play();
    const check = t => {
      if (t >= seg.end) {
        ws.pause();
        btn.textContent = '▶';
        playingSegId = null;
      } else {
        ws.once('timeupdate', check);
      }
    };
    ws.once('timeupdate', check);
  });

  row.querySelector('[data-action="delete"]').addEventListener('click', () => {
    if (regMap[seg.id]) { regMap[seg.id].remove(); delete regMap[seg.id]; }
    segments = segments.filter(s => s.id !== seg.id);
    renderSegments();
  });

  row.querySelector('.seg-text').addEventListener('input', e => {
    const s = segments.find(s => s.id === seg.id);
    if (!s) return;
    s.text = e.target.value;
    if (regMap[seg.id]) regMap[seg.id].setOptions({ content: s.text.slice(0, 40) || '—' });
  });

  row.querySelectorAll('.time-input').forEach(inp => {
    inp.addEventListener('change', () => {
      const s = segments.find(s => s.id === seg.id);
      if (!s) return;
      const val = parseTime(inp.value);
      if (isNaN(val)) { inp.value = fmt(s[inp.dataset.field]); return; }
      s[inp.dataset.field] = val;
      inp.value = fmt(val);
      if (regMap[seg.id]) regMap[seg.id].setOptions({ start: s.start, end: s.end });
    });
  });

  row.addEventListener('click', () => focusSegment(seg.id, true));
  return row;
}

function updateSegmentRow(id) {
  const seg = segments.find(s => s.id === id);
  const row = segmentsList.querySelector(`[data-id="${id}"]`);
  if (!seg || !row) return;
  row.querySelector('[data-field="start"]').value = fmt(seg.start);
  row.querySelector('[data-field="end"]').value   = fmt(seg.end);
}

function focusSegment(id, scroll) {
  activeId = id;
  segmentsList.querySelectorAll('.seg-row').forEach(r => {
    r.classList.toggle('active', Number(r.dataset.id) === id);
  });
  if (scroll) {
    segmentsList.querySelector(`[data-id="${id}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// ── Add segment ───────────────────────────────────────────────────────────────
btnAddSeg.addEventListener('click', () => {
  const t   = ws?.getCurrentTime() ?? 0;
  const dur = ws?.getDuration()    ?? 10;
  const newSeg = normalizeSegment({ id: Date.now(), start: round(t), end: round(Math.min(t + 3, dur)), text: '' });
  segments.push(newSeg);
  segments.sort((a, b) => a.start - b.start);
  drawRegions();
  renderSegments();
  focusSegment(newSeg.id, true);
});

// ── Export ────────────────────────────────────────────────────────────────────
btnExport.addEventListener('click', async () => {
  btnExport.disabled    = true;
  btnExport.textContent = 'Exportando...';
  try {
    const res  = await fetch(`/export/${jobId}`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ segments }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    exportCount.textContent = data.exported;
    exportPath.textContent  = data.path;
    exportToast.hidden      = false;
  } catch (e) {
    alert(`Erro ao exportar: ${e.message}`);
  } finally {
    btnExport.disabled    = false;
    btnExport.textContent = 'Exportar segmentos';
  }
});

// ── Utils ─────────────────────────────────────────────────────────────────────
function showPanel(name) {
  uploadPanel.hidden   = name !== 'upload';
  progressPanel.hidden = name !== 'progress';
  editorPanel.hidden   = name !== 'editor';
}

function fmt(s) {
  if (s == null || isNaN(s)) return '0:00.0';
  const m   = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, '0');
  return `${m}:${sec}`;
}

function parseTime(str) {
  str = str.trim();
  if (/^\d+(\.\d+)?$/.test(str)) return parseFloat(str);
  const m = str.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (m) return parseInt(m[1], 10) * 60 + parseFloat(m[2]);
  return NaN;
}

function round(n)             { return Math.round(n * 1000) / 1000; }
function esc(s)               { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function normalizeSegment(s)  { return { id: s.id ?? Date.now(), start: s.start ?? 0, end: s.end ?? 1, text: s.text ?? '' }; }
