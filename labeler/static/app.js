'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let jobId     = null;
let segments  = [];   // [{id, start, end, text}]
let ws        = null; // WaveSurfer instance
let regions   = null; // RegionsPlugin
let activeId  = null; // segment id em foco

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

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave',  () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

async function handleFile(file) {
  showPanel('progress');
  progressMsg.textContent = 'Enviando arquivo...';

  const form = new FormData();
  form.append('file', file);

  let data;
  try {
    const res  = await fetch('/upload', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    data = await res.json();
  } catch (e) {
    progressMsg.textContent = `Erro: ${e.message}`;
    return;
  }

  jobId    = data.job_id;
  segments = data.segments.map(normalizeSegment);

  progressMsg.textContent = 'Carregando áudio...';
  fileName.textContent    = data.filename;

  await initWaveform();
  renderSegments();
  showPanel('editor');
  btnExport.disabled = false;
}

// ── WaveSurfer ────────────────────────────────────────────────────────────────
async function initWaveform() {
  if (ws) { ws.destroy(); ws = null; regions = null; }

  regions = WaveSurfer.Regions.create({
    dragSelection: { slop: 5 },
  });

  ws = WaveSurfer.create({
    container:    '#waveform',
    waveColor:    '#4a9eff',
    progressColor:'#1a6fcc',
    cursorColor:  '#fff',
    height:       100,
    normalize:    true,
    plugins:      [regions],
  });

  await ws.load(`/audio/${jobId}`);

  ws.on('timeupdate', t => {
    timeDisplay.textContent = `${fmt(t)} / ${fmt(ws.getDuration())}`;
  });

  ws.on('finish', () => { btnPlay.textContent = '▶'; });

  ws.on('seeking', t => {
    const seg = segments.find(s => t >= s.start && t <= s.end);
    if (seg) focusSegment(seg.id, false);
  });

  regions.on('region-clicked', (r, e) => {
    e.stopPropagation();
    focusSegment(Number(r.id), false);
    ws.setTime(r.start);
  });

  regions.on('region-updated', r => {
    const seg = segments.find(s => s.id === Number(r.id));
    if (!seg) return;
    seg.start = round(r.start);
    seg.end   = round(r.end);
    updateSegmentRow(seg.id);
  });

  zoomRange.addEventListener('input', () => ws.zoom(Number(zoomRange.value)));

  drawRegions();
}

function drawRegions() {
  if (!regions) return;
  regions.clearRegions();
  const colors = ['rgba(74,158,255,0.25)', 'rgba(80,220,120,0.25)', 'rgba(255,180,50,0.25)'];
  segments.forEach((seg, i) => {
    regions.addRegion({
      id:      String(seg.id),
      start:   seg.start,
      end:     seg.end,
      content: seg.text.slice(0, 40) || '—',
      color:   colors[i % colors.length],
      drag:    true,
      resize:  true,
    });
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

  row.querySelector('[data-action="play"]').addEventListener('click', () => {
    if (!ws) return;
    ws.setTime(seg.start);
    ws.play();
    ws.once('timeupdate', function check(t) {
      if (t >= seg.end) { ws.pause(); btnPlay.textContent = '▶'; }
      else ws.once('timeupdate', check);
    });
  });

  row.querySelector('[data-action="delete"]').addEventListener('click', () => {
    segments = segments.filter(s => s.id !== seg.id);
    drawRegions();
    renderSegments();
  });

  row.querySelector('.seg-text').addEventListener('input', e => {
    const s = segments.find(s => s.id === seg.id);
    if (s) s.text = e.target.value;
    const r = regions?.getRegions().find(r => r.id === String(seg.id));
    if (r) r.setOptions({ content: s.text.slice(0, 40) || '—' });
  });

  // Time inputs: parse MM:SS.mmm or raw seconds
  row.querySelectorAll('.time-input').forEach(inp => {
    inp.addEventListener('change', () => {
      const s   = segments.find(s => s.id === seg.id);
      if (!s) return;
      const val = parseTime(inp.value);
      if (isNaN(val)) { inp.value = fmt(s[inp.dataset.field]); return; }
      s[inp.dataset.field] = val;
      inp.value = fmt(val);
      const r = regions?.getRegions().find(r => r.id === String(seg.id));
      if (r) r.setOptions({ start: s.start, end: s.end });
    });
  });

  row.addEventListener('click', () => focusSegment(seg.id, true));
  return row;
}

function updateSegmentRow(id) {
  const seg = segments.find(s => s.id === id);
  if (!seg) return;
  const row = segmentsList.querySelector(`[data-id="${id}"]`);
  if (!row) return;
  row.querySelector('[data-field="start"]').value = fmt(seg.start);
  row.querySelector('[data-field="end"]').value   = fmt(seg.end);
}

function focusSegment(id, scroll) {
  activeId = id;
  segmentsList.querySelectorAll('.seg-row').forEach(r => {
    r.classList.toggle('active', Number(r.dataset.id) === id);
  });
  if (scroll) {
    const row = segmentsList.querySelector(`[data-id="${id}"]`);
    row?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// ── Add segment ───────────────────────────────────────────────────────────────
btnAddSeg.addEventListener('click', () => {
  const t   = ws ? ws.getCurrentTime() : 0;
  const dur = ws ? ws.getDuration()    : 10;
  const newSeg = normalizeSegment({
    id: Date.now(),
    start: round(t),
    end:   round(Math.min(t + 3, dur)),
    text: '',
  });
  segments.push(newSeg);
  segments.sort((a, b) => a.start - b.start);
  drawRegions();
  renderSegments();
  focusSegment(newSeg.id, true);
});

// ── Export ────────────────────────────────────────────────────────────────────
btnExport.addEventListener('click', async () => {
  btnExport.disabled = true;
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

// ── Helpers ───────────────────────────────────────────────────────────────────
function showPanel(name) {
  uploadPanel.hidden   = name !== 'upload';
  progressPanel.hidden = name !== 'progress';
  editorPanel.hidden   = name !== 'editor';
}

function fmt(s) {
  if (isNaN(s)) return '0:00.0';
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

function round(n) { return Math.round(n * 1000) / 1000; }
function esc(s)   { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function normalizeSegment(s) { return { id: s.id ?? Date.now(), start: s.start, end: s.end, text: s.text ?? '' }; }
