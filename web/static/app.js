// Floating-point exchange panel frontend. No frameworks, no build step.
const $ = (sel) => document.querySelector(sel);
const state = {
  target: "composite",
  precision: "FLOAT32",
  mode: "color",
  exportPrecision: "FLOAT32",
  representation: "both",
  imageMeta: null,
};

function seg(id, key, onChange) {
  const el = $(id);
  el.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    el.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
    state[key] = btn.dataset.v;
    onChange && onChange();
  });
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

function semPill(s) {
  return `<span class="pill sem">${s}</span>`;
}
function dtypePill(d) {
  return `<span class="pill ${d === "FLOAT16" ? "f16" : "f32"}">${d}</span>`;
}

async function loadState() {
  const data = await getJSON("/api/state");
  state.imageMeta = data;
  const plates = $("#plates");
  plates.innerHTML = data.plates.map((p) => `
    <div style="margin-bottom:10px">
      <div><strong>${p.name}</strong></div>
      <div class="muted">显示窗 [${p.display_window.join(", ")}]</div>
      <div class="muted">数据窗 [${p.data_window.join(", ")}] · 原点 (${p.data_window[0]}, ${p.data_window[1]}) · ${p.width}×${p.height}</div>
      <div class="muted">真实采样 ${p.coverage_samples} · 缺采样 ${p.missing_samples}</div>
      <table><tr><th>通道</th><th>类型</th><th>语义</th><th>min</th><th>max</th></tr>
      ${p.channels.map((c) => `<tr><td>${c.name}</td><td>${dtypePill(c.dtype)}</td>
        <td>${semPill(c.semantic)}</td>
        <td class="metric">${c.min !== undefined ? c.min.toFixed(3) : "—"}</td>
        <td class="metric">${c.max !== undefined ? c.max.toFixed(3) : "—"}</td></tr>`).join("")}
      </table>
    </div>`).join("");

  const c = data.composite;
  $("#preview-meta").textContent =
    `拼接窗 [${c.data_window.join(", ")}] · 覆盖 ${c.coverage_samples} · 缺失 ${c.missing_samples} · ` +
    `重叠一致 ${data.stitch_meta.overlap_agreements} 冲突 ${data.stitch_meta.overlap_conflicts}`;
  buildExposureControls(c);
}

function buildExposureControls(composite) {
  const colorChans = composite.channels.filter((c) => c.semantic === "color");
  $("#exposure-controls").innerHTML = colorChans.map((c) => `
    <div class="grid2" style="align-items:center">
      <label>${c.name} 曝光 <span id="exp-${c.name}-v">0.0</span> 档</label>
      <input type="range" min="-6" max="6" step="0.1" value="0" data-channel="${c.name}" class="exp">
    </div>`).join("") +
    `<div class="muted" style="margin-top:6px">深度/未知通道不做色调映射，预览为伪彩，悬停显示原始数值。</div>`;
  document.querySelectorAll("input.exp").forEach((inp) =>
    inp.addEventListener("input", () => {
      $(`#exp-${inp.dataset.channel}-v`).textContent = Number(inp.value).toFixed(1);
      drawPreview();
    }));
}

function previewURL() {
  const p = new URLSearchParams({
    target: state.target, precision: state.precision, mode: state.mode,
  });
  return `/api/preview?${p}`;
}

let imageBitmapInfo = { x0: 0, y0: 0 };
async function drawPreview() {
  const res = await fetch(previewURL());
  const blob = await res.blob();
  const bmp = await createImageBitmap(blob);
  const canvas = $("#canvas");
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bmp, 0, 0);
  // Map canvas pixels back to world coordinates of the target data window.
  const meta = state.imageMeta;
  const target = state.target === "composite" ? meta.composite
    : meta.plates.find((p) => p.name === state.target);
  imageBitmapInfo = { x0: target.data_window[0], y0: target.data_window[1] };
}

const tooltip = $("#tooltip");
$("#canvas").addEventListener("mousemove", async (e) => {
  const rect = e.target.getBoundingClientRect();
  const col = Math.floor((e.clientX - rect.left) * (e.target.width / rect.width));
  const row = Math.floor((e.clientY - rect.top) * (e.target.height / rect.height));
  const wx = imageBitmapInfo.x0 + col;
  const wy = imageBitmapInfo.y0 + row;
  try {
    const s = await getJSON(`/api/sample?target=${state.target}&x=${wx}&y=${wy}`);
    if (!s.inside) { tooltip.style.display = "none"; return; }
    tooltip.innerHTML = `<strong>世界坐标 (${wx}, ${wy})</strong><br>
      <span class="${s.covered ? "ok" : "bad"}">${s.note}</span><br>` +
      Object.entries(s.channels).map(([n, v]) =>
        `<span class="muted">${n} [${v.semantic}]</span>: <code>${v.value}</code>`).join("<br>");
    tooltip.style.display = "block";
    tooltip.style.left = `${e.clientX + 14}px`;
    tooltip.style.top = `${e.clientY + 14}px`;
  } catch { tooltip.style.display = "none"; }
});
$("#canvas").addEventListener("mouseleave", () => (tooltip.style.display = "none"));

seg("#seg-target", "target", drawPreview);
seg("#seg-precision", "precision", drawPreview);
seg("#seg-mode", "mode", drawPreview);
seg("#seg-export-precision", "exportPrecision");
seg("#seg-representation", "representation");

function fmt(n) { return Number(n).toLocaleString(undefined, { maximumFractionDigits: 4 }); }

$("#btn-export").addEventListener("click", async () => {
  const btn = $("#btn-export");
  btn.disabled = true; btn.textContent = "导出中…";
  try {
    const url = `/api/export?precision=${state.exportPrecision}&representation=${state.representation}`;
    const data = await getJSON(url);
    const rows = Object.entries(data).map(([label, r]) => {
      const s = r.stats, rep = r.report;
      const pc = rep.pixel_comparison;
      const q = Object.entries(pc.channels).map(([n, c]) =>
        `<tr><td>${n}</td><td class="metric">${c.max_abs_error !== undefined ? c.max_abs_error.toExponential(2) : "精确"}</td>
         <td>${c.nan_preserved === false ? '<span class="bad">NaN 丢失</span>' : "NaN 保留"}</td>
         <td>${c.inf_sign_preserved === false ? '<span class="bad">Inf 变化</span>' : "Inf 符号保留"}</td></tr>`).join("");
      return `<div style="margin-bottom:12px">
        <strong>${label === "sparse" ? "稀疏窗口表示" : "全幅 + 有效域表示"}</strong>
        <span class="muted">（${s.precision}，存储 ${s.stored_width}×${s.stored_height}）</span>
        <table>
          <tr><th>溢出(>half)</th><td class="metric">${s.overflow_pixels}</td>
              <th>NaN</th><td class="metric">${s.nan_pixels}</td>
              <th>Inf</th><td class="metric">${s.inf_pixels}</td></tr>
          <tr><th>量化像素</th><td class="metric">${fmt(s.quantized_pixels)}</td>
              <th>覆盖一致</th><td>${rep.coverage_equal_to_source ? '<span class="ok">是</span>' : '<span class="bad">否</span>'}</td>
              <th>世界原点</th><td>(${rep.world_origin.join(", ")})</td></tr>
        </table>
        <table><tr><th>通道</th><th>最大绝对误差</th><th>NaN</th><th>Inf</th></tr>${q}</table>
      </div>`;
    }).join("");
    $("#export-report").classList.remove("muted");
    $("#export-report").innerHTML = rows;
    $("#download-row").innerHTML = Object.entries(data).map(([label]) => {
      const file = `export_${label}_${state.exportPrecision}.fim`;
      return `<a href="/api/download?file=${file}"><button>下载 ${file}</button></a>`;
    }).join("");
    await drawPreview();
  } catch (err) {
    $("#export-report").innerHTML = `<span class="bad">${err.message}</span>`;
  } finally {
    btn.disabled = false; btn.textContent = "导出并回读校验";
  }
});

$("#btn-reopen").addEventListener("click", async () => {
  const label = state.representation === "full" ? "full" : "sparse";
  try {
    const r = await getJSON(`/api/reopen?representation=${label}&precision=${state.exportPrecision}`);
    const sig = r.coverage_signature, st = r.image.status_flags;
    $("#export-report").insertAdjacentHTML("beforeend", `
      <div style="margin-top:10px;border-top:1px solid var(--line);padding-top:8px">
        <strong>重新打开文件回读</strong> <span class="muted">${r.meta.full_frame ? "全幅" : "稀疏"}</span>
        <table>
          <tr><th>数据窗原点</th><td>(${sig.origin.join(", ")})</td>
              <th>覆盖/缺失</th><td>${sig.covered} / ${sig.missing}</td></tr>
          <tr><th>溢出标记</th><td class="metric">${st.overflow}</td>
              <th>NaN / Inf</th><td class="metric">${st.nan} / ${st.inf}</td></tr>
        </table>
      </div>`);
  } catch (err) {
    $("#export-report").insertAdjacentHTML("beforeend",
      `<div class="bad" style="margin-top:8px">回读失败: ${err.message}</div>`);
  }
});

$("#btn-acceptance").addEventListener("click", async () => {
  const btn = $("#btn-acceptance");
  btn.disabled = true; btn.textContent = "验收运行中…";
  try {
    const r = await getJSON("/api/acceptance");
    const sum = $("#acceptance-summary");
    sum.innerHTML = `<strong class="${r.all_passed ? "ok" : "bad"}">` +
      `${r.passed}/${r.total} 通过${r.all_passed ? " ✓ 全部断言通过" : " ✗ 存在失败"}</strong>` +
      `<div class="muted" style="margin-top:6px">实测：half 溢出 ${r.measured.overflow_pixels_half} · ` +
      `NaN ${r.measured.nan_pixels_half} · Inf ${r.measured.inf_pixels_half} · ` +
      `量化像素 ${fmt(r.measured.quantized_pixels_half)} · R 通道最大误差 ${r.measured.half_max_abs_error_R.toExponential(2)}</div>`;
    const log = $("#acceptance-log");
    log.style.display = "block";
    log.innerHTML = r.checks.map((c) =>
      `${c.passed ? "✓" : "✗"} ${c.name} ${c.detail}`).join("\n");
  } catch (err) {
    $("#acceptance-summary").innerHTML = `<span class="bad">${err.message}</span>`;
  } finally {
    btn.disabled = false; btn.textContent = "运行程序化验收";
  }
});

loadState().then(drawPreview).catch((err) => {
  $("#plates").innerHTML = `<span class="bad">加载失败: ${err.message}</span>`;
});

// --- depth occlusion, inspection, edit/undo, order swap -------------------
state.inspectTarget = "depth";
seg("#seg-inspect-target", "inspectTarget");

function renderInspect(s) {
  if (!s.inside) { $("#inspect-report").textContent = "该世界坐标不在数据窗内。"; return; }
  const rows = Object.entries(s.channels).map(([n, v]) => {
    const val = v.value === null ? "—" : v.value;
    const flags = [v.semantic, v.role];
    if (v.passthrough) flags.push("透传");
    if (!v.present) flags.push(v.on_grid === false ? "窗内无样" : "缺采样");
    return `<tr><td>${n}</td><td class="metric">${val}</td><td class="muted">${flags.join(" · ")}</td></tr>`;
  }).join("");
  $("#inspect-report").style.display = "block";
  $("#inspect-report").innerHTML =
    `<strong>世界坐标 (${s.world.join(", ")})</strong> · ${s.note} · token ${s.window_token}\n` +
    `<table><tr><th>通道</th><th>原始值</th><th>语义 · 角色 · 状态</th></tr>${rows}</table>`;
}

$("#btn-inspect").addEventListener("click", async () => {
  const x = $("#inspect-x").value, y = $("#inspect-y").value;
  try {
    const r = await getJSON(`/api/inspect?target=${state.inspectTarget}&x=${x}&y=${y}`);
    renderInspect(r.sample);
  } catch (err) { $("#inspect-report").textContent = err.message; }
});

$("#btn-depth").addEventListener("click", async () => {
  try {
    const r = await getJSON("/api/depth");
    const p = r.provenance;
    $("#depth-report").classList.remove("muted");
    $("#depth-report").innerHTML =
      `<strong>景深合成</strong> 数据窗 [${r.summary.data_window.join(", ")}]<br>` +
      `前层胜出 ${p.foreground_pixels} · 后层胜出 ${p.background_pixels} · 缺样 ${p.missing_pixels}<br>` +
      `深度来源 <code>${(r.depth_sources || []).join(", ")}</code> · 透传 <code>${(r.passthrough || []).join(", ")}</code>`;
    state.inspectTarget = "depth";
  } catch (err) { $("#depth-report").textContent = err.message; }
});

$("#btn-legacy").addEventListener("click", async () => {
  try {
    const r = await getJSON("/api/legacy");
    $("#depth-report").classList.remove("muted");
    $("#depth-report").innerHTML =
      `<strong class="${r.all_passed ? "ok" : "bad"}">旧工程回归 ${r.checks.filter(c => c.passed).length}/${r.checks.length} 通过</strong>` +
      `<div class="log" style="margin-top:6px">${r.checks.map(c => `${c.passed ? "✓" : "✗"} ${c.name}`).join("\n")}</div>`;
  } catch (err) { $("#depth-report").textContent = err.message; }
});

$("#btn-edit").addEventListener("click", async () => {
  try {
    const r = await getJSON("/api/edit?action=apply&x=-5&y=0&size=4&value=1.5&base=Z");
    $("#depth-report").innerHTML =
      `<strong>前景局部编辑</strong> 区域 [${r.rect.join(", ")}] · 变更 <code>${JSON.stringify(r.changed)}</code> · 撤销栈 ${r.undo_depth}`;
  } catch (err) { $("#depth-report").textContent = err.message; }
});

$("#btn-undo").addEventListener("click", async () => {
  try {
    const r = await getJSON("/api/edit?action=undo");
    $("#depth-report").innerHTML = `<strong>已撤销</strong> · 撤销栈 ${r.undo_depth}`;
  } catch (err) { $("#depth-report").textContent = err.message; }
});

$("#btn-order-swap").addEventListener("click", async () => {
  try {
    const r = await getJSON("/api/order?action=swap");
    const allMatch = Object.values(r.logical_values_match).every(Boolean);
    $("#depth-report").innerHTML =
      `<strong>交换存储顺序</strong><br>请求顺序 <code>${r.requested_order.join(", ")}</code><br>` +
      `文件顺序 <code>${r.stored_order.join(", ")}</code> · 顺序保持 ${r.order_preserved ? "是" : "否"}<br>` +
      `逐通道逻辑值一致 <span class="${allMatch ? "ok" : "bad"}">${allMatch ? "是" : "否"}</span>` +
      ` · <a href="/api/download?file=${r.download}"><button>下载 ${r.download}</button></a>`;
  } catch (err) { $("#depth-report").textContent = err.message; }
});

$("#btn-crop").addEventListener("click", async () => {
  try {
    const r = await getJSON("/api/edit?action=crop&x0=-20&y0=-15&x1=20&y1=15");
    $("#depth-report").innerHTML =
      `<strong>裁切前景</strong> 数据窗 [${r.data_window.join(", ")}] · 尺寸 ${r.shape[1]}×${r.shape[0]}`;
  } catch (err) { $("#depth-report").textContent = err.message; }
});
