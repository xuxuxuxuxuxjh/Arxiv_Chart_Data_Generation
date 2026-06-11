const defaultPath = "/home/i-xujiahao/arxiv_data/arXiv_src_2107_050.tar/2107.08430";

const els = {
  paperPath: document.getElementById("paperPath"),
  loadBtn: document.getElementById("loadBtn"),
  status: document.getElementById("status"),
  paperTitle: document.getElementById("paperTitle"),
  paperMeta: document.getElementById("paperMeta"),
  paperScroller: document.getElementById("paperScroller"),
  paperPages: document.getElementById("paperPages"),
  pdfFrame: document.getElementById("pdfFrame"),
  sourceView: document.getElementById("sourceView"),
  viewTabs: document.getElementById("viewTabs"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  figureCounter: document.getElementById("figureCounter"),
  figureName: document.getElementById("figureName"),
  figureImage: document.getElementById("figureImage"),
  captionLatex: document.getElementById("captionLatex"),
  references: document.getElementById("references"),
  metadata: document.getElementById("metadata"),
};

const state = {
  paper: null,
  figureIndex: 0,
  view: "paper",
  preloadedImages: new Map(),
};

function setStatus(message) {
  els.status.textContent = message || "";
}

function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function textNode(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  el.textContent = text || "";
  return el;
}

function blockClass(block) {
  const type = block.type || "paragraph";
  if (type === "heading") return `paper-block block-heading level-${block.level || 2}`;
  return `paper-block block-${type}`;
}

function renderBlock(block) {
  const wrapper = document.createElement("div");
  wrapper.className = blockClass(block);
  wrapper.id = `block-${block.id}`;
  wrapper.dataset.blockId = block.id;

  if (block.type === "figure") {
    const images = document.createElement("div");
    images.className = "paper-figure-images";
    (block.images || []).forEach((img) => {
      const image = document.createElement("img");
      image.src = img.src;
      image.alt = img.name || "figure";
      images.appendChild(image);
    });
    wrapper.appendChild(images);
    const caption = document.createElement("div");
    caption.className = "paper-figure-caption";
    caption.textContent = block.caption ? `Figure ${block.figure_index}. ${block.caption}` : `Figure ${block.figure_index}`;
    wrapper.appendChild(caption);
    appendRawDetails(wrapper, block);
    return wrapper;
  }

  if (block.type === "table") {
    if (block.caption) {
      const caption = document.createElement("div");
      caption.className = "paper-table-caption";
      caption.textContent = `Table. ${block.caption}`;
      wrapper.appendChild(caption);
    }
    if (block.table_rows && block.table_rows.length) {
      wrapper.appendChild(renderTable(block.table_rows));
    } else {
      const tableText = document.createElement("div");
      tableText.textContent = block.raw || block.text || "";
      wrapper.appendChild(tableText);
    }
    appendRawDetails(wrapper, block);
    return wrapper;
  }

  wrapper.appendChild(document.createTextNode(block.text || ""));
  if (block.type !== "title" && block.type !== "author" && block.raw && block.raw !== block.text) {
    appendRawDetails(wrapper, block);
  }
  return wrapper;
}

function renderTable(rows) {
  const wrap = document.createElement("div");
  wrap.className = "paper-table-wrap";
  const table = document.createElement("table");
  table.className = "paper-table";
  rows.forEach((row, rowIndex) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => {
      const cellEl = document.createElement(rowIndex === 0 ? "th" : "td");
      cellEl.textContent = cell;
      tr.appendChild(cellEl);
    });
    table.appendChild(tr);
  });
  wrap.appendChild(table);
  return wrap;
}

function appendRawDetails(wrapper, block) {
  if (!block.raw) return;
  const details = document.createElement("details");
  details.className = "raw-details";
  const summary = document.createElement("summary");
  summary.textContent = "raw tex";
  const pre = document.createElement("pre");
  pre.textContent = block.raw;
  details.appendChild(summary);
  details.appendChild(pre);
  wrapper.appendChild(details);
}

function renderPages(paper) {
  clearNode(els.paperPages);
  const pages = paper.document?.pages || [];
  pages.forEach((page) => {
    const pageEl = document.createElement("article");
    pageEl.className = "page";
    pageEl.dataset.page = String(page.number);

    const content = document.createElement("div");
    content.className = "page-content";
    (page.blocks || []).forEach((block) => content.appendChild(renderBlock(block)));
    pageEl.appendChild(content);

    const number = document.createElement("div");
    number.className = "page-number";
    number.textContent = String(page.number);
    pageEl.appendChild(number);

    els.paperPages.appendChild(pageEl);
  });
}

function renderMetadata(fig) {
  clearNode(els.metadata);
  const rows = [
    ["figure_index", fig.figure_index],
    ["image_index", fig.image_index_in_figure],
    ["figure_env", fig.figure_env],
    ["lines", fig.figure_line_start && fig.figure_line_end ? `${fig.figure_line_start}-${fig.figure_line_end}` : ""],
    ["labels", (fig.labels || []).join(", ")],
    ["original_graphics_path", fig.original_graphics_path],
    ["resolved_source", fig.resolved_source_path_rel_to_paper],
    ["includegraphics_options", fig.includegraphics_options],
    ["json", fig.json_name],
  ];

  rows.forEach(([key, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = value === null || value === undefined ? "" : String(value);
    els.metadata.appendChild(dt);
    els.metadata.appendChild(dd);
  });
}

function renderReferences(fig) {
  clearNode(els.references);
  const refs = fig.reference_paragraphs_latex || [];
  if (!refs.length) {
    els.references.appendChild(textNode("div", "empty", "No reference paragraphs"));
    return;
  }
  refs.forEach((ref, idx) => {
    const item = document.createElement("pre");
    item.className = "reference-item";
    item.textContent = `${idx + 1}. ${ref}`;
    els.references.appendChild(item);
  });
}

function updateNav() {
  const total = state.paper?.figures?.length || 0;
  els.prevBtn.disabled = state.figureIndex <= 0;
  els.nextBtn.disabled = state.figureIndex >= total - 1;
  els.figureCounter.textContent = total ? `${state.figureIndex + 1} / ${total}` : "0 / 0";
}

function clearHighlights() {
  document.querySelectorAll(".is-highlighted, .is-figure-match").forEach((el) => {
    el.classList.remove("is-highlighted", "is-figure-match");
  });
}

function applyHighlights(fig, shouldScroll = true) {
  clearHighlights();
  const refIds = fig.reference_block_ids || [];
  refIds.forEach((id) => {
    const block = document.getElementById(`block-${id}`);
    if (block) block.classList.add("is-highlighted");
  });
  if (fig.figure_block_id) {
    const block = document.getElementById(`block-${fig.figure_block_id}`);
    if (block) block.classList.add("is-figure-match");
  }

  if (!shouldScroll || state.view !== "paper") return;
  const targetId = refIds[0] || fig.figure_block_id;
  const target = targetId ? document.getElementById(`block-${targetId}`) : null;
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
  }
}

function renderFigure(shouldScroll = true) {
  const figures = state.paper?.figures || [];
  const fig = figures[state.figureIndex];
  updateNav();
  if (!fig) {
    els.figureImage.removeAttribute("src");
    els.figureName.textContent = "";
    els.captionLatex.textContent = "";
    clearNode(els.references);
    clearNode(els.metadata);
    clearHighlights();
    return;
  }

  els.figureImage.src = fig.image_url;
  els.figureImage.alt = fig.image_name || fig.id;
  els.figureName.textContent = fig.image_name || fig.id;
  els.captionLatex.textContent = fig.caption_latex || "";
  renderReferences(fig);
  renderMetadata(fig);
  applyHighlights(fig, shouldScroll);
}

function preloadFigureImages(figures) {
  state.preloadedImages.clear();
  (figures || []).forEach((fig) => {
    if (!fig.image_url) return;
    const image = new Image();
    image.decoding = "async";
    image.loading = "eager";
    image.src = fig.image_url;
    state.preloadedImages.set(fig.image_url, image);
  });
}

function setView(view) {
  state.view = view;
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });
  els.paperScroller.classList.toggle("pdf-mode", view === "pdf");
  els.paperScroller.classList.toggle("source-mode", view === "source");
  if (view === "paper" && state.paper?.figures?.length) {
    applyHighlights(state.paper.figures[state.figureIndex], false);
  }
}

async function loadPaper(path) {
  setStatus("Loading...");
  els.loadBtn.disabled = true;
  try {
    const response = await fetch(`/api/paper?path=${encodeURIComponent(path)}`);
    const payload = await response.json();
    if (!response.ok || payload.error) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }
    state.paper = payload;
    state.figureIndex = 0;
    preloadFigureImages(payload.figures || []);
    els.paperTitle.textContent = payload.paper_id;
    const summary = payload.summary || {};
    const figCount = payload.figures?.length || 0;
    const stats = payload.document?.stats || {};
    els.paperMeta.textContent = `${figCount} images · ${summary.main_tex_rel_to_paper || "total.tex"} · ${stats.num_blocks || 0} blocks`;
    els.sourceView.textContent = payload.source_tex || "";
    renderPages(payload);
    if (payload.pdf_url) {
      els.pdfFrame.src = payload.pdf_url;
      els.viewTabs.style.display = "";
      document.querySelector('[data-view="pdf"]').disabled = false;
    } else {
      els.pdfFrame.removeAttribute("src");
      document.querySelector('[data-view="pdf"]').disabled = true;
      setView("paper");
    }
    renderFigure(false);
    setStatus(payload.pdf_url ? "Loaded · PDF preview available" : "Loaded · PDF preview unavailable");
  } catch (error) {
    setStatus(error.message);
    state.paper = null;
    renderPages({ document: { pages: [] } });
    renderFigure(false);
  } finally {
    els.loadBtn.disabled = false;
  }
}

function nextFigure(delta) {
  const total = state.paper?.figures?.length || 0;
  if (!total) return;
  state.figureIndex = Math.max(0, Math.min(total - 1, state.figureIndex + delta));
  renderFigure(true);
}

function init() {
  const params = new URLSearchParams(window.location.search);
  const initialPath = params.get("path") || defaultPath;
  els.paperPath.value = initialPath;

  els.loadBtn.addEventListener("click", () => loadPaper(els.paperPath.value.trim()));
  els.paperPath.addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadPaper(els.paperPath.value.trim());
  });
  els.prevBtn.addEventListener("click", () => nextFigure(-1));
  els.nextBtn.addEventListener("click", () => nextFigure(1));
  els.viewTabs.addEventListener("click", (event) => {
    const button = event.target.closest(".tab-btn");
    if (!button || button.disabled) return;
    setView(button.dataset.view);
  });
  window.addEventListener("keydown", (event) => {
    if (event.target === els.paperPath) return;
    if (event.key === "ArrowLeft") nextFigure(-1);
    if (event.key === "ArrowRight") nextFigure(1);
  });

  setView("paper");
  loadPaper(initialPath);
}

init();
