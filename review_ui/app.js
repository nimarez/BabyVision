const state = {
  datasets: [],
  dataset: null,
  cases: [],
  filtered: [],
  currentIndex: 0,
  reviewDraft: {},
};

const els = {
  datasetSelect: document.getElementById("datasetSelect"),
  datasetSummary: document.getElementById("datasetSummary"),
  progressGrid: document.getElementById("progressGrid"),
  metricReviewed: document.getElementById("metricReviewed"),
  metricPromptFail: document.getElementById("metricPromptFail"),
  metricRubricFail: document.getElementById("metricRubricFail"),
  searchInput: document.getElementById("searchInput"),
  typeFilter: document.getElementById("typeFilter"),
  statusFilter: document.getElementById("statusFilter"),
  caseList: document.getElementById("caseList"),
  caseMeta: document.getElementById("caseMeta"),
  caseTitle: document.getElementById("caseTitle"),
  prevButton: document.getElementById("prevButton"),
  nextButton: document.getElementById("nextButton"),
  caseImage: document.getElementById("caseImage"),
  generationPrompt: document.getElementById("generationPrompt"),
  generationTrace: document.getElementById("generationTrace"),
  promptNotes: document.getElementById("promptNotes"),
  questionText: document.getElementById("questionText"),
  groundTruth: document.getElementById("groundTruth"),
  modelAnswer: document.getElementById("modelAnswer"),
  modelRationale: document.getElementById("modelRationale"),
  correctedAnswer: document.getElementById("correctedAnswer"),
  correctedQuestion: document.getElementById("correctedQuestion"),
  rubricNotes: document.getElementById("rubricNotes"),
  severitySelect: document.getElementById("severitySelect"),
  reviewerInput: document.getElementById("reviewerInput"),
  saveButton: document.getElementById("saveButton"),
  saveState: document.getElementById("saveState"),
};

function text(value) {
  if (value === null || value === undefined || value === "") return "";
  if (Array.isArray(value)) return value.join("\n\n");
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function caseKey(item) {
  return item ? String(item.case_id) : "";
}

function currentCase() {
  return state.filtered[state.currentIndex] || null;
}

function reviewFor(item) {
  if (!item) return {};
  const key = caseKey(item);
  return state.reviewDraft[key] || item.review || {};
}

function reviewStatus(item) {
  const review = reviewFor(item);
  const prompt = review.prompt_adherence || "";
  const rubric = review.answer_rubric || "";
  if (!prompt && !rubric) return "unreviewed";
  if (prompt === "fail") return "prompt_fail";
  if (rubric === "fail") return "rubric_fail";
  if (prompt === "unsure" || rubric === "unsure") return "needs_work";
  if (prompt === "pass" && rubric === "pass") return "accepted";
  return "needs_work";
}

function statusBadge(item) {
  const status = reviewStatus(item);
  if (status === "accepted") return { label: "ok", cls: "pass" };
  if (status === "prompt_fail") return { label: "prompt", cls: "fail" };
  if (status === "rubric_fail") return { label: "rubric", cls: "fail" };
  if (status === "needs_work") return { label: "check", cls: "warn" };
  return { label: "open", cls: "" };
}

function collectReview() {
  const item = currentCase();
  if (!item) return null;
  const existing = reviewFor(item);
  return {
    ...existing,
    dataset_id: state.dataset.id,
    case_id: caseKey(item),
    prompt_notes: els.promptNotes.value.trim(),
    rubric_notes: els.rubricNotes.value.trim(),
    corrected_answer: els.correctedAnswer.value.trim(),
    corrected_question: els.correctedQuestion.value.trim(),
    severity: els.severitySelect.value,
    reviewer: els.reviewerInput.value.trim(),
  };
}

function setReviewField(field, value) {
  const item = currentCase();
  if (!item) return;
  const key = caseKey(item);
  state.reviewDraft[key] = { ...reviewFor(item), [field]: value };
  renderSegments();
  renderCaseList();
  setSaveState("Unsaved");
}

function setSaveState(message) {
  els.saveState.textContent = message;
}

function renderDatasets() {
  els.datasetSelect.innerHTML = "";
  for (const dataset of state.datasets) {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = `${dataset.id} (${dataset.count})`;
    els.datasetSelect.appendChild(option);
  }
}

function renderTypeFilter() {
  const selected = els.typeFilter.value;
  const types = [...new Set(state.cases.map((item) => item.type).filter(Boolean))].sort();
  els.typeFilter.innerHTML = '<option value="">All</option>';
  for (const type of types) {
    const option = document.createElement("option");
    option.value = type;
    option.textContent = type;
    els.typeFilter.appendChild(option);
  }
  if (types.includes(selected)) els.typeFilter.value = selected;
}

function updateProgress() {
  const reviewed = state.cases.filter((item) => reviewStatus(item) !== "unreviewed").length;
  const promptFail = state.cases.filter((item) => reviewStatus(item) === "prompt_fail").length;
  const rubricFail = state.cases.filter((item) => reviewStatus(item) === "rubric_fail").length;
  els.metricReviewed.textContent = String(reviewed);
  els.metricPromptFail.textContent = String(promptFail);
  els.metricRubricFail.textContent = String(rubricFail);
  els.datasetSummary.textContent = state.dataset
    ? `${state.cases.length} cases in ${state.dataset.id}`
    : "No dataset";
}

function applyFilters() {
  const query = els.searchInput.value.trim().toLowerCase();
  const type = els.typeFilter.value;
  const status = els.statusFilter.value;
  state.filtered = state.cases.filter((item) => {
    const haystack = [
      item.case_id,
      item.type,
      item.subtype,
      item.source_kind,
      item.question,
      item.answer,
      item.gemini_extracted_answer,
      item.generated_prompt,
    ]
      .map(text)
      .join(" ")
      .toLowerCase();
    if (query && !haystack.includes(query)) return false;
    if (type && item.type !== type) return false;
    if (status && reviewStatus(item) !== status) return false;
    return true;
  });
  if (state.currentIndex >= state.filtered.length) {
    state.currentIndex = Math.max(0, state.filtered.length - 1);
  }
  renderCaseList();
  renderCurrentCase();
}

function renderCaseList() {
  els.caseList.innerHTML = "";
  state.filtered.forEach((item, index) => {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = index === state.currentIndex ? "active" : "";
    button.addEventListener("click", () => {
      saveCurrentInputsToDraft();
      state.currentIndex = index;
      renderCaseList();
      renderCurrentCase();
    });

    const badge = statusBadge(item);
    button.innerHTML = `
      <span class="case-index">${String(item.index).padStart(2, "0")}</span>
      <span class="case-label">
        <strong>${escapeHtml(item.subtype || item.type || "Case")}</strong>
        <span>${escapeHtml(item.type || "")} / ${escapeHtml(item.source_kind || "")}</span>
      </span>
      <span class="badge ${badge.cls}">${badge.label}</span>
    `;
    li.appendChild(button);
    els.caseList.appendChild(li);
  });
}

function renderSegments() {
  const item = currentCase();
  const review = reviewFor(item);
  document.querySelectorAll(".segmented").forEach((group) => {
    const field = group.dataset.field;
    group.querySelectorAll("button").forEach((button) => {
      button.classList.toggle("selected", review[field] === button.dataset.value);
    });
  });
}

function renderCurrentCase() {
  const item = currentCase();
  const hasCase = Boolean(item);
  els.prevButton.disabled = !hasCase || state.currentIndex <= 0;
  els.nextButton.disabled = !hasCase || state.currentIndex >= state.filtered.length - 1;
  els.saveButton.disabled = !hasCase;

  if (!item) {
    els.caseMeta.textContent = "No case selected";
    els.caseTitle.textContent = "No cases match filters";
    els.caseImage.removeAttribute("src");
    return;
  }

  const review = reviewFor(item);
  els.caseMeta.textContent = `${item.type || "Unknown"} / ${item.subtype || "Unknown"} / ${item.source_kind || "unknown"} / ${item.index} of ${state.cases.length}`;
  els.caseTitle.textContent = `Case ${item.case_id}: ${item.subtype || item.type || "Synthetic case"}`;
  els.caseImage.src = item.image_url;
  els.caseImage.alt = `Case ${item.case_id}`;
  els.generationPrompt.textContent = text(item.generated_prompt);
  els.generationTrace.textContent = text(item.reasoning_trace || item.generation_response_text);
  els.questionText.textContent = text(item.question);
  els.groundTruth.textContent = text(item.answer || item.ground_truth);
  els.modelAnswer.textContent = text(item.gemini_extracted_answer || item.extracted_answer);
  els.modelRationale.textContent = text(item.model_result);
  els.promptNotes.value = review.prompt_notes || "";
  els.rubricNotes.value = review.rubric_notes || "";
  els.correctedAnswer.value = review.corrected_answer || "";
  els.correctedQuestion.value = review.corrected_question || "";
  els.severitySelect.value = review.severity || "";
  if (review.reviewer && !els.reviewerInput.value) els.reviewerInput.value = review.reviewer;
  renderSegments();
  setSaveState(review.updated_at ? `Saved ${review.updated_at}` : "Not saved");
}

function saveCurrentInputsToDraft() {
  const item = currentCase();
  if (!item) return;
  const key = caseKey(item);
  state.reviewDraft[key] = collectReview();
}

async function loadDatasets() {
  const response = await fetch("/api/datasets");
  const payload = await response.json();
  state.datasets = payload.datasets || [];
  renderDatasets();
  if (state.datasets.length > 0) {
    const preferred = state.datasets.find((item) => item.id.includes("mixed")) || state.datasets[0];
    els.datasetSelect.value = preferred.id;
    await loadDataset(preferred.id);
  }
}

async function loadDataset(datasetId) {
  const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/cases`);
  const payload = await response.json();
  state.dataset = payload.dataset;
  state.cases = payload.cases || [];
  state.reviewDraft = {};
  state.currentIndex = 0;
  renderTypeFilter();
  updateProgress();
  applyFilters();
}

async function saveReview() {
  saveCurrentInputsToDraft();
  const payload = collectReview();
  if (!payload) return;
  setSaveState("Saving");
  const response = await fetch("/api/reviews", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "save failed" }));
    setSaveState(error.error || "Save failed");
    return;
  }
  const saved = (await response.json()).review;
  const item = currentCase();
  if (item) {
    item.review = saved;
    state.reviewDraft[caseKey(item)] = saved;
  }
  updateProgress();
  renderCaseList();
  renderSegments();
  setSaveState(`Saved ${saved.updated_at}`);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function bindEvents() {
  els.datasetSelect.addEventListener("change", () => loadDataset(els.datasetSelect.value));
  els.searchInput.addEventListener("input", applyFilters);
  els.typeFilter.addEventListener("change", applyFilters);
  els.statusFilter.addEventListener("change", applyFilters);
  els.prevButton.addEventListener("click", () => {
    saveCurrentInputsToDraft();
    state.currentIndex = Math.max(0, state.currentIndex - 1);
    renderCaseList();
    renderCurrentCase();
  });
  els.nextButton.addEventListener("click", () => {
    saveCurrentInputsToDraft();
    state.currentIndex = Math.min(state.filtered.length - 1, state.currentIndex + 1);
    renderCaseList();
    renderCurrentCase();
  });
  els.saveButton.addEventListener("click", saveReview);
  document.querySelectorAll(".segmented button").forEach((button) => {
    button.addEventListener("click", () => {
      const field = button.closest(".segmented").dataset.field;
      setReviewField(field, button.dataset.value);
    });
  });
  [els.promptNotes, els.rubricNotes, els.correctedAnswer, els.correctedQuestion, els.severitySelect].forEach((el) => {
    el.addEventListener("input", () => {
      saveCurrentInputsToDraft();
      setSaveState("Unsaved");
    });
    el.addEventListener("change", () => {
      saveCurrentInputsToDraft();
      setSaveState("Unsaved");
    });
  });
}

bindEvents();
loadDatasets().catch((error) => {
  els.datasetSummary.textContent = error.message;
  setSaveState("Load failed");
});
