const form = document.getElementById("cad-form");
const resultCard = document.getElementById("result");
const meterFill = document.getElementById("meter-fill");
const riskPercent = document.getElementById("risk-percent");
const riskLevel = document.getElementById("risk-level");
const resultStage = document.querySelector(".result-stage");
const resultFloatCard = document.getElementById("result-float-card");
const floatingRiskEyebrow = document.getElementById("floating-risk-eyebrow");
const floatingRiskLevel = document.getElementById("floating-risk-level");
const floatingRiskCopy = document.getElementById("floating-risk-copy");
const floatingAdvice = document.getElementById("floating-advice");
const scoreDistance = document.getElementById("score-distance");
const resultCopy = document.getElementById("result-copy");
const whyScore = document.getElementById("why-score");
const factorList = document.getElementById("factor-list");
const suggestionList = document.getElementById("suggestion-list");
const missingCopy = document.getElementById("missing-copy");
const editAnswers = document.getElementById("edit-answers");
const tryAgain = document.getElementById("try-again");
const backToForm = document.getElementById("back-to-form");
const guideModal = document.getElementById("guide-modal");
const adviceModal = document.getElementById("advice-modal");
const comparisonModal = document.getElementById("comparison-modal");
const adviceIntro = document.getElementById("advice-intro");
const adviceList = document.getElementById("advice-list");
const comparisonIntro = document.getElementById("comparison-intro");
const comparisonList = document.getElementById("comparison-list");
const comparisonNotes = document.getElementById("comparison-notes");
const openGuideButtons = [
  document.getElementById("open-guide-hero"),
  document.getElementById("open-guide-form"),
];
const openComparisonButtons = [
  document.getElementById("open-comparison-hero"),
  document.getElementById("open-comparison-form"),
];
const closeGuide = document.getElementById("close-guide");
const openAdvice = document.getElementById("open-advice");
const closeAdvice = document.getElementById("close-advice");
const closeComparison = document.getElementById("close-comparison");

let latestPrediction = null;

const adviceSources = [
  {
    key: "blood_pressure",
    title: "Manage blood pressure",
    source: "American Heart Association",
    url: "https://www.heart.org/en/professional/workplace-health/lifes-simple-7",
    action: "Track readings, reduce salt where appropriate, stay active, and review persistent high readings with a clinician.",
  },
  {
    key: "cholesterol",
    title: "Control cholesterol",
    source: "American Heart Association",
    url: "https://www.heart.org/en/healthy-living/healthy-lifestyle/my-life-check--lifes-simple-7/ls7-cholesterol-infographic",
    action: "Discuss lipid results, improve dietary fat quality, add fiber-rich foods, and follow medication advice if prescribed.",
  },
  {
    key: "smoking",
    title: "Quit smoking or nicotine use",
    source: "CDC",
    url: "https://www.cdc.gov/tobacco/about/cigarettes-and-cardiovascular-disease.html",
    action: "Use quit support, counseling, and approved stop-smoking aids. Even low cigarette use can affect cardiovascular risk.",
  },
  {
    key: "diabetes",
    title: "Manage diabetes and heart risk",
    source: "CDC",
    url: "https://www.cdc.gov/diabetes/diabetes-complications/diabetes-and-your-heart.html",
    action: "Review glucose results, diabetes status, blood pressure, cholesterol, weight, and medication plan with a healthcare professional.",
  },
  {
    key: "glucose",
    title: "Follow up on fasting glucose",
    source: "CDC",
    url: "https://www.cdc.gov/diabetes/diabetes-testing/prediabetes-a1c-test.html",
    action: "Confirm high or borderline values with appropriate testing and ask how glucose management affects heart risk.",
  },
  {
    key: "bmi",
    title: "Work toward healthier weight",
    source: "CDC",
    url: "https://www.cdc.gov/healthy-weight-growth/index.html",
    action: "Focus on sustainable eating, movement, sleep, and follow-up rather than quick weight loss. Even modest progress can support BP, glucose, and cholesterol.",
  },
  {
    key: "activity",
    title: "Increase heart-healthy activity",
    source: "American Heart Association",
    url: "https://www.heart.org/en/professional/workplace-health/lifes-simple-7",
    action: "Choose regular movement you can repeat, such as brisk walking, cycling, or similar moderate activity if safe for you.",
  },
  {
    key: "lifestyle",
    title: "Maintain healthy daily habits",
    source: "American Heart Association",
    url: "https://www.heart.org/en/professional/workplace-health/lifes-simple-7",
    action: "Use Life's Essential 8 as a maintenance checklist: food, activity, nicotine, sleep, weight, cholesterol, glucose, and blood pressure.",
  },
  {
    key: "family_history",
    title: "Share family history",
    source: "CDC",
    url: "https://www.cdc.gov/heart-disease/risk-factors/index.html",
    action: "Family history cannot be changed, but it can guide earlier screening and prevention conversations.",
  },
  {
    key: "clinical_followup",
    title: "Arrange clinical follow-up",
    source: "CDC",
    url: "https://www.cdc.gov/heart-disease/about/index.html",
    action: "A high screening result should be reviewed with a healthcare professional, especially if symptoms or multiple risk factors are present.",
  },
];

function formToPayload(formElement) {
  const formData = new FormData(formElement);
  return Object.fromEntries(formData.entries());
}

function updateRangeMessage(input) {
  input.setCustomValidity("");
  if (input.validity.rangeUnderflow || input.validity.rangeOverflow) {
    input.setCustomValidity(input.dataset.rangeMessage || "Please enter a value inside the allowed range.");
  }
}

form.querySelectorAll("input[type='number']").forEach((input) => {
  input.addEventListener("input", () => updateRangeMessage(input));
  input.addEventListener("invalid", () => updateRangeMessage(input));
});

function riskSummary(level, percent) {
  if (level === "High") {
    return `The model estimated a higher CAD risk at ${percent}%. A clinical follow-up would be appropriate.`;
  }
  if (level === "Moderate") {
    return `The model estimated a moderate CAD risk at ${percent}%. It may be useful to review the person's blood pressure, cholesterol, and lifestyle factors.`;
  }
  return `The model estimated a lower CAD risk at ${percent}%, although lower risk does not mean zero risk.`;
}

function scoreDistanceText(level, percent) {
  const lowBoundary = 33;
  const highBoundary = 66;
  const score = Number(percent);

  if (level === "Low") {
    const distance = Math.max(0, lowBoundary - score).toFixed(2);
    return `${distance} percentage points below Moderate risk.`;
  }
  if (level === "Moderate") {
    const fromLow = Math.max(0, score - lowBoundary).toFixed(2);
    const toHigh = Math.max(0, highBoundary - score).toFixed(2);
    return `${fromLow} points above Low, ${toHigh} points below High.`;
  }
  const distance = Math.max(0, score - highBoundary).toFixed(2);
  return `${distance} percentage points above the High risk cutoff.`;
}

function riskStageContent(level, percent) {
  if (level === "High") {
    return {
      className: "high-stage",
      eyebrow: "Priority action",
      title: "High range",
      copy: `This ${percent}% score is above the high cutoff. View the advice list and focus first on high-impact factors.`,
      button: "View high-risk advice",
    };
  }
  if (level === "Moderate") {
    return {
      className: "moderate-stage",
      eyebrow: "Improve target",
      title: "Moderate range",
      copy: `This ${percent}% score sits between Low and High. The advice list prioritizes factors that could move the score down.`,
      button: "View improvement advice",
    };
  }
  return {
    className: "low-stage",
    eyebrow: "Maintain range",
    title: "Low range",
    copy: `This ${percent}% score is below Moderate. Use the advice list to maintain healthy numbers and avoid preventable risk.`,
    button: "View prevention advice",
  };
}

function updateRiskStage(level, percent) {
  const content = riskStageContent(level, percent);
  resultStage.classList.remove("low-stage", "moderate-stage", "high-stage");
  resultStage.classList.add(content.className);
  floatingRiskEyebrow.textContent = content.eyebrow;
  floatingRiskLevel.textContent = content.title;
  floatingRiskCopy.textContent = content.copy;
  floatingAdvice.textContent = content.button;
}

function scrollToForm() {
  form.scrollIntoView({ behavior: "smooth", block: "start" });
}

function clearResult() {
  latestPrediction = null;
  resultCard.classList.add("hidden");
  meterFill.style.width = "0%";
  scoreDistance.textContent = "";
  whyScore.textContent = "";
  factorList.innerHTML = "";
  suggestionList.innerHTML = "";
  missingCopy.textContent = "";
}

function factorInfluence(item, totalContribution) {
  if (totalContribution === 0) {
    return 0;
  }
  return Math.round((Math.abs(item.contribution) / totalContribution) * 100);
}

function factorEffect(item) {
  return item.effect || (item.contribution >= 0 ? "Raises predicted risk" : "Lowers predicted risk");
}

function renderFactors(topFactors) {
  factorList.innerHTML = "";
  const totalContribution = topFactors.reduce((sum, item) => sum + Math.abs(item.contribution), 0);

  topFactors.forEach((item, index) => {
    const influence = factorInfluence(item, totalContribution);
    const li = document.createElement("li");
    li.className = item.contribution >= 0 ? "factor-card raises-risk" : "factor-card lowers-risk";
    li.style.setProperty("--depth", `${Math.min(index * 10, 40)}px`);

    const top = document.createElement("div");
    top.className = "factor-card-top";

    const title = document.createElement("strong");
    title.textContent = item.label;

    const badge = document.createElement("span");
    badge.textContent = `${influence}% weight`;

    const value = document.createElement("p");
    value.textContent = `Value used: ${item.value_used}. ${factorEffect(item)}. Risk points: ${item.contribution}.`;

    const bar = document.createElement("span");
    bar.className = "factor-bar";
    bar.style.width = `${Math.max(8, influence)}%`;

    top.appendChild(title);
    top.appendChild(badge);
    li.appendChild(top);
    li.appendChild(value);
    li.appendChild(bar);
    factorList.appendChild(li);
  });
}

function resultDataNotes(data) {
  const notes = [];
  const estimatedFields = Object.entries(data.estimated_fields || {});

  if (data.missing_fields.length > 0) {
    notes.push(`Unknown answers filled from training defaults: ${data.missing_fields.join(", ")}.`);
  }

  if (estimatedFields.length > 0) {
    const estimates = estimatedFields.map(([field, value]) => `${field}: ${value}`).join(", ");
    notes.push(`Estimated optional values: ${estimates}.`);
  }

  if (data.model_notes && data.model_notes.length > 0) {
    notes.push(...data.model_notes);
  }

  return notes.length ? notes.join(" ") : "All requested fields were provided.";
}

function renderSuggestions(suggestions) {
  suggestionList.innerHTML = "";

  suggestions.forEach((item) => {
    const card = document.createElement("article");
    card.className = "suggestion-card";

    const priority = document.createElement("span");
    priority.className = "suggestion-priority";
    priority.textContent = item.priority;

    const title = document.createElement("strong");
    title.textContent = item.title;

    const detail = document.createElement("p");
    detail.textContent = item.detail;

    card.appendChild(priority);
    card.appendChild(title);
    card.appendChild(detail);
    suggestionList.appendChild(card);
  });
}

function matchingAdviceItems() {
  if (!latestPrediction) {
    return adviceSources.filter((source) => ["lifestyle", "activity"].includes(source.key));
  }

  const suggestionKeys = new Set(latestPrediction.risk_reduction_suggestions.map((item) => item.key));
  const matched = adviceSources.filter((source) => suggestionKeys.has(source.key));

  if (matched.length > 0) {
    return matched;
  }

  return adviceSources.filter((source) => ["lifestyle", "activity"].includes(source.key));
}

function renderAdvicePage() {
  const items = matchingAdviceItems();
  const percent = latestPrediction ? latestPrediction.risk_percent : "the predicted";
  const level = latestPrediction ? latestPrediction.risk_level.toLowerCase() : "current";
  const factorListText = latestPrediction
    ? latestPrediction.risk_reduction_suggestions.map((item) => item.title.toLowerCase()).join(", ")
    : "general prevention";

  adviceIntro.textContent = `Advice for the ${percent}% ${level} risk result, based on this patient's flagged areas: ${factorListText}.`;
  adviceList.innerHTML = "";

  items.forEach((item) => {
    const card = document.createElement("li");
    card.className = "advice-card direct-advice-item";

    const title = document.createElement("strong");
    title.textContent = item.title;

    const action = document.createElement("p");
    action.textContent = item.action;

    const official = document.createElement("div");
    official.className = "official-source";

    const source = document.createElement("span");
    source.textContent = "Official website";

    const link = document.createElement("a");
    link.href = item.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = item.source;

    card.appendChild(title);
    card.appendChild(action);
    official.appendChild(source);
    official.appendChild(link);
    card.appendChild(official);
    adviceList.appendChild(card);
  });
}

function metricRow(label, projectValue, publishedValue, lowerIsBetter = false) {
  const projectWins = lowerIsBetter ? projectValue < publishedValue : projectValue > publishedValue;
  return `
    <div class="comparison-row ${projectWins ? "project-wins" : ""}">
      <span>${label}</span>
      <strong>${projectValue}</strong>
      <em>${publishedValue}</em>
    </div>
  `;
}

async function renderComparisonPage() {
  comparisonIntro.textContent = "Loading benchmark comparison...";
  comparisonList.innerHTML = "";
  comparisonNotes.innerHTML = "";

  const response = await fetch("/api/comparison");
  const data = await response.json();

  if (!data.available) {
    comparisonIntro.textContent = data.message;
    return;
  }

  comparisonIntro.textContent = data.interpretation;
  comparisonList.innerHTML = data.benchmarks
    .map((benchmark) => {
      const project = benchmark.project;
      const published = benchmark.published;
      return `
        <article class="comparison-card">
          <div class="comparison-card-top">
            <span>${benchmark.name}</span>
            <strong>${benchmark.published_label}</strong>
          </div>
          <p>${benchmark.eligibility}</p>
          <div class="comparison-table-head">
            <span>Metric</span>
            <strong>Project</strong>
            <em>Published</em>
          </div>
          ${metricRow("Rows", project.rows, published.rows)}
          ${metricRow("ROC-AUC", project.roc_auc, published.roc_auc)}
          ${metricRow("Brier score", project.brier_score, published.brier_score, true)}
          ${metricRow("Log loss", project.log_loss, published.log_loss, true)}
          ${metricRow("Mean risk", project.mean_predicted_risk, published.mean_predicted_risk)}
        </article>
      `;
    })
    .join("");

  comparisonNotes.innerHTML = `
    <h3>Important notes</h3>
    <ul>
      ${data.notes.map((note) => `<li>${note}</li>`).join("")}
    </ul>
    <p>
      References:
      <a href="${data.references.framingham}" target="_blank" rel="noreferrer">Framingham</a>
      and
      <a href="${data.references.score2}" target="_blank" rel="noreferrer">SCORE2</a>.
    </p>
  `;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const submitButton = form.querySelector('button[type="submit"]');
  submitButton.disabled = true;
  submitButton.textContent = "Calculating...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formToPayload(form)),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.messages ? errorData.messages.join(" ") : "Prediction request failed");
    }

    const data = await response.json();
    latestPrediction = data;
    resultCard.classList.remove("hidden");
    meterFill.style.width = `${data.risk_percent}%`;
    riskPercent.textContent = `${data.risk_percent}%`;
    riskLevel.textContent = `${data.risk_level} Risk`;
    updateRiskStage(data.risk_level, data.risk_percent);
    scoreDistance.textContent = scoreDistanceText(data.risk_level, data.risk_percent);
    resultCopy.textContent = riskSummary(data.risk_level, data.risk_percent);
    whyScore.textContent = data.why_this_score;

    renderFactors(data.top_factors);
    renderSuggestions(data.risk_reduction_suggestions);

    missingCopy.textContent = resultDataNotes(data);

    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    resultCard.classList.remove("hidden");
    riskPercent.textContent = "Error";
    riskLevel.textContent = "Prediction unavailable";
    updateRiskStage("Low", 0);
    floatingRiskLevel.textContent = "Unavailable";
    scoreDistance.textContent = "";
    whyScore.textContent = "";
    resultCopy.textContent = error.message || "The app could not calculate a risk estimate just now. Please try again.";
    factorList.innerHTML = "";
    suggestionList.innerHTML = "";
    missingCopy.textContent = "";
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Calculate CAD Risk";
  }
});

form.addEventListener("reset", () => {
  window.setTimeout(clearResult, 0);
});

editAnswers.addEventListener("click", scrollToForm);
backToForm.addEventListener("click", scrollToForm);

tryAgain.addEventListener("click", () => {
  form.reset();
  clearResult();
  scrollToForm();
});

function openModal(modal, focusButton) {
  modal.classList.remove("hidden");
  document.body.classList.add("modal-open");
  focusButton.focus();
}

function closeModal(modal) {
  modal.classList.add("hidden");
  if (
    guideModal.classList.contains("hidden") &&
    adviceModal.classList.contains("hidden") &&
    comparisonModal.classList.contains("hidden")
  ) {
    document.body.classList.remove("modal-open");
  }
}

openGuideButtons.forEach((button) => {
  button.addEventListener("click", () => openModal(guideModal, closeGuide));
});

closeGuide.addEventListener("click", () => closeModal(guideModal));
guideModal.addEventListener("click", (event) => {
  if (event.target.matches("[data-close-modal]")) {
    closeModal(guideModal);
  }
});

openAdvice.addEventListener("click", () => {
  renderAdvicePage();
  openModal(adviceModal, closeAdvice);
});

floatingAdvice.addEventListener("click", () => {
  renderAdvicePage();
  openModal(adviceModal, closeAdvice);
});

closeAdvice.addEventListener("click", () => closeModal(adviceModal));
adviceModal.addEventListener("click", (event) => {
  if (event.target.matches("[data-close-advice]")) {
    closeModal(adviceModal);
  }
});

openComparisonButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    openModal(comparisonModal, closeComparison);
    await renderComparisonPage();
  });
});

closeComparison.addEventListener("click", () => closeModal(comparisonModal));
comparisonModal.addEventListener("click", (event) => {
  if (event.target.matches("[data-close-comparison]")) {
    closeModal(comparisonModal);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeModal(guideModal);
    closeModal(adviceModal);
    closeModal(comparisonModal);
  }
});

resultStage.addEventListener("pointermove", (event) => {
  const bounds = resultStage.getBoundingClientRect();
  const x = (event.clientX - bounds.left) / bounds.width - 0.5;
  const y = (event.clientY - bounds.top) / bounds.height - 0.5;
  resultStage.style.setProperty("--stage-tilt-x", `${(-y * 3).toFixed(2)}deg`);
  resultStage.style.setProperty("--stage-tilt-y", `${(x * 4).toFixed(2)}deg`);
  resultFloatCard.style.setProperty("--float-x", `${(x * 18).toFixed(1)}px`);
  resultFloatCard.style.setProperty("--float-y", `${(y * 14).toFixed(1)}px`);
});

resultStage.addEventListener("pointerleave", () => {
  resultStage.style.setProperty("--stage-tilt-x", "0deg");
  resultStage.style.setProperty("--stage-tilt-y", "0deg");
  resultFloatCard.style.setProperty("--float-x", "0px");
  resultFloatCard.style.setProperty("--float-y", "0px");
});
