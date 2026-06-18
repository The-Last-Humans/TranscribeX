const state = {
  status: null,
  selectedProfile: null,
};

const fields = {
  healthDot: document.getElementById("healthDot"),
  healthText: document.getElementById("healthText"),
  setupBanner: document.getElementById("setupBanner"),
  facts: document.getElementById("facts"),
  profiles: document.getElementById("profiles"),
  currentState: document.getElementById("currentState"),
  refreshButton: document.getElementById("refreshButton"),
  saveButton: document.getElementById("saveButton"),
  saveStatus: document.getElementById("saveStatus"),
  form: document.getElementById("setupForm"),
  asrModel: document.getElementById("asrModel"),
  device: document.getElementById("device"),
  vadModel: document.getElementById("vadModel"),
  puncModel: document.getElementById("puncModel"),
  spkModel: document.getElementById("spkModel"),
  batchSize: document.getElementById("batchSize"),
  hub: document.getElementById("hub"),
  maxUpload: document.getElementById("maxUpload"),
  apiKey: document.getElementById("apiKey"),
  adminKey: document.getElementById("adminKey"),
  preloadModel: document.getElementById("preloadModel"),
  keepUploads: document.getElementById("keepUploads"),
};

fields.refreshButton.addEventListener("click", () => loadStatus());
fields.saveButton.addEventListener("click", () => saveSetup());
fields.adminKey.value = localStorage.getItem("transcribex.adminKey") || "";
fields.adminKey.addEventListener("change", () => {
  localStorage.setItem("transcribex.adminKey", fields.adminKey.value);
});

loadStatus();

async function loadStatus() {
  setHealth("warn", "Loading");
  try {
    const response = await fetch("/v1/setup/status");
    if (!response.ok) {
      throw new Error(await response.text());
    }
    state.status = await response.json();
    state.selectedProfile = state.status.current.profile_id || firstRecommendedProfileId();
    renderStatus();
    setHealth(state.status.configured ? "ok" : "warn", state.status.configured ? "Configured" : "Setup required");
  } catch (error) {
    setHealth("bad", "Unavailable");
    fields.saveStatus.textContent = `Failed to load setup status: ${error.message}`;
  }
}

function renderStatus() {
  fields.setupBanner.classList.toggle("hidden", !state.status.setup_required);
  renderFacts(state.status.facts);
  renderProfiles(state.status.recommended_profiles, state.status.all_profiles);
  const initialConfig = state.status.configured ? state.status.current : findProfile(state.selectedProfile) || state.status.current;
  fillForm(initialConfig);
  fields.currentState.textContent = JSON.stringify(
    {
      configured: state.status.configured,
      setup_required: state.status.setup_required,
      config_path: state.status.config_path,
      current: state.status.current,
      facts: state.status.facts,
    },
    null,
    2
  );
}

function renderFacts(facts) {
  const gpuText = facts.nvidia_gpus.length
    ? facts.nvidia_gpus.map((gpu) => `${gpu.name} (${gpu.memory_mb || "unknown"} MB)`).join(", ")
    : "none detected";
  const rows = [
    ["OS", `${facts.os} ${facts.machine}`],
    ["CPU", facts.cpu_count || "unknown"],
    ["Memory", facts.memory_gb ? `${facts.memory_gb} GB` : "unknown"],
    ["NVIDIA GPU", gpuText],
  ];
  fields.facts.replaceChildren(
    ...rows.map(([label, value]) => {
      const wrapper = document.createElement("div");
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = label;
      detail.textContent = value;
      wrapper.append(term, detail);
      return wrapper;
    })
  );
}

function renderProfiles(recommendedProfiles, allProfiles) {
  const recommendedIds = new Set(recommendedProfiles.map((profile) => profile.id));
  const profiles = [...recommendedProfiles, ...allProfiles.filter((profile) => !recommendedIds.has(profile.id))];
  fields.profiles.replaceChildren(
    ...profiles.map((profile) => {
      const card = document.createElement("article");
      card.className = `profile ${profile.id === state.selectedProfile ? "selected" : ""}`;

      const content = document.createElement("div");
      const title = document.createElement("h3");
      title.textContent = `${profile.label}${recommendedIds.has(profile.id) ? " - Recommended" : ""}`;

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.append(
        line(`model: ${profile.asr_model}`),
        line(`device: ${profile.device}`),
        line(`speaker: ${profile.spk_model || "disabled"}`)
      );

      const why = document.createElement("p");
      why.textContent = profile.why;
      content.append(title, meta, why);

      if (profile.caution) {
        const caution = document.createElement("p");
        caution.className = "caution";
        caution.textContent = profile.caution;
        content.append(caution);
      }

      const button = document.createElement("button");
      button.type = "button";
      button.className = profile.id === state.selectedProfile ? "" : "secondary";
      button.textContent = profile.id === state.selectedProfile ? "Selected" : "Use profile";
      button.addEventListener("click", () => {
        state.selectedProfile = profile.id;
        fillForm(profile);
        renderProfiles(recommendedProfiles, allProfiles);
      });

      card.append(content, button);
      return card;
    })
  );
}

function fillForm(config) {
  fields.asrModel.value = config.asr_model || "";
  fields.device.value = config.device || "cpu";
  fields.vadModel.value = config.vad_model || "";
  fields.puncModel.value = config.punc_model || "";
  fields.spkModel.value = config.spk_model || "";
  fields.batchSize.value = config.batch_size_s || 300;
  fields.hub.value = config.hub || "";
  fields.maxUpload.value = config.max_upload_mb || state.status?.current?.max_upload_mb || 2048;
  fields.preloadModel.checked = Boolean(config.preload_model ?? state.status?.current?.preload_model);
  fields.keepUploads.checked = Boolean(config.keep_uploads ?? state.status?.current?.keep_uploads);
}

async function saveSetup() {
  fields.saveButton.disabled = true;
  fields.saveStatus.textContent = "Saving setup";

  const payload = {
    profile_id: state.selectedProfile,
    asr_model: emptyToNull(fields.asrModel.value),
    device: fields.device.value,
    vad_model: emptyToNull(fields.vadModel.value),
    punc_model: emptyToNull(fields.puncModel.value),
    spk_model: emptyToNull(fields.spkModel.value),
    hub: emptyToNull(fields.hub.value),
    batch_size_s: Number(fields.batchSize.value || 300),
    max_upload_mb: Number(fields.maxUpload.value || 2048),
    preload_model: fields.preloadModel.checked,
    keep_uploads: fields.keepUploads.checked,
    setup_complete: true,
  };
  if (fields.apiKey.value) {
    payload.api_key = fields.apiKey.value;
  }

  try {
    const headers = { "Content-Type": "application/json" };
    if (fields.adminKey.value) {
      headers.Authorization = `Bearer ${fields.adminKey.value}`;
    }
    const response = await fetch("/v1/setup/apply", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const result = await response.json();
    fields.apiKey.value = "";
    fields.saveStatus.textContent = "Saved";
    await loadStatus();
    fields.currentState.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    fields.saveStatus.textContent = `Save failed: ${error.message}`;
  } finally {
    fields.saveButton.disabled = false;
  }
}

function setHealth(kind, text) {
  fields.healthDot.className = `dot ${kind}`;
  fields.healthText.textContent = text;
}

function firstRecommendedProfileId() {
  return state.status?.recommended_profiles?.[0]?.id || state.status?.all_profiles?.[0]?.id || null;
}

function findProfile(profileId) {
  const profiles = [...(state.status?.recommended_profiles || []), ...(state.status?.all_profiles || [])];
  return profiles.find((profile) => profile.id === profileId);
}

function line(text) {
  const item = document.createElement("span");
  item.textContent = text;
  return item;
}

function emptyToNull(value) {
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}
