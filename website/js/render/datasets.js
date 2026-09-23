import { el, mount, link } from "../dom.js";
import { fmtCount } from "../format.js";

const hubDatasetUrl = repo => "https://huggingface.co/datasets/" + repo;

// family order comes from meta.sampling.families when present
const FAMILY_ORDER = ["pr", "lm", "combine", "mechanical"];

function orderedEntries(counts, order) {
  const entries = Object.entries(counts || {});
  const rank = k => { const i = order.indexOf(k); return i < 0 ? order.length : i; };
  return entries.sort((a, b) => rank(a[0]) - rank(b[0]) || b[1] - a[1]);
}

function langCell(stats) {
  const entries = orderedEntries(stats?.languages, []).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return el("td", { class: "ds-lang" }, "—");
  const label = entries.length > 1 ? `${entries[0][0]} +${entries.length - 1}` : entries[0][0];
  const title = entries.map(([k, v]) => `${k} ${v.toLocaleString()}`).join(" · ");
  return el("td", { class: "ds-lang", title }, label);
}

function familyMixCell(stats, familyOrder) {
  const entries = orderedEntries(stats?.families, familyOrder);
  const total = entries.reduce((sum, [, v]) => sum + v, 0);
  if (!total) return el("td", { class: "ds-mix" }, "—");
  const mix = [...entries].sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k} ${((v / total) * 100).toFixed(0)}%`).join(" · ");
  return el("td", { class: "ds-mix", title: `${mix} (instances)` }, mix);
}

function samplingNote(s) {
  if (!s) return null;
  const shares = list => (list || []).map(([k, v]) => `${k} ${v}%`).join(" · ");
  const caps = [];
  if (s.repo_cap != null) caps.push(`≤${s.repo_cap} per repo`);
  if (s.max_prefix_chars != null) caps.push(`≤${fmtCount(s.max_prefix_chars)} prefix chars`);
  if (s.non_benchmark_language_fraction != null) {
    caps.push(`≤${(s.non_benchmark_language_fraction * 100).toFixed(0)}% non-${s.benchmark_language || "python"}`);
  }
  const parts = [shares(s.phases), shares(s.families), caps.join(" · ")].filter(Boolean);
  return el("div", { class: "ds-sampling" }, "sampling: " + parts.join("  |  "));
}

export function renderDatasets(container, metaEl, manifest) {
  const sources = manifest?.sources || [];
  if (!sources.length) {
    mount(container, el("div", { class: "empty" }, "no dataset manifest."));
    if (metaEl) metaEl.textContent = "";
    return;
  }

  const familyOrder = (manifest.sampling?.families || []).map(([k]) => k);
  const order = familyOrder.length ? familyOrder : FAMILY_ORDER;
  const ordered = [...sources].sort(
    (a, b) => (b.stats?.instances ?? b.total_rows ?? 0) - (a.stats?.instances ?? a.total_rows ?? 0));
  let totalRows = 0, totalShards = 0;

  const rows = ordered.map(s => {
    const rowCount = Number(s.total_rows) || 0;
    const shardCount = (s.shards || []).length;
    totalRows += rowCount;
    totalShards += shardCount;

    return el("tr", {},
      el("td", { class: "ds-name" }, s.name || "—"),
      langCell(s.stats),
      el("td", { class: "r ds-num" }, rowCount.toLocaleString()),
      el("td", { class: "r ds-num" }, s.stats ? s.stats.instances.toLocaleString() : "—"),
      familyMixCell(s.stats, order),
      el("td", { class: "r ds-num" }, shardCount.toLocaleString()),
      el("td", { class: "ds-source" },
        s.repo ? link(hubDatasetUrl(s.repo), s.repo, { title: s.repo }) : "—"));
  });

  if (metaEl) {
    const instances = manifest.unique_instances != null ? ` · ${fmtCount(manifest.unique_instances)} instances` : "";
    metaEl.textContent = `${ordered.length} datasets · ${fmtCount(totalRows)} trajectories${instances} · ${totalShards.toLocaleString()} shards`;
  }

  mount(container, el("div", { class: "data-table-wrap" },
    el("table", { class: "data-table datasets-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "dataset"),
        el("th", {}, "lang"),
        el("th", { class: "r" }, "trajectories"),
        el("th", { class: "r" }, "instances"),
        el("th", {}, "family mix"),
        el("th", { class: "r" }, "shards"),
        el("th", {}, "source"))),
      el("tbody", {}, rows))),
    samplingNote(manifest.sampling));
}
