/* cellstate explorer — view wiring.
 *
 * The server owns every number; this file owns layout, selection and rendering. Where the backend
 * marks a result as computed away from the fitted configuration, the page says so rather than
 * presenting it as the published value.
 */

import {
  loadingBars, stateScatter, spectrumChart, divergingBars,
  cosineHeatmap, responseCurve, panelStrip, trimmedTail, coverageByDepth, fmt,
} from './charts.js';

const state = {
  inventory: null,
  biologyRank: 4,
  nuisanceRank: 3,
  library: null,
  target: null,
  scatterX: 0,
  scatterY: 1,
  basisAxis: 'biology_0',
  basisLibrary: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [k, v] of Object.entries(params)) {
    if (v !== null && v !== undefined) url.searchParams.set(k, v);
  }
  const response = await fetch(url);
  if (!response.ok) {
    let detail = `${response.status}`;
    try { detail = (await response.json()).detail ?? detail; } catch { /* body was not JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

function put(container, node) {
  container.replaceChildren(node);
}

function fail(container, error) {
  const p = document.createElement('div');
  p.className = 'error';
  p.textContent = `could not load: ${error.message}`;
  container.replaceChildren(p);
}

function tile(key, value, meta, tone = '') {
  const d = document.createElement('div');
  d.className = `tile ${tone}`.trim();
  d.innerHTML = `<span class="k"></span><span class="v"></span><span class="m"></span>`;
  d.querySelector('.k').textContent = key;
  d.querySelector('.v').textContent = value;
  d.querySelector('.m').textContent = meta ?? '';
  return d;
}

function table(headers, rows) {
  const t = document.createElement('table');
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  for (const h of headers) {
    const th = document.createElement('th');
    th.textContent = h;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  t.appendChild(thead);
  const tb = document.createElement('tbody');
  for (const row of rows) {
    const tr = document.createElement('tr');
    if (row.__class) tr.className = row.__class;
    for (const cell of row.cells) {
      const td = document.createElement('td');
      if (cell instanceof Node) td.appendChild(cell); else td.textContent = cell;
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  t.appendChild(tb);
  return t;
}

function offDefaultBanner(ranks) {
  if (ranks.is_default) return null;
  const d = document.createElement('div');
  d.className = 'offdefault';
  d.textContent =
    `Computed at biology_rank=${ranks.biology}, nuisance_rank=${ranks.nuisance}. ` +
    `The published measurements are at ${ranks.default_biology} / ${ranks.default_nuisance}; ` +
    `nothing on this screen is the number in the model card.`;
  return d;
}

const ranks = () => ({ biology_rank: state.biologyRank, nuisance_rank: state.nuisanceRank });

/* ---------------------------------------------------------------- model laboratory */

async function renderLab() {
  const banner = offDefaultBanner({
    ...state, biology: state.biologyRank, nuisance: state.nuisanceRank,
    is_default: state.biologyRank === 4 && state.nuisanceRank === 3,
    default_biology: 4, default_nuisance: 3,
  });
  $('lab-offdefault').replaceChildren(...(banner ? [banner] : []));

  try {
    const measure = await api('/api/measure', ranks());
    const d = measure.decomposition;
    const s5 = measure.measurements.find((m) => m.name.startsWith('S5'));
    const tiles = [
      tile('S5', fmt(s5 ? s5.value : NaN, 4),
        s5 && Number.isFinite(s5.interval.lower) ? `[${fmt(s5.interval.lower, 2)}, ${fmt(s5.interval.upper, 2)}] · bound ${measure.bound}` : 'point estimate only',
        'bad'),
      tile('leakage', fmt(d.biology_across_library, 4), 'biology block, across library'),
      tile('signal', fmt(d.between_target, 4), 'between target — the denominator'),
      tile('nuisance', fmt(d.nuisance_across_library, 2), 'where library variation should land'),
      tile('leak / signal', fmt(d.biology_across_library / d.between_target, 2) + '×', ''),
    ];
    $('lab-tiles').replaceChildren(...tiles);

    const rows = measure.measurements.map((m) => {
      const chip = document.createElement('span');
      chip.className = `chip ${m.passed ? 'pass' : 'fail'}`;
      chip.textContent = m.passed ? 'pass' : 'fail';
      return {
        cells: [
          m.name, chip, fmt(m.value, 4),
          Number.isFinite(m.interval.lower) ? `[${fmt(m.interval.lower, 4)}, ${fmt(m.interval.upper, 4)}]` : '—',
          String(m.unit_count),
        ],
      };
    });
    put($('lab-measure'), table(['measurement', '', 'value', 'interval', 'K'], rows));
  } catch (error) { fail($('lab-tiles'), error); }

  try {
    const response = await api('/api/rank-response', { axis: 'biology', held_at: state.nuisanceRank });
    put($('lab-response'), responseCurve(response.points, state.biologyRank, {
      keys: ['biology_across_library', 'between_target'],
      labels: ['leakage', 'signal'],
    }));
  } catch (error) { fail($('lab-response'), error); }

  try {
    const spectrum = await api('/api/spectrum');
    put($('lab-spectrum'), spectrumChart(spectrum.series, { cut: state.biologyRank }));
  } catch (error) { fail($('lab-spectrum'), error); }
}

/* ---------------------------------------------------------------- arm explorer */

async function renderArm() {
  $('arm-title').textContent = `${state.library} / ${state.target}`;
  try {
    const arm = await api(`/api/arm/${encodeURIComponent(state.library)}/${encodeURIComponent(state.target)}`, ranks());
    const chip = $('arm-rankchip');
    chip.hidden = arm.ranks.is_default;
    chip.textContent = `rank ${arm.ranks.biology} / ${arm.ranks.nuisance} — off default`;

    const wrap = document.createElement('div');
    for (const axis of arm.axes) {
      const head = document.createElement('div');
      head.className = 'head';
      head.style.marginTop = '0.5rem';
      head.innerHTML = `<h3></h3><span class="mono" style="color:var(--ink-3);font-size:0.8rem"></span>`;
      head.querySelector('h3').textContent = axis.name;
      head.querySelector('span').textContent =
        `${axis.coordinate >= 0 ? '+' : ''}${fmt(axis.coordinate)} ± ${fmt(axis.standard_deviation)}`;
      wrap.appendChild(head);
      wrap.appendChild(loadingBars(axis.top_genes));
    }
    put($('arm-axes'), wrap);

    const abst = document.createElement('div');
    if (arm.abstention_required) {
      abst.className = 'abstention';
      const title = document.createElement('span');
      title.className = 't';
      title.textContent = 'abstention required';
      const list = document.createElement('ul');
      for (const reason of arm.reasons) {
        const li = document.createElement('li');
        li.textContent = reason;          // textContent, never innerHTML: reasons are server text
        list.appendChild(li);
      }
      abst.append(title, list);
    }
    $('arm-abstention').replaceChildren(abst);
  } catch (error) { fail($('arm-axes'), error); }

  try {
    const states = await api(`/api/library/${encodeURIComponent(state.library)}/states`, ranks());
    const xSel = $('arm-x');
    const ySel = $('arm-y');
    if (xSel.options.length !== states.axis_names.length) {
      for (const sel of [xSel, ySel]) {
        sel.replaceChildren(...states.axis_names.map((n, i) => new Option(n, String(i))));
      }
      xSel.value = String(Math.min(state.scatterX, states.axis_names.length - 1));
      ySel.value = String(Math.min(state.scatterY, states.axis_names.length - 1));
    }
    const xi = Math.min(Number(xSel.value), states.axis_names.length - 1);
    const yi = Math.min(Number(ySel.value), states.axis_names.length - 1);
    put($('arm-scatter'), stateScatter(states.arms, xi, yi, states.axis_names));
  } catch (error) { fail($('arm-scatter'), error); }

  try {
    const sweep = await api(`/api/sweep/${encodeURIComponent(state.library)}`, ranks());
    const chip = $('sweep-floorchip');
    chip.className = 'chip fail';
    chip.textContent = sweep.floor !== null
      ? `floor ${fmt(sweep.floor)} — ${sweep.floor_targets.join(', ')}`
      : 'no not-expressed reference in this library';
    const rows = sweep.rows.map((r) => ({
      __class: r.is_expressed ? '' : 'flag dim',
      cells: [
        r.target, fmt(r.distance), fmt(r.distance_lower_bound_sd),
        r.nt_cpm.toFixed(1),
        sweep.floor ? `${(r.distance / sweep.floor).toFixed(2)}×` : '—',
      ],
    }));
    put($('arm-sweep'), table(['target', '|delta|', 'sd ≥', 'NT CPM', 'vs floor'], rows));
  } catch (error) { fail($('arm-sweep'), error); }
}

/* ---------------------------------------------------------------- substrate */

async function renderSubstrate() {
  try {
    const [knock, day, spectrum] = await Promise.all([
      api('/api/knockdown'), api('/api/day'), api('/api/spectrum'),
    ]);
    const perturbation = spectrum.series[0];
    const placebo = spectrum.series[1];
    const diff = spectrum.series[2];
    $('sub-tiles').replaceChildren(
      tile('on-target log2FC', fmt(knock.mean_log2_fold_change, 3), `over ${knock.target_count} targets`, 'bad'),
      tile('wrong-signed', `${knock.wrong_signed} of ${knock.target_count}`, 'a working screen: −1 to −2', 'bad'),
      tile('perturbation s1/s0', fmt(perturbation.s1_over_s0, 2), `placebo ${fmt(placebo.s1_over_s0, 2)}`, 'bad'),
      tile('differentiation s1/s0', fmt(diff.s1_over_s0, 2), `PC1 ${(diff.pc1_variance_share * 100).toFixed(1)}%`, 'good'),
      tile('differentiation', `${fmt(day.differentiation_over_placebo, 2)}×`, `perturbed ${fmt(day.perturbed_over_placebo, 2)}× placebo`, 'good'),
    );

    put($('sub-spectrum'), spectrumChart(spectrum.series));
    put($('sub-spectrum-table'), table(['contrast matrix', 'rows', 's1/s0', 'PC1 var'],
      spectrum.series.map((s) => ({
        cells: [s.name, String(s.row_count), fmt(s.s1_over_s0, 2), `${(s.pc1_variance_share * 100).toFixed(1)}%`],
      }))));

    put($('sub-knockdown'), divergingBars(knock.rows, {
      valueKey: 'log2_fold_change', labelKey: 'target', flagKey: 'wrong_signed',
    }));

    put($('sub-day'), table(['contrast', '‖Δ‖', 'vs placebo'], [
      { cells: [`NT day ${day.days[0]} → day ${day.days[day.days.length - 1]}`, fmt(day.differentiation_distance), `${fmt(day.differentiation_over_placebo, 2)}×`] },
      { cells: ['perturbed target vs NT (mean)', fmt(day.perturbed_distance), `${fmt(day.perturbed_over_placebo, 2)}×`] },
      { cells: ['placebo NT_B vs NT_A (mean)', fmt(day.placebo_distance), '1.00×'] },
      { __class: 'dim', cells: [`${day.tracking_gene_count} of 100 genes track day at |r| > 0.7`, '', ''] },
    ]));
  } catch (error) { fail($('sub-tiles'), error); }

  try {
    const cal = await api('/api/calibration');
    const failed = cal.outcome !== 'passed';
    const chip = $('cal-chip');
    chip.textContent = `S6 ${failed ? 'FAIL' : 'PASS'}`;
    chip.className = `chip ${failed ? 'fail' : 'pass'}`;
    $('cal-tiles').replaceChildren(
      tile('coverage @ 0.90', fmt(cal.empirical_coverage, 4),
           `[${fmt(cal.interval.lower, 4)}, ${fmt(cal.interval.upper, 4)}] · K=${cal.unit_count}`),
      tile('calibration error', fmt(cal.calibration_error, 4),
           `floor ${cal.minimum_coverage} · max ${cal.maximum_calibration_error}`, 'good'),
      tile('upper bound', fmt(cal.calibration_error_upper_bound, 4),
           `the gate reads this · max ${cal.maximum_calibration_error}`, failed ? 'bad' : 'good'),
      tile('sd of z', fmt(cal.standard_deviation, 4),
           `trimmed ${fmt(cal.trimmed_standard_deviation, 4)} · max |z| ${fmt(cal.largest_absolute_score, 2)}`, 'warn'),
      tile('depth ↔ coverage', fmt(cal.depth_coverage_correlation, 4), 'corr over 14 libraries', 'bad'),
    );
    put($('cal-tail'), trimmedTail(cal));
    put($('cal-depth'), coverageByDepth(cal.by_library, cal.nominal_probability));
  } catch (error) { fail($('cal-tiles'), error); }

  try {
    const ranksTable = await api('/api/ranks', ranks());
    const rows = ranksTable.rows.map((r) => ({
      __class: r.is_expressed ? '' : 'flag dim',
      cells: [r.target, r.mean_rank.toFixed(1), String(r.best), String(r.worst), r.mean_nt_cpm.toFixed(1)],
    }));
    const silent = ranksTable.winners.filter((w) => w.is_expressed === 'False');
    const container = document.createElement('div');
    container.appendChild(table(['target', 'mean rank', 'best', 'worst', 'NT CPM'], rows));
    if (silent.length) {
      const note = document.createElement('p');
      note.className = 'warnline';
      note.style.marginTop = '0.6rem';
      note.textContent = silent
        .map((w) => `In ${w.library}, ${w.target} produces the largest contrast of all ${ranksTable.target_count} targets (${w.distance}) — and it is not expressed.`)
        .join(' ');
      container.appendChild(note);
    }
    put($('sub-ranks'), container);
  } catch (error) { fail($('sub-ranks'), error); }

  try {
    put($('sub-panel'), panelStrip(await api('/api/panel')));
  } catch (error) { fail($('sub-panel'), error); }
}

/* ---------------------------------------------------------------- fold / basis */

async function renderBasis() {
  try {
    const basis = await api('/api/basis', ranks());
    const axisSel = $('basis-axis');
    if (axisSel.options.length !== basis.axis_names.length) {
      axisSel.replaceChildren(...basis.axis_names.map((n) => new Option(n, n)));
      if (!basis.axis_names.includes(state.basisAxis)) state.basisAxis = basis.axis_names[0];
      axisSel.value = state.basisAxis;
    }
    const axis = axisSel.value;
    const flips = basis.sign_flips_by_axis[axis];
    $('basis-flipchip').textContent = `${flips} of ${basis.pair_count} fold pairs anti-aligned`;

    put($('basis-heatmap'), cosineHeatmap(basis.cosine_by_axis[axis], basis.libraries, { label: axis }));

    const anchors = basis.anchor_gene_by_axis[axis];
    const first = basis.cosine_by_axis[axis][0];
    put($('basis-anchors'), table(['fold', 'anchor gene', `cos vs ${basis.libraries[0]}`], basis.libraries.map((lib, i) => ({
      __class: first[i] < 0 ? 'dim' : '',
      cells: [lib, anchors[i], (first[i] >= 0 ? '+' : '') + fmt(first[i])],
    }))));
  } catch (error) { fail($('basis-heatmap'), error); }

  try {
    const fold = await api(`/api/fold/${encodeURIComponent(state.basisLibrary)}`, ranks());
    $('fold-title').textContent = `fold holding out ${fold.library}`;
    $('fold-tiles').replaceChildren(
      tile('ψ²', fmt(fold.psi_squared, 5), fold.dispersion_is_clamped ? 'CLAMPED' : 'fitted', fold.dispersion_is_clamped ? 'bad' : 'good'),
      tile('pre-clamp', fmt(fold.psi_squared_before_clamp, 5), ''),
      tile('fit libraries', String(fold.fit_library_count), 'held-out excluded'),
    );

    const wrap = document.createElement('div');
    for (const [heading, columns] of [['biology · W', fold.biology], ['nuisance · V', fold.nuisance]]) {
      const h = document.createElement('h3');
      h.textContent = heading;
      h.style.marginTop = '0.7rem';
      wrap.appendChild(h);
      for (const column of columns) {
        const sub = document.createElement('div');
        sub.className = 'eyebrow';
        sub.textContent = column.name;
        wrap.appendChild(sub);
        wrap.appendChild(loadingBars(column.top_genes.slice(0, 6), { rowHeight: 19 }));
      }
    }
    put($('fold-bases'), wrap);

    const residuals = Object.entries(fold.residual_norm_by_target)
      .sort((a, b) => b[1] - a[1])
      .map(([target, norm]) => ({
        __class: target === 'NT' ? '' : '',
        cells: [target === 'NT' ? 'NT (placebo split)' : target, fmt(norm, 4)],
      }));
    put($('fold-residuals'), table(['target', '|u_g|'], residuals));
  } catch (error) { fail($('fold-bases'), error); }
}

/* ---------------------------------------------------------------- shell */

const renderers = { lab: renderLab, arm: renderArm, substrate: renderSubstrate, basis: renderBasis };
let current = 'lab';

function show(view) {
  current = view;
  for (const button of document.querySelectorAll('nav.tabs button')) {
    button.setAttribute('aria-selected', String(button.dataset.view === view));
  }
  for (const section of document.querySelectorAll('.view')) {
    section.classList.toggle('active', section.id === `view-${view}`);
  }
  renderers[view]();
}

function debounce(fn, ms) {
  let handle = null;
  return (...args) => {
    clearTimeout(handle);
    handle = setTimeout(() => fn(...args), ms);
  };
}

async function boot() {
  try {
    const inventory = await api('/api/inventory');
    state.inventory = inventory;
    state.library = inventory.libraries[0].library;
    state.basisLibrary = state.library;
    state.target = inventory.targets.includes('GATA1') ? 'GATA1' : inventory.targets[0];

    $('slice-summary').textContent =
      `${inventory.libraries.length} libraries · ${inventory.targets.length} targets · ` +
      `${inventory.total_cells.toLocaleString()} cells · ${inventory.gene_count}-gene panel`;

    const libraryNames = inventory.libraries.map((l) => l.library);
    $('arm-library').replaceChildren(...libraryNames.map((l) => new Option(l, l)));
    $('basis-library').replaceChildren(...libraryNames.map((l) => new Option(l, l)));
    $('arm-target').replaceChildren(...inventory.targets.map((t) => new Option(t, t)));
    $('arm-library').value = state.library;
    $('basis-library').value = state.basisLibrary;
    $('arm-target').value = state.target;
  } catch (error) {
    const message = document.createElement('div');
    message.className = 'error';
    message.textContent = `could not reach the server: ${error.message}`;
    document.querySelector('main').replaceChildren(message);
    return;
  }

  for (const button of document.querySelectorAll('nav.tabs button')) {
    button.addEventListener('click', () => show(button.dataset.view));
  }

  const onRank = debounce(() => { if (current === 'lab') renderLab(); else renderers[current](); }, 160);
  for (const [id, key, label] of [['lab-bio', 'biologyRank', 'lab-bio-val'], ['lab-nui', 'nuisanceRank', 'lab-nui-val']]) {
    $(id).addEventListener('input', (event) => {
      state[key] = Number(event.target.value);
      $(label).textContent = event.target.value;
      onRank();
    });
  }
  $('lab-reset').addEventListener('click', () => {
    state.biologyRank = 4; state.nuisanceRank = 3;
    $('lab-bio').value = '4'; $('lab-nui').value = '3';
    $('lab-bio-val').textContent = '4'; $('lab-nui-val').textContent = '3';
    renderLab();
  });

  $('arm-library').addEventListener('change', (e) => { state.library = e.target.value; renderArm(); });
  $('arm-target').addEventListener('change', (e) => { state.target = e.target.value; renderArm(); });
  $('arm-x').addEventListener('change', (e) => { state.scatterX = Number(e.target.value); renderArm(); });
  $('arm-y').addEventListener('change', (e) => { state.scatterY = Number(e.target.value); renderArm(); });
  $('basis-axis').addEventListener('change', (e) => { state.basisAxis = e.target.value; renderBasis(); });
  $('basis-library').addEventListener('change', (e) => { state.basisLibrary = e.target.value; renderBasis(); });

  show('lab');
}

boot();
