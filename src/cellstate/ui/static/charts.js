/* Inline-SVG chart primitives for the cellstate explorer.
 *
 * No chart library and no CDN: the page must work offline, and a research tool should not carry a
 * build step. Each function takes data and returns an <svg> element sized by viewBox, so the page
 * scales it with CSS.
 *
 * Colour is by role, never by rank, and comes from CSS custom properties so the palette lives in
 * one place and both themes swap together. The two categorical hues were validated for lightness,
 * chroma, CVD separation and contrast against both surfaces.
 */

const NS = 'http://www.w3.org/2000/svg';

function el(name, attrs = {}, text = null) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== null) node.textContent = text;
  return node;
}

function svg(width, height, label) {
  const node = el('svg', {
    viewBox: `0 0 ${width} ${height}`,
    role: 'img',
    'aria-label': label,
    preserveAspectRatio: 'xMidYMid meet',
    // Text is sized in viewBox units, so letting a 620-wide chart stretch to 1300px would
    // double every label. Scale down freely, never up past the design width.
    style: `max-width: ${width}px`,
  });
  return node;
}

function fmt(value, digits = 3) {
  if (!Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

/* ------------------------------------------------------------------ diverging gene loadings
 *
 * An axis IS its loadings, so this is the most literal chart on the page: each gene as a bar from a
 * central zero, positive one way and negative the other. Both poles are kept because the
 * informative axes here are contrasts — a gene at -0.31 is as much a part of the axis as one at
 * +0.33.
 */
export function loadingBars(genes, { width = 460, rowHeight = 21 } = {}) {
  const height = genes.length * rowHeight + 20;
  const node = svg(width, height, 'Gene loadings on this axis, positive and negative poles');
  const max = Math.max(...genes.map((g) => Math.abs(g.loading))) || 1;
  const mid = width * 0.52;
  const scale = (width * 0.34) / max;

  genes.forEach((gene, i) => {
    const y = i * rowHeight + 12;
    const w = Math.abs(gene.loading) * scale;
    const positive = gene.loading >= 0;
    node.appendChild(
      el('rect', {
        x: positive ? mid : mid - w,
        y: y - 7,
        width: Math.max(w, 1),
        height: 12,
        rx: 2,
        fill: positive ? 'var(--pos-pole)' : 'var(--neg-pole)',
      })
    );
    node.appendChild(
      el(
        'text',
        {
          x: mid - 8,
          y: y + 3,
          'text-anchor': 'end',
          class: 'lbl',
          fill: 'var(--ink-2)',
        },
        positive ? gene.symbol : ''
      )
    );
    node.appendChild(
      el(
        'text',
        { x: mid + 8, y: y + 3, 'text-anchor': 'start', class: 'lbl', fill: 'var(--ink-2)' },
        positive ? '' : gene.symbol
      )
    );
    node.appendChild(
      el(
        'text',
        {
          x: positive ? mid + w + 6 : mid - w - 6,
          y: y + 3,
          'text-anchor': positive ? 'start' : 'end',
          class: 'num',
          fill: 'var(--ink-3)',
        },
        gene.loading.toFixed(2)
      )
    );
  });
  node.appendChild(
    el('line', { x1: mid, y1: 4, x2: mid, y2: height - 8, stroke: 'var(--rule-2)', 'stroke-width': 1 })
  );
  return node;
}

/* ------------------------------------------------------------------ state scatter with ellipses
 *
 * Arms of ONE library in two of its shared biology coordinates, each with a 1-sd posterior ellipse.
 * One library only: a belief about library L comes from the fold that excluded L, so arms in
 * different libraries are in different fitted bases and do not compare.
 */
export function stateScatter(arms, xi, yi, axisNames, { width = 560, height = 420 } = {}) {
  const node = svg(width, height, `Arm states in ${axisNames[xi]} against ${axisNames[yi]}`);
  const pad = { l: 56, r: 18, t: 16, b: 44 };

  const xs = arms.map((a) => a.coordinates[xi]);
  const ys = arms.map((a) => a.coordinates[yi]);
  const sx = arms.map((a, i) => Math.sqrt(Math.max(a.covariance[xi][xi], 0)));
  const sy = arms.map((a, i) => Math.sqrt(Math.max(a.covariance[yi][yi], 0)));
  const xmin = Math.min(...xs.map((v, i) => v - sx[i])) ;
  const xmax = Math.max(...xs.map((v, i) => v + sx[i]));
  const ymin = Math.min(...ys.map((v, i) => v - sy[i]));
  const ymax = Math.max(...ys.map((v, i) => v + sy[i]));
  const padX = (xmax - xmin) * 0.08 || 1;
  const padY = (ymax - ymin) * 0.08 || 1;
  const X = (v) => pad.l + ((v - xmin + padX) / (xmax - xmin + 2 * padX)) * (width - pad.l - pad.r);
  const Y = (v) => height - pad.b - ((v - ymin + padY) / (ymax - ymin + 2 * padY)) * (height - pad.t - pad.b);

  // recessive frame
  node.appendChild(
    el('rect', {
      x: pad.l, y: pad.t, width: width - pad.l - pad.r, height: height - pad.t - pad.b,
      fill: 'none', stroke: 'var(--rule)', 'stroke-width': 1,
    })
  );
  if (xmin < 0 && xmax > 0) {
    node.appendChild(el('line', { x1: X(0), y1: pad.t, x2: X(0), y2: height - pad.b, stroke: 'var(--rule)', 'stroke-dasharray': '3 3' }));
  }
  if (ymin < 0 && ymax > 0) {
    node.appendChild(el('line', { x1: pad.l, y1: Y(0), x2: width - pad.r, y2: Y(0), stroke: 'var(--rule)', 'stroke-dasharray': '3 3' }));
  }

  arms.forEach((arm, i) => {
    const cx = X(arm.coordinates[xi]);
    const cy = Y(arm.coordinates[yi]);
    const rx = Math.abs(X(arm.coordinates[xi] + sx[i]) - cx);
    const ry = Math.abs(Y(arm.coordinates[yi] + sy[i]) - cy);
    const colour = arm.is_null ? 'var(--series-b)' : 'var(--series-a)';
    const g = el('g', { class: 'arm' });
    g.appendChild(
      el('ellipse', {
        cx, cy, rx: Math.max(rx, 1), ry: Math.max(ry, 1),
        fill: colour, 'fill-opacity': 0.1, stroke: colour, 'stroke-opacity': 0.45, 'stroke-width': 1,
      })
    );
    g.appendChild(
      el('circle', { cx, cy, r: arm.is_null ? 5 : 3.6, fill: colour, stroke: 'var(--surface)', 'stroke-width': 1.5 })
    );
    g.appendChild(el('title', {}, `${arm.target}  ${axisNames[xi]} ${fmt(arm.coordinates[xi])} ± ${fmt(sx[i])}  ${axisNames[yi]} ${fmt(arm.coordinates[yi])} ± ${fmt(sy[i])}`));
    node.appendChild(g);
  });

  // label NT and the two extremes only — never a number on every point
  const extremes = [...arms].sort((a, b) => {
    const da = Math.hypot(a.coordinates[xi], a.coordinates[yi]);
    const db = Math.hypot(b.coordinates[xi], b.coordinates[yi]);
    return db - da;
  }).slice(0, 2);
  for (const arm of [...extremes, ...arms.filter((a) => a.is_null)]) {
    node.appendChild(
      el('text', {
        x: X(arm.coordinates[xi]) + 8, y: Y(arm.coordinates[yi]) - 7,
        class: 'lbl', fill: arm.is_null ? 'var(--series-b)' : 'var(--ink-2)',
        'font-weight': arm.is_null ? 600 : 400,
      }, arm.target)
    );
  }

  node.appendChild(el('text', { x: (width + pad.l) / 2, y: height - 12, 'text-anchor': 'middle', class: 'ax', fill: 'var(--ink-3)' }, axisNames[xi]));
  const yl = el('text', { x: 16, y: (height - pad.b + pad.t) / 2, 'text-anchor': 'middle', class: 'ax', fill: 'var(--ink-3)', transform: `rotate(-90 16 ${(height - pad.b + pad.t) / 2})` }, axisNames[yi]);
  node.appendChild(yl);
  return node;
}

/* ------------------------------------------------------------------ spectrum decay */
export function spectrumChart(series, { width = 620, height = 300, cut = null } = {}) {
  const node = svg(width, height, 'Singular value spectra, normalized to the leading value');
  const pad = { l: 54, r: 92, t: 20, b: 46 };
  const n = Math.min(6, Math.max(...series.map((s) => s.normalized.length)));
  const X = (i) => pad.l + (i / (n - 1)) * (width - pad.l - pad.r);
  const Y = (v) => height - pad.b - v * (height - pad.t - pad.b);
  const colours = ['var(--series-a)', 'var(--series-ref)', 'var(--series-b)'];
  const ends = [];

  for (let g = 0; g <= 4; g += 1) {
    const y = Y(g / 4);
    node.appendChild(el('line', { x1: pad.l, y1: y, x2: width - pad.r, y2: y, stroke: 'var(--rule)' }));
    node.appendChild(el('text', { x: pad.l - 10, y: y + 4, 'text-anchor': 'end', class: 'num', fill: 'var(--ink-3)' }, (g / 4).toFixed(2)));
  }
  for (let i = 0; i < n; i += 1) {
    node.appendChild(el('text', { x: X(i), y: height - pad.b + 20, 'text-anchor': 'middle', class: 'num', fill: 'var(--ink-3)' }, `s${i}`));
  }

  if (cut !== null && cut >= 1 && cut <= n) {
    const x = X(cut - 0.5);
    node.appendChild(el('line', { x1: x, y1: pad.t, x2: x, y2: height - pad.b, stroke: 'var(--warn)', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }));
    node.appendChild(el('text', { x: x + 6, y: pad.t + 12, class: 'lbl', fill: 'var(--warn)', 'font-weight': 600 }, `rank ${cut}`));
  }

  series.forEach((s, si) => {
    const pts = s.normalized.slice(0, n).map((v, i) => `${X(i)},${Y(v)}`).join(' ');
    const dashed = si === 1;
    node.appendChild(el('polyline', {
      points: pts, fill: 'none', stroke: colours[si], 'stroke-width': 2,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round',
      ...(dashed ? { 'stroke-dasharray': '5 4' } : {}),
    }));
    s.normalized.slice(0, n).forEach((v, i) => {
      const c = el('circle', { cx: X(i), cy: Y(v), r: 4, fill: colours[si], stroke: 'var(--surface)', 'stroke-width': 2 });
      c.appendChild(el('title', {}, `${s.name} — s${i}/s0 = ${fmt(v)}`));
      node.appendChild(c);
    });
    ends.push({ y: Y(s.normalized[Math.min(n, s.normalized.length) - 1]), colour: colours[si], text: s.name.split(':')[0] });
  });

  // Push coincident end labels apart. The perturbation and placebo curves land within a hundredth
  // of each other -- that coincidence IS the result, so the labels have to stay readable through it.
  ends.sort((a, b) => a.y - b.y);
  const MIN_GAP = 13;
  for (let i = 1; i < ends.length; i += 1) {
    if (ends[i].y - ends[i - 1].y < MIN_GAP) ends[i].y = ends[i - 1].y + MIN_GAP;
  }
  for (const end of ends) {
    node.appendChild(el('text', {
      x: width - pad.r + 8, y: end.y + 4, class: 'lbl', fill: end.colour, 'font-weight': 600,
    }, end.text));
  }
  return node;
}

/* ------------------------------------------------------------------ diverging horizontal bars
 * Used for knockdown: one bar per target from a central zero, sorted.
 */
export function divergingBars(rows, { width = 620, rowHeight = 22, valueKey, labelKey, flagKey } = {}) {
  const height = rows.length * rowHeight + 34;
  const node = svg(width, height, 'Per-target values, diverging from zero');
  const max = Math.max(...rows.map((r) => Math.abs(r[valueKey]))) || 1;
  const left = 92;
  const mid = left + (width - left - 64) * 0.62;
  const scale = (width - left - 74) * 0.34 / max;

  node.appendChild(el('line', { x1: mid, y1: 8, x2: mid, y2: height - 26, stroke: 'var(--rule-2)' }));
  rows.forEach((row, i) => {
    const y = i * rowHeight + 20;
    const v = row[valueKey];
    const w = Math.abs(v) * scale;
    const positive = v >= 0;
    const flagged = flagKey ? row[flagKey] : false;
    const colour = positive ? 'var(--neg-pole)' : 'var(--pos-pole)';
    const g = el('g');
    g.appendChild(el('rect', {
      x: positive ? mid : mid - w, y: y - 8, width: Math.max(w, 1), height: 13, rx: 2,
      fill: colour, 'fill-opacity': flagged ? 0.35 : 1,
      ...(flagged ? { stroke: colour, 'stroke-width': 1, 'stroke-dasharray': '2 2' } : {}),
    }));
    g.appendChild(el('text', { x: left - 10, y: y + 3, 'text-anchor': 'end', class: 'lbl', fill: flagged ? 'var(--ink-3)' : 'var(--ink-2)' }, row[labelKey]));
    g.appendChild(el('text', {
      x: positive ? mid + w + 7 : mid - w - 7, y: y + 3,
      'text-anchor': positive ? 'start' : 'end', class: 'num', fill: 'var(--ink-3)',
    }, (v >= 0 ? '+' : '') + v.toFixed(3)));
    g.appendChild(el('title', {}, `${row[labelKey]}: ${fmt(v)}${flagged ? ' — not expressed, cannot have been knocked down' : ''}`));
    node.appendChild(g);
  });
  return node;
}

/* ------------------------------------------------------------------ cross-fold cosine heatmap
 *
 * The defect made visible. A diverging scale with a neutral midpoint: teal for agreement, rust for
 * anti-alignment, grey at zero. Sign flips read as a checkerboard.
 */
export function cosineHeatmap(matrix, labels, { cell = 26, label = 'axis' } = {}) {
  const pad = { l: 58, t: 58 };
  const size = labels.length * cell;
  const node = svg(pad.l + size + 12, pad.t + size + 12, `Cross-fold cosine similarity for ${label}`);

  const colourFor = (v) => {
    const a = Math.min(Math.abs(v), 1);
    if (v >= 0) return `color-mix(in oklab, var(--pos-pole) ${Math.round(a * 100)}%, var(--neutral-mid))`;
    return `color-mix(in oklab, var(--neg-pole) ${Math.round(a * 100)}%, var(--neutral-mid))`;
  };

  labels.forEach((row, i) => {
    node.appendChild(el('text', { x: pad.l - 8, y: pad.t + i * cell + cell / 2 + 4, 'text-anchor': 'end', class: 'num', fill: 'var(--ink-3)' }, row));
    const t = el('text', { x: pad.l + i * cell + cell / 2, y: pad.t - 8, 'text-anchor': 'start', class: 'num', fill: 'var(--ink-3)', transform: `rotate(-90 ${pad.l + i * cell + cell / 2} ${pad.t - 8})` }, row);
    node.appendChild(t);
    labels.forEach((col, j) => {
      const v = matrix[i][j];
      const g = el('g');
      g.appendChild(el('rect', {
        x: pad.l + j * cell + 1, y: pad.t + i * cell + 1,
        width: cell - 2, height: cell - 2, rx: 2,
        fill: colourFor(v), stroke: 'var(--surface)', 'stroke-width': 1,
      }));
      g.appendChild(el('title', {}, `${row} vs ${col}: cos = ${fmt(v)}${v < 0 ? '  (anti-aligned)' : ''}`));
      node.appendChild(g);
    });
  });
  return node;
}

/* ------------------------------------------------------------------ rank response curve */
export function responseCurve(points, current, { width = 620, height = 280, keys, labels } = {}) {
  const node = svg(width, height, 'Measurement response across the rank grid');
  const pad = { l: 58, r: 96, t: 18, b: 42 };
  const ranks = points.map((p) => p.rank);
  const X = (r) => pad.l + ((r - Math.min(...ranks)) / (Math.max(...ranks) - Math.min(...ranks))) * (width - pad.l - pad.r);
  const all = keys.flatMap((k) => points.map((p) => p[k])).filter(Number.isFinite);
  const lo = Math.min(...all, 0);
  const hi = Math.max(...all);
  const Y = (v) => height - pad.b - ((v - lo) / (hi - lo || 1)) * (height - pad.t - pad.b);
  const colours = ['var(--series-a)', 'var(--series-b)', 'var(--series-ref)'];

  for (let g = 0; g <= 4; g += 1) {
    const v = lo + ((hi - lo) * g) / 4;
    node.appendChild(el('line', { x1: pad.l, y1: Y(v), x2: width - pad.r, y2: Y(v), stroke: 'var(--rule)' }));
    node.appendChild(el('text', { x: pad.l - 10, y: Y(v) + 4, 'text-anchor': 'end', class: 'num', fill: 'var(--ink-3)' }, v.toFixed(2)));
  }
  ranks.forEach((r) => node.appendChild(el('text', { x: X(r), y: height - pad.b + 20, 'text-anchor': 'middle', class: 'num', fill: 'var(--ink-3)' }, String(r))));

  if (current !== null && ranks.includes(current)) {
    node.appendChild(el('line', { x1: X(current), y1: pad.t, x2: X(current), y2: height - pad.b, stroke: 'var(--warn)', 'stroke-width': 1.5 }));
  }

  keys.forEach((key, ki) => {
    const usable = points.filter((p) => Number.isFinite(p[key]));
    node.appendChild(el('polyline', {
      points: usable.map((p) => `${X(p.rank)},${Y(p[key])}`).join(' '),
      fill: 'none', stroke: colours[ki], 'stroke-width': 2, 'stroke-linejoin': 'round',
    }));
    usable.forEach((p) => {
      const c = el('circle', { cx: X(p.rank), cy: Y(p[key]), r: p.rank === current ? 5.5 : 3.5, fill: colours[ki], stroke: 'var(--surface)', 'stroke-width': 2 });
      c.appendChild(el('title', {}, `${labels[ki]} at rank ${p.rank}: ${fmt(p[key], 4)}`));
      node.appendChild(c);
    });
    const last = usable[usable.length - 1];
    if (last) node.appendChild(el('text', { x: width - pad.r + 8, y: Y(last[key]) + 4, class: 'lbl', fill: colours[ki], 'font-weight': 600 }, labels[ki]));
  });
  node.appendChild(el('text', { x: (width - pad.r + pad.l) / 2, y: height - 8, 'text-anchor': 'middle', class: 'ax', fill: 'var(--ink-3)' }, 'rank'));
  return node;
}

/* ------------------------------------------------------------------ log-scale expression strip */
export function panelStrip(genes, { width = 1180, height = 150 } = {}) {
  const node = svg(width, height, 'Panel genes by mean NT expression, log scale');
  const pad = { l: 46, r: 14, t: 18, b: 40 };
  const values = genes.map((g) => Math.max(g.mean_nt_cpm, 0.1));
  const lo = Math.log10(Math.min(...values));
  const hi = Math.log10(Math.max(...values));
  const X = (i) => pad.l + (i / (genes.length - 1)) * (width - pad.l - pad.r);
  const Y = (v) => height - pad.b - ((Math.log10(Math.max(v, 0.1)) - lo) / (hi - lo)) * (height - pad.t - pad.b);

  for (let d = Math.ceil(lo); d <= Math.floor(hi); d += 1) {
    node.appendChild(el('line', { x1: pad.l, y1: Y(10 ** d), x2: width - pad.r, y2: Y(10 ** d), stroke: 'var(--rule)' }));
    node.appendChild(el('text', { x: pad.l - 8, y: Y(10 ** d) + 4, 'text-anchor': 'end', class: 'num', fill: 'var(--ink-3)' }, `1e${d}`));
  }
  genes.forEach((gene, i) => {
    const colour = !gene.is_expressed ? 'var(--neg-pole)' : gene.is_target ? 'var(--series-b)' : 'var(--rule-2)';
    const bar = el('line', {
      x1: X(i), y1: height - pad.b, x2: X(i), y2: Y(gene.mean_nt_cpm),
      stroke: colour, 'stroke-width': gene.is_target || !gene.is_expressed ? 2.4 : 1.4,
    });
    bar.appendChild(el('title', {}, `${gene.symbol}: ${gene.mean_nt_cpm.toFixed(1)} CPM${gene.is_target ? ' — CRISPRi target' : ''}${gene.is_expressed ? '' : ' — not expressed'}`));
    node.appendChild(bar);
  });
  node.appendChild(el('text', { x: pad.l, y: height - 10, class: 'ax', fill: 'var(--ink-3)' }, '100 panel genes, mean NT CPM (log)'));
  return node;
}

/* ------------------------------------------------------------------ S6: the trimmed tail
 *
 * Two bars against a reference line at 1.0, which is where a calibrated spread sits. The whole
 * point is the distance between them: untrimmed the standardized residuals spread 1.28, and
 * removing 2% of the outcomes brings that to 1.00. A ratio reports only the first and reads as a
 * uniform scale error; these two together say it is a tail.
 */
export function trimmedTail(data, { width = 460, height = 190 } = {}) {
  const node = svg(width, height, 'Spread of standardized residuals, before and after trimming');
  const pad = { l: 140, r: 58, t: 22, b: 34 };
  const bars = [
    { label: 'all 1,400', value: data.standard_deviation, colour: 'var(--series-b)' },
    {
      label: `trimming ${Math.round(data.trimmed_fraction * 100)}%`,
      value: data.trimmed_standard_deviation,
      colour: 'var(--series-a)',
    },
  ];
  const hi = Math.max(...bars.map((b) => b.value), 1.0) * 1.12;
  const X = (v) => pad.l + (v / hi) * (width - pad.l - pad.r);
  const rowHeight = (height - pad.t - pad.b) / bars.length;

  const one = X(1.0);
  node.appendChild(el('line', {
    x1: one, y1: pad.t - 6, x2: one, y2: height - pad.b, stroke: 'var(--ink-3)',
    'stroke-width': 1.5, 'stroke-dasharray': '4 3',
  }));
  node.appendChild(el('text', {
    x: one, y: pad.t - 10, 'text-anchor': 'middle', class: 'num', fill: 'var(--ink-3)',
  }, 'calibrated = 1.00'));

  bars.forEach((bar, index) => {
    const y = pad.t + index * rowHeight + rowHeight * 0.22;
    const barHeight = rowHeight * 0.46;
    const rect = el('rect', {
      x: pad.l, y, width: Math.max(X(bar.value) - pad.l, 1), height: barHeight,
      fill: bar.colour, rx: 3,
    });
    rect.appendChild(el('title', {}, `${bar.label}: sd ${fmt(bar.value, 4)}`));
    node.appendChild(rect);
    node.appendChild(el('text', {
      x: pad.l - 10, y: y + barHeight / 2 + 4, 'text-anchor': 'end', class: 'lbl',
      fill: 'var(--ink-2)',
    }, bar.label));
    node.appendChild(el('text', {
      x: X(bar.value) + 8, y: y + barHeight / 2 + 4, class: 'num', fill: 'var(--ink)',
      'font-weight': 600,
    }, fmt(bar.value, 4)));
  });
  node.appendChild(el('text', {
    x: pad.l, y: height - 8, class: 'ax', fill: 'var(--ink-3)',
  }, `sd of z · largest |z| = ${fmt(data.largest_absolute_score, 2)}`));
  return node;
}

/* ------------------------------------------------------------------ S6: coverage against depth
 *
 * One point per library, coverage against log panel depth, with the nominal drawn across. The
 * pooled number is a single value; this is the thing it averages over, and it is not flat.
 */
export function coverageByDepth(rows, nominal, { width = 460, height = 250 } = {}) {
  const node = svg(width, height, 'Per-library coverage against panel depth');
  const pad = { l: 52, r: 20, t: 18, b: 46 };
  const logs = rows.map((r) => Math.log10(r.depth));
  const lo = Math.min(...logs);
  const hi = Math.max(...logs);
  const covers = rows.map((r) => r.coverage);
  const cLo = Math.min(...covers, nominal) - 0.04;
  const cHi = Math.max(...covers, nominal) + 0.03;
  const X = (d) => pad.l + ((Math.log10(d) - lo) / (hi - lo || 1)) * (width - pad.l - pad.r);
  const Y = (c) => height - pad.b - ((c - cLo) / (cHi - cLo || 1)) * (height - pad.t - pad.b);

  for (let g = 0; g <= 4; g += 1) {
    const value = cLo + ((cHi - cLo) * g) / 4;
    node.appendChild(el('line', {
      x1: pad.l, y1: Y(value), x2: width - pad.r, y2: Y(value), stroke: 'var(--rule)',
    }));
    node.appendChild(el('text', {
      x: pad.l - 8, y: Y(value) + 4, 'text-anchor': 'end', class: 'num', fill: 'var(--ink-3)',
    }, value.toFixed(2)));
  }
  node.appendChild(el('line', {
    x1: pad.l, y1: Y(nominal), x2: width - pad.r, y2: Y(nominal), stroke: 'var(--warn)',
    'stroke-width': 1.5, 'stroke-dasharray': '5 3',
  }));
  node.appendChild(el('text', {
    x: width - pad.r, y: Y(nominal) - 6, 'text-anchor': 'end', class: 'num', fill: 'var(--warn)',
  }, `nominal ${nominal.toFixed(2)}`));

  rows.forEach((row) => {
    const dot = el('circle', {
      cx: X(row.depth), cy: Y(row.coverage), r: 5,
      fill: row.coverage >= nominal ? 'var(--series-a)' : 'var(--series-b)',
      stroke: 'var(--surface)', 'stroke-width': 2,
    });
    dot.appendChild(el('title', {},
      `${row.library}: coverage ${fmt(row.coverage, 3)} at depth ${Math.round(row.depth).toLocaleString()}`));
    node.appendChild(dot);
  });
  node.appendChild(el('text', {
    x: (pad.l + width - pad.r) / 2, y: height - 8, 'text-anchor': 'middle', class: 'ax',
    fill: 'var(--ink-3)',
  }, 'replicate panel depth (log₁₀)'));
  return node;
}

export { fmt };
