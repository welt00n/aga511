const PARTS = [
  'História: de onde vêm as ideias',
  'Arquitetura: o que há dentro de uma GPU',
  'Guia I: o modelo de execução',
  'Guia II: ponto flutuante',
  'Números aleatórios em paralelo',
  'EP3: a agulha de Buffon',
  'Guia III: a memória',
  'Guia IV: a ocupação',
  'EP4: a estrela manchada',
  'Fechamento',
];

const PART_MINUTES = [3, 12, 6, 16, 8, 8, 14, 8, 10, 25, 5];

const ERAS = ['1999', '2001–2006', '2006–2007', '2008–2010', '2012–2017', 'hoje'];

function agendaList(current) {
  const list = document.createElement('ol');
  list.className = 'agenda';
  PARTS.forEach((name, index) => {
    const item = document.createElement('li');
    item.innerHTML = `<b>${index + 1}</b>${name}`;
    if (index + 1 === current) item.className = 'current';
    list.appendChild(item);
  });
  return list;
}

function fillDivider(section) {
  const part = Number(section.dataset.part);
  const number = document.createElement('div');
  number.className = 'part-number';
  number.textContent = `Parte ${part} de ${PARTS.length}`;
  const title = document.createElement('h2');
  title.textContent = PARTS[part - 1];
  section.prepend(title);
  section.prepend(number);
  section.appendChild(agendaList(part));
}

function fillEraStrip(strip) {
  const current = Number(strip.dataset.era);
  ERAS.forEach((name, index) => {
    const era = document.createElement('span');
    era.textContent = name;
    if (index + 1 === current) era.className = 'current';
    strip.appendChild(era);
  });
}

const SVG = 'http://www.w3.org/2000/svg';
const SLOT = { x: 250, width: 12, step: 14, height: 36 };
const SLOT_COLORS = { resident: '#2a78d6', allowed: '#a9c8ef', none: '#e4e3df' };

const OCCUPANCY_ROWS = [
  { id: 'occupancy-fp64', y: 20, maximum: 48, allowed: 25, resident: 24, text: '24 residentes de 48 → ocupação 0,50 (cabem 25, mas o bloco de 24 warps só entra inteiro)' },
  { id: 'occupancy-fp32', y: 120, maximum: 48, allowed: 28, resident: 28, text: '28 de 48 → ocupação 0,58' },
  { id: 'occupancy-a30', y: 220, maximum: 64, allowed: 21, resident: 21, text: '21 de 64 → ocupação no máximo 0,33' },
];

function svgElement(name, attributes) {
  const element = document.createElementNS(SVG, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function slotKind(index, row) {
  if (index < row.resident) return 'resident';
  return index < row.allowed ? 'allowed' : 'none';
}

function drawWarpSlots(row) {
  const group = document.getElementById(row.id);
  if (!group) return;
  for (let index = 0; index < row.maximum; index++) {
    group.appendChild(svgElement('rect', {
      x: SLOT.x + index * SLOT.step, y: row.y, width: SLOT.width, height: SLOT.height,
      fill: SLOT_COLORS[slotKind(index, row)],
    }));
  }
  const label = svgElement('text', { x: SLOT.x, y: row.y + 58, class: 'small' });
  label.textContent = row.text;
  group.appendChild(label);
}

const SM_GRID = { id: 'sm-grid', sms: 36, allowed: 12, columns: 6, width: 112, height: 64, step_x: 122, step_y: 76 };

function smTile(group, sm) {
  const x = 10 + (sm % SM_GRID.columns) * SM_GRID.step_x;
  const y = 10 + Math.floor(sm / SM_GRID.columns) * SM_GRID.step_y;
  const works = sm < SM_GRID.allowed;
  group.appendChild(svgElement('rect', {
    x, y, width: SM_GRID.width, height: SM_GRID.height, rx: 4, class: works ? 'device' : 'gray',
  }));
  const name = svgElement('text', { x: x + 10, y: y + 26, class: 'small bold' });
  name.textContent = `SM ${sm}`;
  const what = svgElement('text', { x: x + 10, y: y + 50, class: works ? 'tiny' : 'tiny muted' });
  what.textContent = works ? '768 threads' : 'volta na hora';
  group.appendChild(name);
  group.appendChild(what);
}

function drawSmGrid() {
  const svg = document.getElementById(SM_GRID.id);
  if (!svg) return;
  for (let sm = 0; sm < SM_GRID.sms; sm++) smTile(svg, sm);
  const legend = svgElement('text', { x: 10, y: 490, class: 'small muted' });
  legend.textContent = `azul: os ${SM_GRID.allowed} SMs de --sms ${SM_GRID.allowed} · cinza: o bloco lê %smid e sai`;
  svg.appendChild(legend);
}

function partOfSlide(slides, index) {
  for (let i = index; i >= 0; i--) {
    if (slides[i].classList.contains('divider')) return Number(slides[i].dataset.part);
  }
  return 0;
}

function plannedEndMinute(part) {
  return PART_MINUTES.slice(0, part + 1).reduce((sum, minutes) => sum + minutes, 0);
}

function buildDeck() {
  document.querySelectorAll('section.divider').forEach(fillDivider);
  document.querySelectorAll('.era-strip').forEach(fillEraStrip);
  document.querySelectorAll('.agenda-here').forEach((place) => place.replaceWith(agendaList(0)));
  OCCUPANCY_ROWS.forEach(drawWarpSlots);
  drawSmGrid();
}

buildDeck();
