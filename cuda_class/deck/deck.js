const PARTS = [
  'História: de onde vêm as ideias',
  'Arquitetura: o que há dentro de uma GPU',
  'Terminologia e memória',
  'O fluxo de execução',
  'O modelo de programação',
  'Números aleatórios em paralelo',
  'EP3: a agulha de Buffon',
  'Mais fundo: warps, registradores, ocupação',
  'EP4: a estrela manchada',
  'EP4: Amdahl e a comparação',
  'Fechamento',
];

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

function buildDeck() {
  document.querySelectorAll('section.divider').forEach(fillDivider);
  document.querySelectorAll('.era-strip').forEach(fillEraStrip);
  document.querySelectorAll('.agenda-here').forEach((place) => place.replaceWith(agendaList(0)));
  OCCUPANCY_ROWS.forEach(drawWarpSlots);
}

buildDeck();
