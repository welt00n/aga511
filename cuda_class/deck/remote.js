const COMMAND_POLL_MS = 150;

const COMMANDS = {
  next: () => Reveal.next(),
  prev: () => Reveal.prev(),
  first: () => Reveal.slide(0),
  black: () => Reveal.togglePause(),
};

function servedByPresenter() {
  return location.protocol.startsWith('http');
}

function titleOf(slide) {
  const heading = slide && slide.querySelector('h1, h2, .statement');
  if (!heading) return '';
  const visibleText = heading.cloneNode(true);
  visibleText.querySelectorAll('small, .katex-mathml').forEach((hidden) => hidden.remove());
  return visibleText.textContent.trim();
}

function slideState() {
  const slides = Reveal.getSlides();
  const current = Reveal.getCurrentSlide();
  const index = slides.indexOf(current);
  const part = partOfSlide(slides, index);
  return {
    number: index + 1,
    total: slides.length,
    title: titleOf(current),
    next: titleOf(slides[index + 1]),
    stepsLeftHere: Reveal.availableFragments().next,
    notes: Reveal.getSlideNotes(current) || '',
    paused: Reveal.isPaused(),
    part,
    parts: PARTS.length,
    partName: part ? PARTS[part - 1] : 'Abertura',
    plannedEndMinute: plannedEndMinute(part),
    classMinutes: plannedEndMinute(PART_MINUTES.length - 1),
  };
}

function publishState() {
  fetch('/estado', { method: 'POST', body: JSON.stringify(slideState()) }).catch(() => {});
}

async function runWaitingCommands() {
  try {
    const waiting = await (await fetch('/comandos')).json();
    waiting.forEach((name) => COMMANDS[name] && COMMANDS[name]());
  } catch (error) {}
}

function followThePhone() {
  if (!servedByPresenter()) return;
  ['slidechanged', 'fragmentshown', 'fragmenthidden', 'paused', 'resumed'].forEach((event) => Reveal.on(event, publishState));
  publishState();
  setInterval(runWaitingCommands, COMMAND_POLL_MS);
}
