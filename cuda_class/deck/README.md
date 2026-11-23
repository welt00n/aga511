# The class deck

The slides of the class as a web page (reveal.js), presented from the laptop, with a tablet as the presenter screen and remote. Everything it needs is in this folder; nothing loads from the internet.

## Presenting

1. Turn on the phone's hotspot and connect the laptop and the tablet to it. Don't rely on the room's Wi-Fi: university networks often keep devices from reaching each other.
2. On the laptop, in this folder, with **Windows** Python (not WSL: the tablet can't reach WSL's network):
   ```
   py apresentar.py
   ```
   The first time, Windows asks whether Python may use the network: allow it, or the tablet won't connect.
3. The deck opens in the browser. Press **F** for full screen.
4. On the tablet, open the address the script prints (`http://192.168.….:8765/`), in landscape.

**The tablet** shows, for the current slide:
- **pistas** (cues), big: a few words and numbers to glance at; **roteiro completo**: the full script, one tap away, if you lose the thread;
- the yellow box: **▶ actions**, what to do now (click to reveal, ask the room, wait);
- the part, the slide number, the next slide's title, and whether a click will still reveal something;
- the clock (it starts at the first ▶; tap it to start or reset), green within the part's planned time, yellow at the limit, red when late;
- buttons: back, next, black screen, start, smaller and bigger text.

Without the script, `aula.html` also opens directly in the browser; only the tablet is missing (S still opens reveal.js's own speaker view).

**To study before the class:** `roteiro.html`, the whole script, slide by slide, with the cues and the actions; it prints one part per page.

## Keys on the laptop

| key | what it does |
|---|---|
| → / space, ← | next, back (including the steps inside a slide) |
| F | full screen |
| S | reveal.js speaker view |
| Esc | overview of all slides |
| B | black screen |
| `?print-pdf` at the end of the address, then print | the deck as a PDF |

## Editing

The deck is built from parts and notes; don't edit `aula.html` or `roteiro.html` by hand.

- `parts/NN_name.html`: the slides of each part, one `<section>` per slide. A step that appears on a click has `class="fragment"`. The A30 spaces are `<span class="a30">[A30: …]</span>`.
- `notes/NN_name.md`: the notes of the same part, one block per slide, opened by `## <the start of the slide's title>` (`## divisor` for the part's divider). In a block: `• ` cue, `▶ ` action, any other line is the script; `**bold**`.
- Then: `python3 build_deck.py` (rebuilds `aula.html` and `roteiro.html`, and lists any slide without notes).
- The part names and the minutes planned for each part are in `deck.js` (`PARTS`, `PART_MINUTES`); `build_deck.py` has the names again for `roteiro.html`.
- Math is LaTeX between `\(` and `\)`; the look is in `style.css`.

## Files

| file | what it is |
|---|---|
| `parts/`, `notes/` | the sources: slides and notes per part |
| `build_deck.py` | builds `aula.html` and `roteiro.html` |
| `aula.html` | the slides (built) |
| `roteiro.html` | the full script (built) |
| `guia.html` | the handout: the CUDA guide with our code, four sections |
| `style.css`, `deck.js` | the look; the agenda, the era strip, the drawn warp slots and SM grid, the parts and their minutes |
| `remote.js` | the deck's side of the remote (only active when served by `apresentar.py`) |
| `tablet.html` | the presenter screen |
| `apresentar.py` | serves the deck and the tablet; Python standard library only |
| `prepare_media.sh` | copies the figures from the runs and cuts one rotation out of each video (run in WSL) |
| `lib/` | reveal.js 5.1.0 and KaTeX 0.16.11, from the npm registry |

The videos are git-ignored (`*.mp4`): `prepare_media.sh` rebuilds them from the runs. Bring this whole folder on a USB stick as well.
