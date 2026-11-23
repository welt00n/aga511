# The class deck

The slides of the class as a web page (reveal.js), with the phone as the remote. Everything it needs is in this folder; nothing loads from the internet.

## Presenting

1. Turn on the phone's hotspot and connect the laptop to it. Don't rely on the room's Wi-Fi: university networks often keep devices from reaching each other.
2. On the laptop, in this folder, with **Windows** Python (not WSL: the phone can't reach WSL's network):
   ```
   py apresentar.py
   ```
   The first time, Windows asks whether Python may use the network: allow it, on private and public networks, or the phone won't connect.
3. The deck opens in the browser. Press **F** for full screen.
4. On the phone, open the address the script prints (`http://192.168.….:8765/`).

The phone page has next and back, a black screen, the current slide's notes, the next slide's title and a timer (it starts at the first "next"; tap it to restart). Set the phone's screen timeout long enough before starting.

Without the script, `aula.html` also opens directly in the browser; only the phone remote is missing.

## Keys on the laptop

| key | what it does |
|---|---|
| → / space, ← | next, back (including the steps inside a slide) |
| F | full screen |
| S | speaker view: current and next slide, notes, clock |
| Esc | overview of all slides |
| B | black screen |
| `?print-pdf` at the end of the address, then print | the deck as a PDF |

## Editing

- The slides are `aula.html`: one `<section>` per slide, the notes in `<aside class="notes">`. A step that appears on a click has `class="fragment"`.
- The A30 spaces are `<span class="a30">[A30: …]</span>`: replace each with the measured value.
- Math is LaTeX between `\(` and `\)`.
- The look is in `style.css`.

## Files

| file | what it is |
|---|---|
| `aula.html` | the slides |
| `style.css` | the look |
| `remote.js` | the deck's side of the phone remote (only active when served by `apresentar.py`) |
| `celular.html` | the phone page |
| `apresentar.py` | serves both; Python standard library only |
| `prepare_media.sh` | copies the figures from the runs and cuts one rotation out of each video (run in WSL) |
| `lib/` | reveal.js 5.1.0 and KaTeX 0.16.11, from the npm registry |

The videos are git-ignored (`*.mp4`): `prepare_media.sh` rebuilds them from the runs. Bring this whole folder on a USB stick as well.
