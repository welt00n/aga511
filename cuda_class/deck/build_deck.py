"""Builds aula.html (the slides) and roteiro.html (the whole script, to study and print) from parts/ and notes/.

parts/_top.html, the parts in order (NN_name.html), parts/_bottom.html. Each part's notes come from notes/NN_name.md:
one block per slide, opened by "## <the start of the slide's title>" ("## divisor" for the part's divider). In a block:
  • a line starting with "•" is a cue: a few words to glance at (shown big on the tablet);
  • a line starting with "▶" is an action: click, ask, wait;
  • any other line is the script: what to say, in full.
**text** is bold. Edit the parts and the notes, then run: python3 build_deck.py"""
import glob
import html as html_text
import os
import re

DECK = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(DECK, "parts")
NOTES = os.path.join(DECK, "notes")
SECTION = re.compile(r"<section.*?</section>", re.S)
ASIDE = re.compile(r'<aside class="notes">.*?</aside>', re.S)
CUE, ACTION = "•", "▶"
PART_NAMES = {
    "00_opening": "Abertura", "01_history": "1. História", "02_hook": "O gancho", "03_architecture": "2. Arquitetura",
    "04_execution_model": "3. Guia I: o modelo de execução", "05_floating_point": "4. Guia II: ponto flutuante",
    "06_random_numbers": "5. Números aleatórios em paralelo", "07_buffon": "6. EP3: a agulha de Buffon",
    "08_memory": "7. Guia III: a memória", "09_occupancy": "8. Guia IV: a ocupação", "10_star": "9. EP4: a estrela",
    "11_closing": "10. Fechamento",
}


def read(path):
    with open(path, encoding="utf-8") as file:
        return file.read()


def plain(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\\(|\\\)|\\[a-z]+|[{}^_]", "", text)
    return re.sub(r"\s+", " ", html_text.unescape(text)).strip().lower()


def slide_title(section):
    for pattern in (r"<h2>(.*?)</h2>", r"<h1>(.*?)</h1>", r'class="statement"[^>]*>(.*?)<small'):
        found = re.search(pattern, section, re.S)
        if found:
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", found.group(1))).strip()
    return ""


def slide_key(section):
    return "divisor" if 'class="divider"' in section else plain(slide_title(section))


def inline(text):
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def kinds_of(lines):
    kinds = {"cues": [], "script": [], "actions": []}
    for line in (line.strip() for line in lines):
        if not line:
            continue
        if line.startswith(CUE):
            kinds["cues"].append(inline(line[len(CUE):].strip()))
        elif line.startswith(ACTION):
            kinds["actions"].append(inline(line[len(ACTION):].strip()))
        else:
            kinds["script"].append(inline(line))
    return kinds


def aside_html(kinds):
    return "\n".join(f'<div class="{kind}">' + "".join(f"<p>{line}</p>" for line in kinds[kind]) + "</div>"
                     for kind in ("cues", "script", "actions") if kinds[kind])


def read_notes(name):
    path = os.path.join(NOTES, name + ".md")
    if not os.path.exists(path):
        return []
    blocks, title, lines = [], None, []
    for line in read(path).splitlines():
        if line.startswith("## "):
            if title is not None:
                blocks.append((plain(title), kinds_of(lines)))
            title, lines = line[3:], []
        elif title is not None:
            lines.append(line)
    if title is not None:
        blocks.append((plain(title), kinds_of(lines)))
    return blocks


def part_slides(part_html, blocks, name, missing):
    unused = list(blocks)
    slides = []
    for section in SECTION.findall(part_html):
        key = slide_key(section)
        found = next((block for block in unused if key.startswith(block[0])), None)
        if found:
            unused.remove(found)
        else:
            missing.append(f"{name}: {key[:50]}")
        slides.append((section, found[1] if found else {"cues": [], "script": [], "actions": []}))
    missing.extend(f"{name}: notes for '{title}' match no slide" for title, _ in unused)
    return slides


def with_notes(section, kinds):
    return ASIDE.sub(lambda _: f'<aside class="notes">\n{aside_html(kinds)}\n</aside>', section, count=1)


def script_section(number, section, kinds):
    title = "— divisor da parte —" if 'class="divider"' in section else slide_title(section)
    cues = "".join(f"<li>{cue}</li>" for cue in kinds["cues"])
    script = "".join(f"<p>{line}</p>" for line in kinds["script"])
    actions = "".join(f'<p class="action">▶ {line}</p>' for line in kinds["actions"])
    return (f'<article><h3><span class="number">{number}</span> {title}</h3>'
            + (f'<ul class="cues">{cues}</ul>' if cues else "") + script + actions + "</article>")


ROTEIRO_STYLE = """
body { margin: 0; font: 17px/1.55 "Segoe UI", Arial, sans-serif; color: #1b1b1b; background: white; }
main { max-width: 820px; margin: 0 auto; padding: 28px 16px 80px; }
h1 { font-size: 1.7em; margin: 0 0 0.2em; } h2 { font-size: 1.3em; margin: 2em 0 0.4em; border-top: 2px solid #d6d6d6; padding-top: 0.6em; }
h3 { font-size: 1.02em; margin: 1.4em 0 0.3em; } .number { display: inline-block; min-width: 2.2em; color: #2a78d6; }
ul.cues { margin: 0.2em 0 0.5em; padding: 6px 12px 6px 28px; background: #eef4fc; border-radius: 6px; }
p.action { color: #b25b00; margin: 0.3em 0; } p { margin: 0 0 0.6em; }
@media print { h2 { break-before: page; } article { break-inside: avoid; } }
"""


def write_roteiro(parts_with_slides):
    body, number = [], 0
    for name, slides in parts_with_slides:
        body.append(f"<h2>{PART_NAMES.get(name, name)}</h2>")
        for section, kinds in slides:
            number += 1
            body.append(script_section(number, section, kinds))
    html = (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>Roteiro da aula</title>'
            f"<style>{ROTEIRO_STYLE}</style></head><body><main><h1>Roteiro da aula de CUDA</h1>"
            f"<p>Para estudar antes: o roteiro completo de cada slide, com as pistas (em azul, o que aparece grande no "
            f"tablet) e as ações (▶). Gerado de notes/ por build_deck.py.</p>" + "\n".join(body)
            + "</main></body></html>")
    with open(os.path.join(DECK, "roteiro.html"), "w", encoding="utf-8") as file:
        file.write(html)


def build():
    parts = sorted(glob.glob(os.path.join(PARTS, "[0-9]*.html")))
    missing, body, parts_with_slides = [], [], []
    for path in parts:
        name = os.path.splitext(os.path.basename(path))[0]
        slides = part_slides(read(path), read_notes(name), name, missing)
        parts_with_slides.append((name, slides))
        body.append(f"<!-- ================================ {name} -->\n"
                    + "\n\n".join(with_notes(section, kinds) for section, kinds in slides))
    html = read(os.path.join(PARTS, "_top.html")) + "\n".join(body) + read(os.path.join(PARTS, "_bottom.html"))
    with open(os.path.join(DECK, "aula.html"), "w", encoding="utf-8") as file:
        file.write(html)
    write_roteiro(parts_with_slides)
    print(f"aula.html and roteiro.html: {len(parts)} parts, {len(SECTION.findall(html))} slides")
    for line in missing:
        print("  without notes:", line)


if __name__ == "__main__":
    build()
