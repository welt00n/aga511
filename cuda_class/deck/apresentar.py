"""Serves the deck to the laptop's browser and the phone remote to the phone, on the same network."""
import http.server
import json
import os
import re
import socket
import sys
import threading
import webbrowser

DECK_FOLDER = os.path.dirname(os.path.abspath(__file__))
FIRST_PORT = 8765
PORTS_TO_TRY = 10
KNOWN_COMMANDS = {"next", "prev", "first", "black"}
RANGE = re.compile(r"bytes=(\d*)-(\d*)")
COPY_BLOCK = 1 << 20


class Presentation:
    def __init__(self):
        self.lock = threading.Lock()
        self.waiting_commands = []
        self.state = {}

    def send(self, command):
        with self.lock:
            self.waiting_commands.append(command)

    def take_commands(self):
        with self.lock:
            taken, self.waiting_commands = self.waiting_commands, []
        return taken

    def publish(self, state):
        with self.lock:
            self.state = state

    def current(self):
        with self.lock:
            return self.state


def byte_range(header, size):
    match = RANGE.fullmatch(header.strip()) if header else None
    if not match or not (match.group(1) or match.group(2)):
        return None
    if not match.group(1):
        return max(0, size - int(match.group(2))), size - 1
    first = int(match.group(1))
    last = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
    return (first, last) if first <= last < size else None


def handler_for(presentation):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=DECK_FOLDER, **kwargs)

        def do_GET(self):
            if self.path == "/comandos":
                return self.reply_json(presentation.take_commands())
            if self.path == "/estado":
                return self.reply_json(presentation.current())
            if self.path in ("/", "/celular"):
                self.path = "/celular.html"
            if self.headers.get("Range") and self.send_part_of_file():
                return
            super().do_GET()

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
            if self.path == "/comando" and body in KNOWN_COMMANDS:
                presentation.send(body)
                return self.reply_json("ok")
            if self.path == "/estado":
                presentation.publish(json.loads(body))
                return self.reply_json("ok")
            self.send_error(404)

        def reply_json(self, value):
            data = json.dumps(value).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_part_of_file(self):
            path = self.translate_path(self.path)
            if not os.path.isfile(path):
                return False
            size = os.path.getsize(path)
            wanted = byte_range(self.headers["Range"], size)
            if wanted is None:
                return False
            first, last = wanted
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", "bytes %d-%d/%d" % (first, last, size))
            self.send_header("Content-Length", str(last - first + 1))
            self.end_headers()
            with open(path, "rb") as file:
                file.seek(first)
                left = last - first + 1
                while left > 0:
                    block = file.read(min(COPY_BLOCK, left))
                    if not block:
                        break
                    self.wfile.write(block)
                    left -= len(block)
            return True

        def end_headers(self):
            self.send_header("Accept-Ranges", "bytes")
            super().end_headers()

        def log_message(self, *args):
            pass

    return Handler


def main_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            return [probe.getsockname()[0]]
    except OSError:
        return []


def network_addresses():
    try:
        all_addresses = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        all_addresses = []
    found = []
    for address in main_address() + all_addresses:
        if not address.startswith("127.") and address not in found:
            found.append(address)
    return found


def open_server(presentation):
    handler = handler_for(presentation)
    for port in range(FIRST_PORT, FIRST_PORT + PORTS_TO_TRY):
        try:
            return http.server.ThreadingHTTPServer(("", port), handler)
        except OSError:
            continue
    sys.exit("nenhuma porta livre entre %d e %d" % (FIRST_PORT, FIRST_PORT + PORTS_TO_TRY - 1))


def show_addresses(port):
    print("A aula, neste computador:  http://localhost:%d/aula.html" % port)
    addresses = network_addresses()
    if not addresses:
        print("Nenhuma rede encontrada: ligue o hotspot do celular e conecte o laptop a ele.")
    for address in addresses:
        print("No celular, na mesma rede: http://%s:%d/" % (address, port))
    print("Teclado: F tela cheia, S notas, Esc visão geral, B tela preta. Ctrl+C encerra.", flush=True)


def main():
    server = open_server(Presentation())
    port = server.server_address[1]
    show_addresses(port)
    if "--sem-navegador" not in sys.argv:
        webbrowser.open("http://localhost:%d/aula.html" % port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrado")


if __name__ == "__main__":
    main()
