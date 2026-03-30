from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path


class SpaRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        requested = self.translate_path(self.path)
        if self.path == "/" or Path(requested).exists():
            return super().do_GET()

        self.path = "/index.html"
        return super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a SPA build directory with index fallback.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--dir", default="frontend/dist")
    args = parser.parse_args()

    build_dir = Path(args.dir).resolve()
    handler = lambda *h_args, **h_kwargs: SpaRequestHandler(*h_args, directory=str(build_dir), **h_kwargs)

    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving {build_dir} at http://{args.host}:{args.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
