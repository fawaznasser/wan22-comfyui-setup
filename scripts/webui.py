#!/usr/bin/env python3
"""
Wan Studio: a simple browser UI for Wan 2.2 on top of ComfyUI.

    python webui.py            # starts ComfyUI if needed, then opens http://127.0.0.1:7860

Standard library only. Shares its logic with generate.py.
"""
import argparse
import base64
import json
import mimetypes
import os
import random
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate as g  # noqa: E402

HERE = g.HERE
CLIENT_ID = "wanstudio-" + uuid.uuid4().hex[:8]
JOBS = {}            # prompt_id -> job dict
LOCK = threading.Lock()
STATE = {"comfy": "starting", "running": None}


# --------------------------------------------------------------------------- ComfyUI process
def comfy_up(server):
    u = urllib.parse.urlparse(server)
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((u.hostname, u.port or 80)) == 0


def ensure_comfy(cfg, log_path):
    if comfy_up(cfg["server"]):
        STATE["comfy"] = "ready"
        return None
    comfy = Path(cfg["comfy_dir"])
    py = comfy / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    print(f"Starting ComfyUI ({' '.join(cfg['launch_flags'])}) - log: {log_path}")
    logf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen([str(py), "main.py", *cfg["launch_flags"]], cwd=comfy,
                            stdout=logf, stderr=subprocess.STDOUT)

    def wait():
        while proc.poll() is None:
            if comfy_up(cfg["server"]):
                STATE["comfy"] = "ready"
                print("ComfyUI is ready.")
                return
            time.sleep(1)
        STATE["comfy"] = f"ComfyUI exited (code {proc.returncode}) - see {log_path}"
        print(STATE["comfy"])
    threading.Thread(target=wait, daemon=True).start()
    return proc


# --------------------------------------------------------------------------- websocket progress
def ws_listen(server):
    """Minimal websocket client for ComfyUI /ws progress events."""
    u = urllib.parse.urlparse(server)
    while True:
        try:
            s = socket.create_connection((u.hostname, u.port or 80), timeout=10)
            key = base64.b64encode(os.urandom(16)).decode()
            s.sendall((f"GET /ws?clientId={CLIENT_ID} HTTP/1.1\r\nHost: {u.netloc}\r\n"
                       f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                       "Sec-WebSocket-Version: 13\r\n\r\n").encode())
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    raise ConnectionError("closed during handshake")
                buf += chunk
            head, buf = buf.split(b"\r\n\r\n", 1)
            if b" 101 " not in head.split(b"\r\n")[0]:
                raise ConnectionError("upgrade refused")
            s.settimeout(None)
            reader = _Reader(s, buf)
            while True:
                op, payload = reader.frame()
                if op == 8:
                    break
                if op == 9:  # ping -> pong (client frames must be masked)
                    mask = os.urandom(4)
                    s.sendall(bytes([0x8A, 0x80 | len(payload)]) + mask +
                              bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
                elif op == 1:
                    _on_event(json.loads(payload.decode("utf-8", "replace")))
        except Exception:
            pass
        time.sleep(2)


class _Reader:
    def __init__(self, sock, buf):
        self.s, self.buf = sock, buf

    def read(self, n):
        while len(self.buf) < n:
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def frame(self):
        b1, b2 = self.read(2)
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self.read(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self.read(8))[0]
        mask = self.read(4) if b2 & 0x80 else None
        data = self.read(n)
        if mask:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        return b1 & 0x0F, data


def _on_event(msg):
    t, d = msg.get("type"), msg.get("data", {}) or {}
    pid = d.get("prompt_id")
    with LOCK:
        job = JOBS.get(pid)
        if t == "execution_start" and job:
            job["started"] = time.time()
            job["stage"] = "Loading models"
        elif t == "executing" and job and d.get("node"):
            node = job["wf"].get(str(d["node"]), {})
            job["stage"] = NODE_STAGE.get(node.get("class_type"), job.get("stage"))
            job["node_title"] = node.get("_meta", {}).get("title")
        elif t == "progress" and job:
            node = job["wf"].get(str(d.get("node")), {})
            if node.get("class_type") in ("KSampler", "KSamplerAdvanced"):
                title = node.get("_meta", {}).get("title", "")
                job["stage"] = "Sampling" + (" (high-noise expert)" if "high" in title.lower() else
                                             " (low-noise expert)" if "low" in title.lower() else "")
                job["progress"] = [d.get("value", 0), d.get("max", 1)]
                job["step"] = d.get("value", 0) + job["step_offset"].get(str(d.get("node")), 0)
        elif t == "status":
            q = d.get("status", {}).get("exec_info", {}).get("queue_remaining")
            STATE["queue"] = q


NODE_STAGE = {
    "UNETLoader": "Loading models", "UnetLoaderGGUF": "Loading models", "CLIPLoader": "Loading models",
    "VAELoader": "Loading models", "CLIPTextEncode": "Encoding prompt", "LoadImage": "Loading image",
    "WanImageToVideo": "Encoding image", "Wan22ImageToVideoLatent": "Preparing latent",
    "VAEDecode": "Decoding frames", "CreateVideo": "Encoding video", "SaveVideo": "Saving video",
}


# --------------------------------------------------------------------------- jobs
def submit(cfg, p):
    d = cfg["defaults"]

    def num(key, cast, default):
        v = p.get(key)
        return default if v in (None, "") else cast(v)

    wf_key = p.get("workflow") or ""
    image_name = None
    if p.get("image_b64"):
        raw = base64.b64decode(p["image_b64"].split(",", 1)[-1])
        name = re.sub(r"[^\w.-]", "_", p.get("image_name") or "input.png")
        tmp = Path(tempfile.gettempdir()) / f"wanstudio_{uuid.uuid4().hex[:6]}_{name}"
        tmp.write_bytes(raw)
        try:
            image_name = g.upload_image(cfg["server"], tmp)
        finally:
            tmp.unlink(missing_ok=True)
    if wf_key in ("", "auto"):
        wf_key = "i2v" if image_name else "t2v"
    mult = 32 if wf_key == "5b" or cfg["tier"] != "a14b" else 16
    a = types.SimpleNamespace(
        prompt=(p.get("prompt") or "").strip(), negative=p.get("negative") or g.NEG_DEFAULT,
        width=g.fix_dim(num("width", int, d["width"]), mult, "width"),
        height=g.fix_dim(num("height", int, d["height"]), mult, "height"),
        frames=g.fix_frames(num("frames", int, d["frames"])), steps=num("steps", int, d["steps"]),
        seed=num("seed", int, random.randint(0, 2**31 - 1)), cfg=num("cfg", float, None),
        fps=num("fps", float, None))
    if not a.prompt:
        raise g.ComfyError("Please enter a prompt.")
    wf_file, wf = g.load_workflow(cfg, wf_key)
    wf = g.build(cfg, wf, a, image_name)
    pid = g.queue(cfg["server"], wf, CLIENT_ID)

    # offsets so the two A14B samplers show one continuous step count
    offsets = {}
    for nid, n in wf.items():
        if n["class_type"] == "KSamplerAdvanced":
            offsets[nid] = n["inputs"]["start_at_step"]
    fps = a.fps or next((n["inputs"]["fps"] for n in wf.values() if n["class_type"] == "CreateVideo"), 24)
    meta = {k: getattr(a, k) for k in ("prompt", "negative", "width", "height", "frames", "steps", "seed")}
    meta.update(cfg=a.cfg, fps=fps, workflow=wf_key, image=p.get("image_name") if image_name else None)
    with LOCK:
        JOBS[pid] = {"id": pid, "wf": wf, "meta": meta, "queued": time.time(), "started": None,
                     "stage": "Queued", "progress": None, "step": 0, "total_steps": a.steps,
                     "step_offset": offsets, "status": "queued", "videos": [], "error": None}
    threading.Thread(target=watch, args=(cfg, pid), daemon=True).start()
    return pid, meta


def watch(cfg, pid):
    while True:
        time.sleep(1.5)
        try:
            h = json.loads(g.http(cfg["server"], f"/history/{pid}"))
        except g.ComfyError:
            continue
        if pid not in h:
            with LOCK:
                job = JOBS[pid]
                if job["status"] == "queued" and job["started"]:
                    job["status"] = "running"
            continue
        entry = h[pid]
        with LOCK:
            job = JOBS[pid]
        err = g.error_text(entry)
        elapsed = time.time() - (job["started"] or job["queued"])
        if err:
            with LOCK:
                job.update(status="error", error=err, elapsed=round(elapsed))
            return
        meta = dict(job["meta"], seconds=round(elapsed))
        try:
            saved = g.save_outputs(cfg["server"], entry, HERE / "videos", "wan22", meta)
        except g.ComfyError as e:
            saved, err = [], str(e)
        with LOCK:
            job.update(status="done" if saved else "error", elapsed=round(elapsed),
                       videos=[f"/videos/{urllib.parse.quote(s.name)}" for s in saved],
                       error=None if saved else (err or "Finished, but no video was produced."))
        return


def job_view(job):
    j = {k: v for k, v in job.items() if k not in ("wf", "step_offset")}
    ref = job["started"] or job["queued"]
    j["elapsed"] = job.get("elapsed") or round(time.time() - ref)
    if job["status"] == "queued" and job["started"]:
        j["status"] = "running"
    return j


def gallery():
    vids = sorted((p for p in (HERE / "videos").glob("*") if p.suffix.lower() in g.VIDEO_EXT),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in vids[:200]:
        meta = {}
        side = p.with_suffix(".json")
        if side.exists():
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
            except ValueError:
                pass
        out.append({"name": p.name, "url": f"/videos/{urllib.parse.quote(p.name)}",
                    "size_mb": round(p.stat().st_size / 1e6, 3), "mtime": p.stat().st_mtime, "meta": meta})
    return out


# --------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    cfg = None

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = (HERE / "webui.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/info":
            c = self.cfg
            self.send_json({"tier": c["tier"], "gpu": c.get("gpu"), "vram_gb": c.get("vram_gb"),
                            "defaults": c["defaults"], "workflows": list(c["workflows"]),
                            "comfy": STATE["comfy"], "queue": STATE.get("queue"),
                            "comfy_url": c["server"]})
        elif path.startswith("/api/job/"):
            pid = path.rsplit("/", 1)[-1]
            with LOCK:
                job = JOBS.get(pid)
                self.send_json(job_view(job) if job else {"error": "unknown job"}, 200 if job else 404)
        elif path == "/api/jobs":
            with LOCK:
                self.send_json([job_view(j) for j in JOBS.values() if j["status"] in ("queued", "running")])
        elif path == "/api/videos":
            self.send_json(gallery())
        elif path.startswith("/videos/"):
            self.serve_file(HERE / "videos" / Path(urllib.parse.unquote(path[8:])).name)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        try:
            p = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self.send_json({"error": "bad json"}, 400)
        if path == "/api/generate":
            if STATE["comfy"] != "ready":
                return self.send_json({"error": f"ComfyUI is not ready yet ({STATE['comfy']})."}, 503)
            try:
                pid, meta = submit(self.cfg, p)
                self.send_json({"id": pid, "meta": meta})
            except g.ComfyError as e:
                self.send_json({"error": str(e)}, 400)
        elif path == "/api/interrupt":
            try:
                g.http(self.cfg["server"], "/interrupt", b"{}", {"Content-Type": "application/json"})
                self.send_json({"ok": True})
            except g.ComfyError as e:
                self.send_json({"error": str(e)}, 502)
        elif path == "/api/delete":
            f = HERE / "videos" / Path(p.get("name", "")).name
            if f.exists() and f.suffix.lower() in g.VIDEO_EXT:
                f.unlink()
                f.with_suffix(".json").unlink(missing_ok=True)
                self.send_json({"ok": True})
            else:
                self.send_json({"error": "not found"}, 404)
        else:
            self.send_error(404)

    def serve_file(self, f: Path):
        if not f.is_file():
            return self.send_error(404)
        size = f.stat().st_size
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m:
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(f, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionResetError, BrokenPipeError):
                pass


def main():
    ap = argparse.ArgumentParser(description="Wan Studio web UI")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-comfy", action="store_true", help="don't start ComfyUI automatically")
    a = ap.parse_args()
    try:
        cfg = g.load_config()
    except g.ComfyError as e:
        sys.exit(str(e))
    (HERE / "videos").mkdir(exist_ok=True)
    Handler.cfg = cfg
    proc = None
    if a.no_comfy:
        def poll():
            while not comfy_up(cfg["server"]):
                STATE["comfy"] = "not running - start it with start_wan"
                time.sleep(2)
            STATE["comfy"] = "ready"
        threading.Thread(target=poll, daemon=True).start()
    else:
        proc = ensure_comfy(cfg, HERE / "comfyui.log")
    threading.Thread(target=ws_listen, args=(cfg["server"],), daemon=True).start()

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}"
    print(f"Wan Studio running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if proc:
            print("Stopping ComfyUI ...")
            proc.terminate()
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
