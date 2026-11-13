"""Laufbass Vorbereiten - Mac-App (pywebview-Fenster um laufbass_prep.py).

Startet laufbass_prep.py pro Song als Unterprozess, liest dessen Ausgabe
("Schritt n/5: ...") und meldet Fortschritt an die Oberflaeche (ui.html).
"""
from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREP = HERE / "laufbass_prep.py"
PRACTICE_APP = HERE / "practice" / "index.html"
ICON = HERE / "icon.png"
SETTINGS = Path.home() / ".laufbass" / "app-settings.json"
AUDIO_EXT = {".mp3", ".m4a", ".wav", ".aif", ".aiff", ".flac", ".aac", ".ogg"}
STEP_RE = re.compile(r"Schritt\s+(\d+)\s*/\s*(\d+)\s*:\s*(.*)")

# ffmpeg & Co. aus Homebrew finden (Apps erben kein Terminal-PATH)
os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        return {"outdir": ""}


def save_settings(s: dict) -> None:
    try:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(s))
    except Exception:
        pass


def summarize(zip_path: Path) -> dict:
    try:
        with zipfile.ZipFile(zip_path) as z:
            song = json.loads(z.read("song.json"))
        return {
            "tempo": round(song.get("tempo") or 0),
            "notes": len(song.get("bass", {}).get("notes", [])),
            "hits": len(song.get("drums", {}).get("hits", [])),
            "duration": song.get("duration") or 0,
            "beatMode": song.get("beatMode", ""),
        }
    except Exception:
        return {}


class Api:
    """Methoden sind aus JavaScript als window.pywebview.api.<name>() aufrufbar."""

    def __init__(self) -> None:
        self.window = None
        self.jobs: dict[int, dict] = {}
        self.queue: "queue.Queue[int]" = queue.Queue()
        self.next_id = 1
        self.current: subprocess.Popen | None = None
        self.current_id: int | None = None
        self.settings = load_settings()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- an die Oberflaeche senden ---------------------------------------
    def _emit(self, ev: dict) -> None:
        if self.window is None:
            return
        try:
            self.window.evaluate_js("window.onLaufbassEvent(%s)" % json.dumps(ev))
        except Exception:
            pass

    # ---- Aufrufe aus der Oberflaeche -------------------------------------
    def status(self) -> dict:
        ffmpeg = any((Path(p) / "ffmpeg").exists() for p in os.environ["PATH"].split(":"))
        beat = (Path.home() / ".cache/torch/hub/checkpoints/beat_this-final0.ckpt").exists()
        return {"ffmpeg": ffmpeg, "beatModel": beat, "outdir": self.settings.get("outdir", ""),
                "practice": PRACTICE_APP.exists()}

    def choose_files(self) -> list:
        import webview
        res = self.window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=True,
            file_types=("Audio (*.mp3;*.m4a;*.wav;*.aif;*.aiff;*.flac;*.aac)", "Alle Dateien (*.*)"))
        return self.add(list(res or []))

    def choose_outdir(self) -> str:
        import webview
        res = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if res:
            self.settings["outdir"] = res[0] if isinstance(res, (list, tuple)) else res
            save_settings(self.settings)
        return self.settings.get("outdir", "")

    def reset_outdir(self) -> str:
        self.settings["outdir"] = ""
        save_settings(self.settings)
        return ""

    def add(self, paths: list) -> list:
        added = []
        for p in paths:
            path = Path(p).expanduser()
            if path.is_dir():
                files = sorted(f for f in path.iterdir() if f.suffix.lower() in AUDIO_EXT)
            else:
                files = [path]
            for f in files:
                if f.suffix.lower() not in AUDIO_EXT:
                    self._emit({"type": "toast", "text": f"„{f.name}“ ist keine Audiodatei."})
                    continue
                jid = self.next_id
                self.next_id += 1
                outdir = Path(self.settings["outdir"]) if self.settings.get("outdir") else f.parent
                job = {"id": jid, "name": f.stem, "file": str(f), "outdir": str(outdir),
                       "zip": str(outdir / f"{f.stem}.laufbass.zip"), "state": "wartet"}
                self.jobs[jid] = job
                added.append(job)
                self._emit({"type": "added", "job": job})
                self.queue.put(jid)
        return added

    def cancel(self, jid: int) -> None:
        job = self.jobs.get(jid)
        if not job:
            return
        if self.current_id == jid and self.current and self.current.poll() is None:
            job["cancelled"] = True
            try:
                self.current.send_signal(signal.SIGINT)
                time.sleep(1.5)
                if self.current.poll() is None:
                    self.current.kill()
            except Exception:
                pass
        elif job["state"] == "wartet":
            job["state"] = "abgebrochen"
            self._emit({"type": "state", "id": jid, "state": "abgebrochen"})

    def reveal(self, path: str) -> None:
        subprocess.run(["open", "-R", path])

    def open_practice(self) -> None:
        if PRACTICE_APP.exists():
            subprocess.run(["open", str(PRACTICE_APP)])

    def open_log(self, jid: int) -> str:
        return self.jobs.get(jid, {}).get("log", "")

    # ---- Abarbeitung ------------------------------------------------------
    def _worker(self) -> None:
        while True:
            jid = self.queue.get()
            job = self.jobs.get(jid)
            if not job or job["state"] != "wartet":
                continue
            self._run(job)

    def _run(self, job: dict) -> None:
        jid = job["id"]
        job["state"] = "läuft"
        log: list[str] = []
        t0 = time.time()
        self._emit({"type": "state", "id": jid, "state": "läuft", "started": t0})
        cmd = [sys.executable, "-u", str(PREP), job["file"], "-o", job["outdir"]]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            Path(job["outdir"]).mkdir(parents=True, exist_ok=True)
            self.current = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, bufsize=1, env=env)
            self.current_id = jid
        except Exception as e:  # pragma: no cover
            job["state"] = "fehler"
            self._emit({"type": "state", "id": jid, "state": "fehler", "error": str(e)})
            return
        error = ""
        warn = ""
        buf = ""
        # tqdm schreibt mit \r - deshalb zeichenweise in Zeilen zerlegen
        for chunk in iter(lambda: self.current.stdout.read(1), ""):
            if chunk not in "\r\n":
                buf += chunk
                continue
            line = buf.strip()
            buf = ""
            if not line:
                continue
            if chunk == "\n":
                log.append(line)
            m = STEP_RE.search(line)
            if m:
                self._emit({"type": "step", "id": jid, "step": int(m.group(1)), "total": int(m.group(2)),
                            "text": m.group(3).strip(" .…")})
                continue
            pct = re.search(r"(\d{1,3})%\|", line)  # tqdm-Fortschritt (Demucs)
            if pct:
                self._emit({"type": "sub", "id": jid, "pct": int(pct.group(1))})
                continue
            if "FEHLER" in line:
                error = re.sub(r"^[\s:]*(bei\s+'[^']*'\s*:\s*)?", "", line.split("FEHLER", 1)[1])
            elif "WARNUNG" in line and not warn:
                warn = line.split("WARNUNG", 1)[1].lstrip(" :")
            if chunk == "\n":
                self._emit({"type": "log", "id": jid, "line": line})
        code = self.current.wait()
        self.current = None
        self.current_id = None
        job["log"] = "\n".join(log)
        dur = time.time() - t0
        if job.get("cancelled"):
            job["state"] = "abgebrochen"
            self._emit({"type": "state", "id": jid, "state": "abgebrochen"})
        elif code == 0 and Path(job["zip"]).exists():
            job["state"] = "fertig"
            self._emit({"type": "state", "id": jid, "state": "fertig", "seconds": dur,
                        "summary": summarize(Path(job["zip"])), "warning": warn})
        else:
            job["state"] = "fehler"
            self._emit({"type": "state", "id": jid, "state": "fehler",
                        "error": error or f"Das Skript wurde mit Code {code} beendet."})


def set_mac_identity() -> None:
    """Name und Symbol im Dock/Menue auf Laufbass setzen (sonst steht dort 'Python')."""
    try:
        from Foundation import NSBundle  # type: ignore
        from AppKit import NSApplication, NSImage  # type: ignore
        info = NSBundle.mainBundle().localizedInfoDictionary() or NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = "Laufbass Vorbereiten"
        if ICON.exists():
            img = NSImage.alloc().initWithContentsOfFile_(str(ICON))
            NSApplication.sharedApplication().setApplicationIconImage_(img)
    except Exception:
        pass


def main() -> None:
    import webview
    from webview.dom import DOMEventHandler

    api = Api()
    window = webview.create_window(
        "Laufbass Vorbereiten", url=str(HERE / "ui.html"), js_api=api,
        width=820, height=680, min_size=(560, 480), background_color="#14161b")
    api.window = window

    def on_drop(e):
        files = e.get("dataTransfer", {}).get("files", [])
        paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
        if paths:
            api.add(paths)
        else:
            api._emit({"type": "toast", "text": "Datei konnte nicht gelesen werden – bitte „Songs auswählen“ nutzen."})

    def bind(win):
        set_mac_identity()
        win.dom.document.events.dragover += DOMEventHandler(lambda e: None, True, True)
        win.dom.document.events.drop += DOMEventHandler(on_drop, True, True)

    set_mac_identity()
    webview.start(bind, window)


if __name__ == "__main__":
    main()
