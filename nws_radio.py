#!/usr/bin/env python3
"""
noaa_weather_radio.py – Multi-channel NOAA Weather Radio demod/stream/EAS-record
=======================================================================

* 1 RTL-SDR dongle, 7 NOAA frequencies (choose which in .env)
* Per-channel:
      IQ  ➔  WBFM demod  ➔  22 050 Hz PCM
      ├──> ffmpeg → Icecast
      └──> multimon-ng (EAS) → auto-recorder (WAV+TXT)

Author  : Ian Carey
Updated : 2025-05-01 (“record on EAS” version)
License : MIT
"""

from __future__ import annotations
import requests, threading
from datetime import datetime, timezone
import os, sys, queue, signal, subprocess, threading, itertools
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np, scipy.signal as sig
from dotenv import load_dotenv
from rtlsdr import RtlSdr, rtlsdr

# ────────────── Configuration ────────────────────────────────────────────
load_dotenv()

DEVICE = int(os.getenv("RTL_DEVICE_INDEX", 0))
PPM = int(os.getenv("RTL_PPM", 0))
GAIN = int(os.getenv("RTL_GAIN", 38))
FS_IQ = 2_400_000  # dongle sample-rate (Hz)
CENTER = 162_575_000  # middle of NOAA block
FS_AUDIO = 22_050  # stream sample-rate (Hz)
CHUNK_IQ = 32_768  # samples per USB read
BUF_MAX = 10  # IQ chunks queued per thread

# FILE OUTPUT
OUTPUT_DIR = Path(os.getenv("EAS_OUTPUT_DIR", "~/eas_full_msgs")).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NOAA = [162_400_000, 162_425_000, 162_450_000,
        162_475_000, 162_500_000, 162_525_000, 162_550_000]

# DSP constants
DECIM = 10  # 2.4 MHz → 240 kHz
FS_INTER = FS_IQ // DECIM
DEEMP_TAU = 75e-6  # 75 µs deemphasis (NOAA voice)
DEEMP_ALPHA = np.exp(-1 / (FS_AUDIO * DEEMP_TAU))

# FIR for 14 kHz LPF before decimation
LPF = sig.firwin(129, 14_000, fs=FS_IQ)


def quad_discriminator(iq: np.ndarray) -> np.ndarray:
    """angle(iq[n] · conj(iq[n-1]))"""
    return np.angle(iq[1:] * np.conj(iq[:-1]))


def deemphasis(x: np.ndarray, zi: float) -> tuple[np.ndarray, float]:
    y, zo = sig.lfilter([1 - DEEMP_ALPHA], [1, -DEEMP_ALPHA], x, zi=[zi])
    return y, zo[0]


@dataclass
class ChanCfg:
    freq: int
    name: str
    icecast_enabled: bool
    icecast_url: str | None
    icecast_title: str | None
    icecast_desc: str | None
    rdio_enabled: bool
    rdio_url: str
    rdio_key: str
    rdio_system: str
    rdio_talkgroup: str
    rdio_frequency: str

def enabled_channels() -> list[ChanCfg]:
    chan_cfgs = []
    for i, freq in enumerate(NOAA, 1):
        if os.getenv(f"CH_{i}_ENABLED", "0") != "1":
            continue

        name    = os.getenv(f"CH_{i}_NAME", f"NWR{freq//1000}")
        icecast_enabled = os.getenv(f"CH_{i}_ICECAST_ENABLED", "1") == "1"
        icecast_url = os.getenv(f"CH_{i}_ICECAST", "")
        icecast_title     = os.getenv(f"CH_{i}_ICECAST_TITLE", name)
        icecast_desc      = os.getenv(f"CH_{i}_ICECAST_DESC", f"{name} live NOAA WX")

        rdio_enabled  = os.getenv(f"CH_{i}_RDIO_ENABLED", "0") == "1"
        rdio_url = os.getenv(f"CH_{i}_RDIO_URL", "")
        rdio_key = os.getenv(f"CH_{i}_RDIO_KEY", "")
        rdio_system = os.getenv(f"CH_{i}_RDIO_SYSTEM", "")
        rdio_talkgroup = os.getenv(f"CH_{i}_RDIO_TALKGROUP", str(i))
        rdio_frequency = str(freq),

        if rdio_enabled and (not rdio_url or not rdio_key or not rdio_system):
            print(f"⚠️  CH_{i}: RDIO upload enabled but URL/key missing – disabling")
            rdio_enabled = False

        if icecast_enabled and not icecast_url:
            print(f"⚠️  CH_{i} streaming enabled but no ICECAST url, disabling stream")
            icecast_enabled = False

        chan_cfgs.append(ChanCfg(freq, name, icecast_enabled, icecast_url if icecast_enabled else None, icecast_title, icecast_desc, rdio_enabled, rdio_url, rdio_key, rdio_system, rdio_talkgroup, rdio_frequency))
    return chan_cfgs


CHANNELS = enabled_channels()
if not CHANNELS:
    sys.exit("❌  No NOAA channels enabled – check your .env")

# ────────────── DSP helpers ──────────────────────────────────────────────
LPF = sig.firwin(129, 15_000, fs=FS_IQ)  # ±15 kHz for WBFM audio


def fm_demod(iq: np.ndarray) -> np.ndarray:
    ph = np.angle(iq)
    return np.unwrap(np.diff(ph)).astype(np.float32)


# ────────────── Per-channel worker ───────────────────────────────────────
class Channel(threading.Thread):
    """
    A single NOAA frequency:
        ├─ demodulates IQ blocks from rtl_reader
        ├─ streams audio to Icecast via ffmpeg
        └─ pipes the same audio to multimon-ng to watch for EAS bursts
    If ffmpeg or multimon-ng exit unexpectedly, they are relaunched
    without interrupting the IQ pipeline.
    """

    def __init__(self, cfg: ChanCfg, iq_q: "queue.Queue[np.ndarray]"):
        super().__init__(daemon=True, name=cfg.name)
        self.cfg, self.q = cfg, iq_q
        self.stop = threading.Event()

        self.mixer = np.exp(
            -2j * np.pi * (cfg.freq - CENTER) * np.arange(CHUNK_IQ) / FS_IQ
        ).astype(np.complex64)

        # start helper processes
        self.ff  = self._launch_ffmpeg() if cfg.icecast_enabled else None
        self.mon = self._launch_multimon()

        # recorder state
        self.recording = False
        self.cap       = None
        self.txt: list[str] = []

    # ─── helper spawners ────────────────────────────────────────────────
    def _launch_ffmpeg(self) -> subprocess.Popen:
        return subprocess.Popen(
            ["ffmpeg", "-re", "-hide_banner", "-loglevel", "error",
             "-f", "s16le", "-ar", str(FS_AUDIO), "-ac", "1", "-i", "pipe:0",
             "-af", "highpass=f=300,lowpass=f=3000,afftdn,"
                    "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-c:a", "libmp3lame", "-b:a", "64k", "-content_type", "audio/mpeg",
             "-ice_name", self.cfg.icecast_title,
             "-ice_description", self.cfg.icecast_desc,
             "-ice_genre", "Weather",
             "-f", "mp3", self.cfg.icecast_url],
            stdin=subprocess.PIPE
        )

    def _launch_multimon(self) -> subprocess.Popen:
        mon = subprocess.Popen(
            ["multimon-ng", "-t", "raw", "-a", "EAS", "-q", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=False
        )
        threading.Thread(target=self._watch_mon, args=(mon,), daemon=True).start()
        return mon

    def _ensure_alive(self):
        """Restart helpers if they have terminated."""
        if self.cfg.icecast_enabled and self.ff and self.ff.poll() is not None:        # ffmpeg died
            print(f"[{self.cfg.name}] 🔄 restarting ffmpeg")
            self.ff = self._launch_ffmpeg()

        if self.mon.poll() is not None:       # multimon-ng died
            print(f"[{self.cfg.name}] 🔄 restarting multimon-ng")
            self.mon = self._launch_multimon()

    # ─── multimon watcher ───────────────────────────────────────────────
    def _watch_mon(self, mon_proc: subprocess.Popen):
        for raw in mon_proc.stdout:
            if self.stop.is_set():
                break

            try:
                line = raw.decode("utf‑8", errors="ignore").strip()
            except AttributeError:
                line = raw.strip()

            if "ZCZC" in line and not self.recording:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                base = f"EAS_{self.cfg.name}_{ts}"
                self.wav      = OUTPUT_DIR / f"{base}.wav"
                self.txt_path = OUTPUT_DIR / f"{base}.txt"

                self.cap = subprocess.Popen(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-f", "s16le", "-ar", str(FS_AUDIO), "-ac", "1", "-i", "pipe:0",
                     "-af", "highpass=f=300,lowpass=f=3000,afftdn,"
                            "loudnorm=I=-16:TP=-1.5:LRA=11",
                     self.wav],
                    stdin=subprocess.PIPE)
                self.recording = True
                self.txt = [line.rstrip()]
                print(f"[{self.cfg.name}] ▶️ start {self.wav}")

            elif "NNNN" in line and self.recording:
                self._finish_record()
            elif self.recording and line.strip() not in ("ZCZC", "NNNN"):
                self.txt.append(line.rstrip())

    # ─── finish EAS recording ───────────────────────────────────────────
    def _finish_record(self):
        self.recording = False

        if self.cap and self.cap.stdin:
            self.cap.stdin.close()
            self.cap.wait()

        self.txt_path.write_text("\n".join(self.txt))
        print(f"[{self.cfg.name}] 📝 saved {self.txt_path}")
        self._upload_to_rdio(self.wav)

        self.cap = None
        self.txt = []

    # ─── RDIO Uploader ───────────────────────────────────────────
    def _upload_to_rdio(self, wav_path: Path):
        if not self.cfg.rdio_enabled:
            return
        def _worker():
            ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            files = {
                "audio": (wav_path.name, wav_path.open("rb"), "audio/x-wav")
            }
            data = {
                "audioName": wav_path.name,
                "audioType": "audio/x-wav",
                "dateTime": ts,
                "frequencies": "[]",
                "frequency": self.cfg.rdio_frequency,
                "key": self.cfg.rdio_key,
                "patches": "[]",
                "source": -1,
                "sources": "[]",
                "system": self.cfg.rdio_system,
                "talkgroup": self.cfg.rdio_talkgroup
            }
            try:
                r = requests.post(self.cfg.rdio_url, files=files, data=data, timeout=20)
                r.raise_for_status()
                print(f"[{self.cfg.name}] ☁️  uploaded to rdio ({r.status_code})")
            except Exception as exc:
                print(f"[{self.cfg.name}] ⚠️  rdio upload failed: {exc}")
        threading.Thread(target=_worker, daemon=True).start()

    # ─── main loop ──────────────────────────────────────────────────────
    def run(self):
        print(f"✅ {self.cfg.name} streaming → {self.cfg.icecast_url}")
        while not (self.stop.is_set() or STOP.is_set()):
            try:
                iq = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            audio = self._process(iq)
            self._send(audio)

        # graceful shutdown
        for p in (self.ff, self.mon, self.cap):
            if p and p.poll() is None:
                if p.stdin:
                    p.stdin.close()
                p.wait()

    # ─── DSP pipeline ───────────────────────────────────────────────────
    def _process(self, iq: np.ndarray) -> bytes:
        bb = sig.lfilter(LPF, 1, iq * self.mixer)[::DECIM]
        audio_f = quad_discriminator(bb).astype(np.float32)
        audio_f = sig.resample_poly(audio_f, FS_AUDIO, FS_INTER)
        if not hasattr(self, "_deemp_zi"):
            self._deemp_zi = 0.0
        audio_f, self._deemp_zi = deemphasis(audio_f, self._deemp_zi)
        pcm = np.int16(np.tanh(audio_f * 5.0) * 32767).tobytes()
        return pcm

    # ─── fan-out to helpers ────────────────────────────────────────────
    def _send(self, pcm: bytes):
        self._ensure_alive()

        # ── live Icecast stream ───────────────────────────
        if self.ff and self.ff.stdin:
            try:
                self.ff.stdin.write(pcm)
            except (BrokenPipeError, ValueError):
                pass

        # ── multimon‑ng decoder ──────────────────────────
        if self.mon and self.mon.stdin:
            try:
                self.mon.stdin.write(pcm)
            except BrokenPipeError:
                # will be relaunched by _ensure_alive() next round
                pass

        # ── active EAS capture ───────────────────────────
        if self.recording and self.cap and self.cap.stdin:
            try:
                self.cap.stdin.write(pcm)
            except (BrokenPipeError, ValueError):
                # pipe already closed – finalise and reset flags
                self._finish_record()


# ────────────── RTL-SDR reader thread ───────────────────────────────────
class SDRReader(threading.Thread):
    def __init__(self, chans: list[Channel]):
        super().__init__(daemon=True);
        self.chans = chans

    def run(self):
        try:
            sdr = RtlSdr(device_index=DEVICE)
        except Exception as e:
            print(f"❌  SDR open failed: {e}")
            STOP.set()
            return

        sdr.sample_rate, sdr.center_freq, sdr.gain = FS_IQ, CENTER, GAIN
        if PPM:
            try:
                sdr.freq_correction = PPM
            except rtlsdr.LibUSBError:
                print("⚠️  PPM unsupported on this dongle")
                print("🛰️  RTL-SDR streaming… Ctrl-C to quit")

        try:
            while not STOP.is_set():
                iq = sdr.read_samples(CHUNK_IQ)
                for c in self.chans:
                    if c.q.qsize() < BUF_MAX: c.q.put_nowait(iq)
        finally:
            sdr.close()
            for c in self.chans:
                c.stop.set()
            STOP.set()


# ────────────── Main ────────────────────────────────────────────────────
STOP = threading.Event()


def main():
    chans = []
    for cfg in CHANNELS:
        q: "queue.Queue[np.ndarray]" = queue.Queue(BUF_MAX)
        c = Channel(cfg, q)
        c.start()
        chans.append(c)

    sdr = SDRReader(chans)
    sdr.start()

    def _sig(*_):
        STOP.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        while not STOP.is_set(): signal.pause()
    except KeyboardInterrupt:
        STOP.set()

    sdr.join();
    [c.join() for c in chans]
    print("👋  Exit clean.")


if __name__ == "__main__":
    main()