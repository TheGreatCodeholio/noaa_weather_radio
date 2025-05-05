# NOAA Weather Radio — Multi‑Channel Demod / Stream / EAS‑Record

Python 3 script that turns a **single RTL‑SDR dongle** into a full‑featured  
NOAA Weather Radio receiver:

* **Up to 7 simultaneous channels** (162.400 – 162.550MHz, individually enabled in `.env`)
* **Live re‑broadcast** to Icecast (`MP3 64kb/s, ‑16 LUFS loudnorm`)
* **Automatic EAS capture** — records audio **and** decodes SAME headers with `multimon‑ng`
* **Optional upload** of completed EAS messages to a remote “Rdio” API
* **Self‑healing helpers** — `ffmpeg` and `multimon‑ng` are relaunched if they exit
* Clean shutdown on `SIGINT`/`SIGTERM`

---

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Environment Variables](#environment-variables)
5. [How it Works](#how-it-works)
6. [File Layout](#file-layout)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)

---

## Requirements
* Python ≥ `3.10`
* **rtl-sdr** drivers (`librtlsdr`) and `rtlsdr` Python bindings
* `ffmpeg` ≥ 6.0 (built with _libmp3lame_)
* `multimon‑ng`
* `numpy` / `scipy` / `python‑dotenv` / `requests`

> On Debian / Ubuntu:  
> ```bash
> sudo apt install rtl-sdr ffmpeg multimon-ng python3-pip
> pip install -r requirements.txt
> ```

---

## Installation

Below is a **turn‑key recipe** that:

1. Creates a **Python virtual‑environment** (`.venv`) so system packages stay clean
2. Installs all Python dependencies inside that venv
3. Places the project in */opt/noaa_weather_radio* ready for a systemd service

```bash
# 1) Get root or prepend sudo to the mkdir/cp commands
sudo mkdir -p /opt
sudo chown $USER /opt

# 2) Clone the repo
cd /opt
git clone https://github.com/TheGreatCodeholio/noaa_weather_radio.git
cd noaa-weather-radio

# 3) Create & activate a Python venv
python3 -m venv .venv
source .venv/bin/activate

# 4) Install requirements into the venv
pip install --upgrade pip
pip install -r requirements.txt

# 5) Make the EAS output directory (or edit EAS_OUTPUT_DIR in .env)
mkdir -p ~/eas_full_msgs
deactivate   # venv will be activated automatically by systemd later
```

---

## Quick Start (manual)

1. Copy and edit the environment file:  
   ```bash
   cp .env_example .env
   nano .env
   ```  
   *Enable at least one `CH_N_ENABLED=1`.*

2. Test‑run in the foreground:

   ```bash
   source .venv/bin/activate
   python3 nws_radio.py
   ```

> Press **`Ctrl‑C`** to exit; helpers and SDR close cleanly.

If audio and EAS recording work, continue with the **systemd service** below.

---

## Running at Boot with systemd

1. Copy the service file and adjust **User**, **Group**, and paths if needed:

   ```bash
   sudo cp nws-radio.service /etc/systemd/system/nws-radio.service
   sudo nano /etc/systemd/system/nws-radio.service
   ```

   Key lines to verify:

   ```ini
   User=YOUR_LINUX_USER
   Group=YOUR_LINUX_GROUP
   EnvironmentFile=/opt/noaa_weather_radio/.env
   ExecStart=/opt/noaa_weather_radio/.venv/bin/python /opt/noaa_weather_radio/nws_radio.py
   ```

2. Reload systemd and enable + start the service:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now nws-radio.service
   ```

3. Check its status and logs:

   ```bash
   sudo systemctl status noaa-radio
   journalctl -u nws_radio -f
   ```

The script now starts automatically at boot, launches inside its dedicated
virtual‑environment, and restarts on failure thanks to **systemd**.

---

## Environment Variables
Every configurable option lives in `.env`. Below is a condensed reference.
 
Every variable is case‑sensitive and read **once at startup**.

| Variable                             | Default / Example                          | Description                                |
|--------------------------------------|--------------------------------------------|--------------------------------------------|
| **Global SDR & Audio**               |                                            |                                            |
| `RTL_DEVICE_INDEX`                   | `0`                                        | RTL‑SDR dongle index (integer)             |
| `RTL_PPM`                            | `0`                                        | Frequency correction in ppm                |
| `RTL_GAIN`                           | `38`                                       | Gain in dB                                 |
| `CENTER_FREQ`                        | `162575000`                                | Optional override of tuner centre (Hz)     |
| `AUDIO_RATE`                         | `22050`                                    | Output sample‑rate sent to helpers (Hz)    |
| `EAS_OUTPUT_DIR`                     | `~/eas_full_msgs`                          | Folder for captured EAS WAV + TXT files    |
| **Per‑Channel (replace N with 1–7)** |                                            |                                            |
| `CH_N_ENABLED`                       | `0` / `1`                                  | Enable (1) or disable (0) channel _N_      |
| `CH_N_NAME`                          | `NWR162400`                                | Friendly name shown in logs & filenames    |
| **Icecast streaming**                |                                            |                                            |
| `CH_N_ICECAST_ENABLED`               | `0`                                        | Enable live Icecast stream for channel _N_ |
| `CH_N_ICECAST`                       | `icecast://source:pwd@host:port/mount`     | Full Icecast mount URL                     |
| `CH_N_ICECAST_TITLE`                 | `NOAA Weather Radio`                       | Stream title metadata                      |
| `CH_N_ICECAST_DESC`                  | `Live Wx radio rebroadcast`                | Stream description metadata                |
| **Rdio (alert upload)**              |                                            |                                            |
| `CH_N_RDIO_ENABLED`                  | `0`                                        | Enable upload of completed EAS recordings  |
| `CH_N_RDIO_URL`                      | `https://rdio.example.com/api/call-upload` | Rdio API endpoint                          |
| `CH_N_RDIO_KEY`                      | `rdio-api-key`                             | API key for authentication                 |
| `CH_N_RDIO_SYSTEM`                   | `1`                                        | Numeric system ID RDIO system              |
| `CH_N_RDIO_TALKGROUP`                | `101`                                      | Numeric talkgroup ID from RDIO             |

See the **full example** in this repo for every field.

---

## How it Works

```text
                           IQ 2.4 Ms/s
        ┌──────────────────────────────────────────┐
        │                  RTL‑SDR                 │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
              ┌────────────────────────────┐
              │  Python DSP  (per channel) │
              │  • Channel mixer           │
              │  • WBFM → 22kHz audio     │
              └──────────┬───────┬────────┘
                         │       │
                         │       │
                         │       │
                         │       ▼
                         │  ┌────────────┐
                         │  │ multimon‑ng│
                         │  └────┬───────┘
                         │       │
                         │       │ SAME / EAS
                         │       ▼
┌──────────────┐         │   ┌─────────────┐
│   ffmpeg     │◄────────┘   │   ffmpeg    │
└────┬─────────┘             └────┬────────┘
     │                            │
     │  MP3 stream                │ Completed alert (.wav + .txt)
     ▼                            ▼
 ┌──────────┐                ┌──────────┐    
 │ icecast  │                │   RDIO   │
 └──────────┘                └──────────┘

```

* **DSP pipeline**: FIR LPF → decimate → FM discriminator → deemphasis → resample
* **Audio fan‑out**: one raw PCM stream is sent to `ffmpeg`, `multimon‑ng`, and (during alerts) the WAV recorder simultaneously.

---

## File Layout
```text
.
├─ nws_radio.py
├─ nws-radio.service
├─ .env_example
├─ requirements.txt
└─ eas_full_msgs/
├─ EAS_NWR162550_20250501_140102.wav
└─ EAS_NWR162550_20250501_140102.txt
```

The `.txt` sidecar contains the SAME header lines captured between `ZCZC` … `NNNN`.

---

## Contributing
Pull requests are welcome!  Please open an issue first to discuss significant changes.

1. Fork the project
2. Create a feature branch:  
   ```bash
   git checkout -b feature/foo
   ```
3. Commit & push
4. Open a PR describing your changes

---

## License
MIT — see `LICENSE` for full text.
