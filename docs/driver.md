# The driver (design)

How the Python driver turns the design of [jobs_and_passes.md](jobs_and_passes.md) into commands: describing a job, running it here or at IAG, following it, bringing results back, and analysing them, including the Amdahl and Gustafson studies. **Status: architecture agreed in discussion, not implemented yet.**

## 1. Principles

1. **The engine computes and records; the driver decides and analyses.** The engine writes every result *and its record* itself, so a result counts even when the driver is not running (section 5).
2. **Remote is explicit, and light.** A remote machine is a GPU to run on, not a second workstation: it runs the engine and answers a few questions about its runs. Everything else (planning, pictures, videos, plots, fits) happens locally, after `remote fetch` (section 4).
3. **A job is a file, or a conversation.** Job files in TOML for what must be repeatable; interactive prompts for exploring, which end by saving a job file (section 3).
4. **Every run is idempotent:** jobs and sweep points alike are recognized by their fingerprint and never redone without asking ([jobs_and_passes.md §5](jobs_and_passes.md#5-identity-and-resuming-running-a-configuration-is-idempotent)).

## 2. The commands

### Local

| You want to… | Command | What happens |
|---|---|---|
| know what a job would take | **`plan`** `[job.toml] [--machine iag-a30]` | footprint, K, passes, N ↔ SNR, time, disk, on this machine or from another machine's saved profile. Runs nothing |
| start or continue a job | **`run`** `[job.toml] [--detach]` | without a file: asks (section 3). Finds or creates the job folder, shows its state, runs the missing observers |
| see what is going on | **`status`** `[job]` | all jobs, or one: observers done, current pass, last progress line, ETA, alive or not |
| stop a detached run | **`stop`** `job` | stops the engine; the pass in progress becomes "interrupted" and is redone next time |
| measure scaling | **`sweep`** `sweep.toml` | the runs of a strong or weak scaling study (section 6) |
| analyse scaling | **`scaling`** `sweep…` | Amdahl and Gustafson fits and plots, one machine or several side by side (section 7) |
| look at results | **`pictures`** / **`video`** `job` | frames or mp4 per observer, with the measured SNR vs the predicted one |
| check the code | **`check`** `[split \| passes \| ep2 \| precision]` | today's checks plus passes vs one pass |
| know this machine can run it | **`check system`** | the preflight: GPU visible, toolkit vs driver, the engine runs on this GPU and matches the CPU, free memory, other users, disk ([roadmap.md](roadmap.md#preflight-can-we-run-here-local-and-remote-asked-2026-11-21)). `run` and `sweep` call it first |
| know this machine | **`machine probe \| calibrate \| show`** | hardware, speeds, the saved profile (`show` also lists the profiles fetched from other machines) |
| tidy up | **`clean`** `[job] [--backups] [--all]` | leftovers, old backups, or a whole job |

### Remote: `remote <command> <host>`

| Command | What happens on the remote machine | What comes back |
|---|---|---|
| **`remote setup`** | checks `ssh`, `nvcc`, `nvidia-smi`, `make`, `setsid`; syncs the code; builds (`make ARCH=sm_80`) | a report of what is there and what is missing |
| **`remote run`** `job.toml` | syncs and builds if the code changed; reads the job folder; launches the engine **detached** with the missing observers | the job's state and how to follow it |
| **`remote status`** `[job]` | reads records, the last progress line and the pid | the same table as `status` |
| **`remote stop`** `job` | stops the engine | — |
| **`remote fetch`** `[job \| sweep]` | — | finished images and records, by `rsync` (only what is new), with the duplicate check |
| **`remote sweep`** `sweep.toml` | launches the sweep's runs detached | — |
| **`remote check`** `[split \| passes]` | the bit-for-bit checks, compared with `cmp` there | identical or not |
| **`remote check system`** | the preflight there, plus SSH, the repo path, the tools, time limits | `CHECK` lines, saved with the A30's profile in `machines/` |
| **`remote probe`** / **`calibrate`** | the same measurements as locally | the profile, saved locally in `machines/` |
| **`remote clean`** `[job]` | leftovers, old backups, or a job, on the server's disk | — |

- **What stays local only:** `plan`, `scaling`, `pictures`, `video`, `check ep2`, `check precision`, `machine show`. They are light, need Python and matplotlib, and work on fetched files.
- **The remote side needs no Python:** `bash`, `make`, `nvcc`, `nvidia-smi`, `rsync` or `git`, `setsid`/`nohup`. The driver talks to it over `ssh`. That removes one of the questions for IAG ([gpus.md](gpus.md#questions-for-iag)).
- **Hosts are SSH aliases** (`~/.ssh/config`), plus `machines/hosts.toml` for what SSH does not know: the repository path on the server, the `ARCH` to build for.

**One implementation behind both:** the commands are explicit, but the job logic (state, resuming, backups, the attention message) is written once, against a small `Host` interface (`run`, `read`, `list`, `move`, `launch_detached`, `is_alive`, `stop`), with a `LocalHost` and an `SSHHost`. `run` and `remote run` differ only in the host they hand to the same function. It can be tested before IAG with an `SSHHost` pointing at `localhost`.

## 3. Describing a job: files and prompts

### Job files

```toml
# EP4/jobs/2k-snr10.toml
preset = "2k"
observers = [1.5708, 1.0472, 0.5236, 0.0]
snr = 10
precision = 64
```

- **Precedence:** preset defaults < job file < command-line flags. `run jobs/2k-snr10.toml --snr 5` is a quick variation.
- **Versioned in git,** so the same job runs on the laptop and at IAG.
- **A job file is a request, not a result:** the complete configuration that was actually run is the job folder's `job.json` (with the fingerprint).

### Interactive prompts

`run` (or `plan`) without a job file, at a terminal, asks:

```
$ python star.py run
Preset [512, 1k, 2k, 4k] (1k): 2k
Observers, inclinations in radians (1.5708,1.0472,0.5236,0):
Choose by [snr, n] (snr):
SNR at the limb (10):
Precision [64, 32] (64):

Plan on this machine (laptop-rtx5070, calibrated 2026-11-21):
  footprint 5.03 GB per observer, budget 7.0 GB → K = 1, 4 passes
  N = 1.6e13 (SNR 10 at the limb, 12 at the center)
  time ≈ 18 h, disk 20.1 GB
Save as jobs/2k-snr10.toml and [start, save only, abort] (start):
```

- **Every question shows its default;** Enter accepts it.
- **It always ends by saving a job file,** so an interactive run can be repeated or taken to IAG.
- **The same prompts decide the job's fate** when it already exists ([jobs_and_passes.md §5](jobs_and_passes.md#what-happens-when-a-configuration-is-launched)): continue / rerun / abort, then the backup question.
- **Unattended** (no terminal, or `--yes`): nothing is asked. A missing value is an error, an existing job continues, a rerun backs up unless `--no-backup`.
- Plain `input()`, no extra library.

## 4. Running: local, detached, remote

| | Local | Local `--detach` | Remote |
|---|---|---|---|
| Engine runs | in the foreground | in the background | in the background, always |
| Output | the terminal and `log.txt` | `log.txt` | `log.txt` on the server |
| Survives closing | — | the terminal | the SSH session and the laptop |
| Follow it with | — | `status` | `remote status` |

- **Detaching:** `setsid nohup ./star … > log.txt 2>&1 &`, the pid in `run.pid` in the job folder. `tmux` can come later if it is installed at IAG.
- **One engine per job:** if `run.pid` names a living process, `run` refuses to start a second one ("already running, see `status`"). A dead pid is an interrupted run: cleaned and continued.
- **Who decides what to run:** the local driver, always. For a remote job it reads the server's job folder over SSH, decides which observers are missing, and launches the engine there with exactly those. Nothing on the server has to remember anything between runs: the files are the memory.

## 5. The engine writes its own records

**Why:** at IAG the driver is literally not running: the laptop is at home, closed, while the A30 works for days. The result and its record must be complete without it.

- **The driver passes in what the engine cannot know:** `--fingerprint`, `--code-hash`, and the requested inclinations (in the observer list).
- **Per observer, at the end of its pass, the engine writes:**
  1. the image, to a temporary name, then renames it;
  2. **then the record**, also temporary then renamed: `observer_mu0042_i1.5708.record`. The record is the "done" marker.
- **The record is `KEY=VALUE`, like the engine's output,** parsed by the same function:

  ```
  FINGERPRINT=1a2b3c4d…
  CODE_HASH=3f2a91c
  MU_BIN=42
  INCLINATION_REQUESTED=1.5708
  INCLINATION_BIN_CENTER=1.5708
  PASS=2/4
  DEVICE=NVIDIA A30
  PRECISION=64
  PHOTONS=… BLOCKED=… MISSED=… RECORDED=…
  PHOTONS_TIME=… SAVE_TIME=…
  FINISHED=2026-12-03T04:12:55
  ```

- This replaces the `.json` record of [jobs_and_passes.md §5](jobs_and_passes.md#on-disk); `job.json` (the complete configuration) is still written by the driver, before the first launch.

## 6. Sweeps: the runs of a scaling study

A **sweep** is a set of short runs that vary one thing, described in a TOML file like a job.

```toml
# EP4/sweeps/strong-sms.toml          Amdahl: the problem is fixed
law = "strong"
over = "sms"
sms = [1, 2, 4, 8, 12, 18, 24, 30, 36]
preset = "512"
photons = 1e10
repeats = 3
```

```toml
# EP4/sweeps/weak-sms.toml            Gustafson: the problem grows with the SMs
law = "weak"
over = "sms"
sms = [1, 2, 4, 8, 12, 18, 24, 30, 36]
preset = "512"
photons_per_sm = 3e8
repeats = 3
```

```toml
# EP4/sweeps/weak-size.toml           Gustafson with a serial part that grows
law = "weak"
over = "size"
sizes = [512, 1024, 2048]
snr = 2
repeats = 3
```

- **What can vary:** `sms` (the GPU engine's SM mask, as in EP3), `size`, `photons`, `precision`. On the CPU the engine is sequential, so a CPU sweep can vary size, photons and precision, not processors.
- **Each point is a run with its own record**, in `sweeps/<name>-<fingerprint>/point_sms08_rep2.record`. A sweep is idempotent like a job: an interrupted sweep continues with the missing points.
- **The image is recorded but not kept:** the atomics are part of the cost being measured; writing the image is timed (`SAVE_TIME`) and then the file is deleted.
- **Repeats:** each point runs `repeats` times; the analysis uses the median and shows the spread.
- **Small on purpose:** sweeps use a low SNR (or fewer frames) so the whole study fits in one session. Only ratios matter, and the time per photon does not depend on the SNR.

## 7. Scaling analysis: Amdahl and Gustafson

`scaling` reads one or more fetched sweeps and produces the figures for the TCD and the class ([scaling_laws.md](scaling_laws.md)).

| Figure | From | What it shows |
|---|---|---|
| **Strong scaling** | a strong sweep | time T(n) (log-log) and speedup S(n) = T(1)/T(n), with the ideal line and the Amdahl fit T(n) = T(1)·(s + p/n); efficiency S(n)/n; the ceiling 1/s |
| **p as a function of N** | strong sweeps at several N | Amdahl's p grows with N, as in EP3 |
| **Weak scaling** | a weak sweep | efficiency E(n) = T(1, W)/T(n, n·W), and the scaled speedup with the Gustafson fit S(n) = n − s′(n − 1) |
| **Strong and weak side by side** | both sweeps of one machine | the flattening curve vs the straight line: the class's central plot |
| **Serial part, measured vs fitted** | the engine's phase times | s′ measured directly as (setup + save) / total, against the s′ from the fit; and the same for Amdahl's s |
| **Weak over the image** | a size sweep | stacked bars of setup / photons / save per size: the serial part that grows with the image |
| **Across machines** | the same sweep on the laptop and the A30 | both curves on one figure, against n and against absolute photons per second |

- **Fits are least squares** on the medians, with the parameter's uncertainty from the repeats.
- **The serial part is not only a fitted number:** the engine reports setup, photon and save times separately, so the fit can be checked against a measurement, which is stronger than either alone.
- **Every figure is saved with its data** (a `.json` next to the `.png`), so the plots can be redrawn for the slides without rerunning anything.

## 8. Modules

`EP4/star.py` stays the entry point; the rest becomes a package, one responsibility per module.

| Module | Responsibility |
|---|---|
| `star.py` | the command line: parse, call one function per command |
| `driver/spec.py` | presets, job files, flag overrides → `JobSpec`; the fingerprint |
| `driver/prompts.py` | interactive questions with defaults; detecting unattended runs |
| `driver/jobs.py` | `Job`: folder, state from the records, backups, code hash, the attention message |
| `driver/engine.py` | building the engine command, parsing `KEY=VALUE` (output and records) |
| `driver/hosts.py` | `LocalHost`, `SSHHost` |
| `driver/remote.py` | setup, code sync, build, fetch with the duplicate check |
| `driver/planner.py` | `Plan`: footprint, K, passes, N ↔ SNR, time from a profile |
| `driver/machines.py` | `MachineProfile`: probe, calibrate, load, save |
| `driver/sweeps.py` | `SweepSpec`, its points, running the missing ones |
| `driver/scaling.py` | Amdahl and Gustafson fits and their figures |
| `driver/images.py` | loading images, pictures, video, the measured SNR |
| `driver/comparisons.py` | χ², photons moved, relative noise (today's `analysis.py`) |
| `driver/checks.py` | split, passes, EP2, precision |

The analysis is now three modules (`images`, `comparisons`, `scaling`) instead of one, because it does three different things.

## 9. Implementation order

**The overall plan and its status live in [roadmap.md](roadmap.md)** (phases A–I). This list is the driver's part of it: steps 1–2 = phase C, 3 = D, 4 = E, 5 = F, 6 = G, 7 = H.

1. **[done 2026-11-21] Engine, CPU:** passes, the `[observer][frame][iy][ix]` layout, observers in radians, one file and **one record per observer written by the engine**, `--fingerprint`, `--code-hash`, progress lines.
2. **[done 2026-11-21, in today's `star.py`] Checks:** passes vs one pass, bit for bit; the existing checks updated.
3. **Driver, local:** `spec` (presets, TOML, overrides), `prompts`, `jobs`; `run` (with `--detach`, `run.pid`), `status`, `stop`, `plan`, `clean`.
4. **Remote:** `hosts`, `remote setup / run / status / stop / fetch / clean`; `machine probe / calibrate` local and remote. Tested against `localhost`.
5. **GPU engine:** the same passes on the device, `--sms` (the SM mask), the GPU budget from `cudaMemGetInfo`.
6. **Sweeps and scaling analysis:** the machinery can be built on the CPU with size and photon sweeps; the SM sweeps arrive with the GPU engine.
7. **Later:** `video` (mp4 through ffmpeg), `pictures` with the measured SNR.

## 10. Open details

- [x] Job files in TOML **and** interactive prompts; prompts end by saving a job file.
- [x] Remote is an explicit `remote` command group, light on the server; analysis is local.
- [x] The engine writes its own records (`KEY=VALUE`).
- [x] `--detach` locally too.
- [x] Amdahl and Gustafson studies as sweeps, analysed by `scaling`.
- [ ] Sweep file fields: settle when implementing step 6.
- [ ] `tmux` instead of `setsid`/`nohup`, if IAG has it.
- [ ] The rotation length (sets the frames at 30 fps): still open.
