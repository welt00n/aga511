# Jobs, observers and passes (design)

How EP4 turns "I want these observers at this size" into runs that fit any machine's memory, on the CPU and on the GPU, in one sitting, over several days, or on several machines. **Status: the engine part is implemented (section 15); the driver part is designed in [driver.md](driver.md), not implemented yet.**

## 1. Terms

| Term | Meaning |
|---|---|
| **Job** | everything one experiment asks for: the star, N, the seed, the image size, the frames, the precision, the list of observers. Lives in a folder with a `job.json`. |
| **Observer** | one line of sight, given by its **inclination i in radians** (0 = pole-on from +z, π/2 = equator-on, π = pole-on from −z). Produces one image file and, later, one video. |
| **Footprint** | the memory of one observer's image: `size² × frames × 4 bytes`. |
| **Pass** | one sweep over all N photons that records **K observers** at once. Memory ≈ K × footprint. |
| **Observers per pass (K)** | set directly (`--observers-per-pass`) or derived from a memory budget (`--memory-budget`): K = ⌊budget / footprint⌋. |
| **Run** | one invocation of the engine. A run can do **one or many passes**. |

Passes = ⌈observers / K⌉. **Memory becomes a trade with time:** half the memory per pass, twice the passes.

## 2. Why passes are exact

Every pass uses the same seed and the same chunks, so it simulates **the same photons**.

- A pass recording observer 3 alone produces exactly the image a single big pass would have produced for observer 3.
- Same reasoning as the chunk range.
- Checkable bit for bit: 1 pass of 4 observers vs 4 passes of 1.

The cost: **every pass traces all N photons.** Photons toward observers not in this pass are simply counted as `MISSED`. Time ≈ passes × photon time.

## 3. Requirements (from the discussion)

1. **Observers as a list of inclinations, in radians.**
2. **Memory set by budget or by K**, whichever the user prefers; the budget default will come from the machine's probe.
3. **Passes can run in one go or in pieces:**
   - launch the whole job on a server, go home, find many passes done days later;
   - or run some passes today (3 hours at IAG), more tomorrow, and join the results later;
   - or split the observers between machines at the same time.
4. **One file per observer,** image layout `[observer][frame][iy][ix]`, implemented now.
5. **The same on the CPU and the GPU.** On the CPU the budget is RAM; on the GPU, device memory.
6. **Running a configuration is idempotent:** what is done is recognized and not redone (section 5).
7. **Remote runs over SSH, now:** choose a configuration at IAG, hit run, go home (section 7).
8. **Not now:** resuming in the middle of a pass (by chunk). N is fixed per job; the chunk range already exists if it is ever needed.

## 4. Who does what

### The engine: does the passes it is given

```
./star -n 1e9 --size 1024 --frames 300 --nmu 180 \
       --observers 1.5708,1.0472,0.5236,0 --observers-per-pass 2 \
       --output-dir runs/job/
```

- **Takes the observer list** (inclinations in radians), maps each to its μ bin, and refuses two observers in the same bin.
- **Allocates the image once,** for K observers, and reuses it: each pass zeroes it, simulates all chunks, saves.
- **Saves each observer as soon as its pass ends** (names in section 5).
  - It writes to a temporary name first and renames it when complete.
  - So a file that exists is always whole, even if the run is killed in the middle of the next pass.
- **Prints one `KEY=VALUE` line per pass** (pass number, observers, their μ bins, photon counts, times), and a summary line at the end.
- **The engine knows nothing about jobs or resuming:** it does exactly the observers it is told to.

### The driver: owns the job

**The commands are designed in [driver.md](driver.md).** In short:

- **`plan`:** footprint, K, passes, N ↔ SNR, time and disk, without running anything.
- **`run`:** finds or creates the job folder from the fingerprint, reports its state, asks or decides what to do (section 5), and calls the engine with the observers still missing. Running it again is how you resume: today, tomorrow, or after a crash, it is always the same command.
- **`status`:** every job, or one job's observers, pass and progress (this replaces the `jobs` command of the first sketch).
- **`clean`:** leftovers, old backups, or a whole job.

**Why the driver owns resuming:** the job folder and `job.json` are the single truth of what was asked. The engine only sees one call and cannot tell whether an existing file belongs to the same job.

## 5. Identity and resuming: running a configuration is idempotent

Given its configuration, every observer's image is determined bit for bit (same seed, same chunks, same photons). A result never needs recomputing: it only needs to be **recognized**.

### The job fingerprint

A hash of exactly the parameters that change the pixels:

| In the fingerprint (changes the image) | Not in it (only changes how we get there) |
|---|---|
| seed, N, chunk size | observers per pass (K), memory budget |
| limb darkening, spots (count, radii) | output folder, dates, number of runs |
| size, frames (`nphi`), `nmu` | the device (laptop or A30) |
| **floating-point precision of the physics (FP64 or FP32)** | the code version (see "code changes" below) |

- **Precision is part of the job:** FP32 and FP64 move ~5·10⁻⁴ of the photons to other pixels ([precision.md §7](precision.md#7-first-measurement-cpu-n--10⁸-ep2s-default-star-measured)), so they are different results, kept in different job folders. The FP32 vs FP64 comparison is then a comparison of two jobs.
- **An observer's identity** = job fingerprint + **its μ bin**. Two inclinations in the same bin (1.5708 and 1.57079) are the same observer.
- **The observer list is not in the fingerprint:** a job is "this star, this image, this N, this precision", and observers can be added to it later. Only the new ones run.

### On disk

```
runs/2k-f300-mu96-n2e13-s42-fp64-1a2b3c4d/      main parameters + fingerprint
    job.json                          the fingerprinted configuration
    observer_mu0042_i1.5708.bin        the image (temporary name, renamed when complete)
    observer_mu0042_i1.5708.record     its record, KEY=VALUE, written by the engine
    log.txt, run.pid                  the engine's output; the pid while it runs
    backups/<date>_<time>/            results set aside by a rerun, if you chose to keep them
```

- **The folder name carries the main parameters** (size, frames, μ bins, N, seed, precision) so a listing is readable, plus the first 8 characters of the fingerprint, which is what really identifies the job. An optional `--name` prefix can be added for humans.
- **The file name carries both** the μ bin (the identity) and the inclination.
  - The inclination in the name is the **center of the bin**, so the name is the same whatever inclination inside the bin was typed.
  - The inclination actually requested is in the record.
- **The record** (`.record`, written by the engine itself, [driver.md §5](driver.md#5-the-engine-writes-its-own-records)) holds:
  - the fingerprint, the requested and bin-center inclinations, the μ bin;
  - the photon counts and the times;
  - the device, the date, and the code hash.
- **The record is written last** and renamed into place: it is the commit marker. A `.bin` without its `.record`, or a leftover temporary file, means "interrupted" and is redone.

### What happens when a configuration is launched

| Situation | What we see | Behavior |
|---|---|---|
| **New** | no job with this fingerprint | create it, run all observers |
| **Already complete** | every requested observer has its record | do nothing, report it |
| **Partial** | some observers done | continue with the missing ones |
| **Interrupted mid-pass** | temporary files, or a `.bin` without its record | delete them, redo that pass |
| **More observers than before** | old ones done, new ones requested | run only the new ones |
| **Different configuration** | e.g. N, the chunk size or the precision changed | a different fingerprint is a different job folder: never mixed |
| **Code changed since the results** | the current code hash differs from the one in the records | **attention message**, see below |

**Asking or not** **[decided]:**

- **At the keyboard:** if the job already exists, the driver shows its state ("2 of 4 observers done; last run 2026-11-21 on the A30") and asks: continue / rerun / abort.
- **Unattended** (server, script, no terminal): it never asks; it continues by default.
- **`--fresh`** deletes the job and starts over, after confirming at the keyboard.

### Rerunning, and backups **[decided]**

- **`--rerun`** (or choosing "rerun" at the keyboard) redoes the requested observers.
- **First it asks whether to back up the current results:**
  - **yes:** they move to `backups/<date>_<time>/` in the job folder. Afterwards the driver compares old and new and reports whether they are identical (they should be, unless the physics changed).
  - **no:** they are overwritten.
- **Unattended,** `--backup` or `--no-backup` decides. Without either, it backs up: the safe choice when nobody is there to answer.
- **`clean`** removes old backups when they are no longer needed.

### Code changes **[decided]**

- **The code is not in the fingerprint,** because results are only invalid when the *physics* changed, and only you can judge that.
- **A code hash is recorded instead,** in every observer's record: a hash of the engine's source files (`EP4/engine/*.cu`, `*.cuh`, `Makefile`). It is computed from the files themselves, so it also catches changes that are not committed.
- **When it differs from the current code,** the driver prints an attention message before anything else, for example:

  > **Attention:** the engine code changed since these results were produced (observers mu0042, mu0098: code `3f2a91c`, now `b77e05d`). If the change affects the physics, consider rerunning them.

  At the keyboard, the question that follows offers rerun first. `status` marks such jobs too.

### Several machines **[decided: simplest]**

- **The device is recorded per observer and not checked.** Observers from the laptop and from the A30 can share a job.
- **Splitting a job between machines at the same time:** give each machine different observers of the same configuration (`--observers` with a subset).
  - The fingerprint is the same everywhere, so each machine creates the same job folder name.
  - Joining is copying observer files into one folder (`remote fetch` does it).
- **If the same observer exists on both sides,** the files should be identical. `fetch` reports it if they are not, and keeps both (one goes to `backups/`).

## 6. Examples

| Job | Footprint | Laptop (~7 GB) | A30 (~22 GB) |
|---|---|---|---|
| 1k, 300 frames, 4 observers | 1.26 GB | K = 5 → 1 pass | 1 pass |
| 2k, 300 frames, 4 observers | 5.03 GB | K = 1 → **4 passes** | K = 4 → 1 pass |
| 2k, 420 frames (~7 GB each), 4 observers | 7.05 GB | does not fit even K = 1 → **lower frames or a smaller size** | K = 3 → 2 passes |
| 4k, 300 frames, 2 observers | 20.1 GB | does not fit | K = 1 → 2 passes |

The planner says "does not fit" with the largest size or frame count that would.

**Gustafson, seen through passes:**

- **A fixed machine:** a bigger job means more passes and more time, but it never becomes impossible while one observer fits.
- **A bigger machine:** fewer passes, so the same job in less time, or a bigger job in the same time.
- **Passes per job, machine by machine,** is a measurable form of Gustafson's question ([scaling_laws.md](scaling_laws.md)).

## 7. Remote runs over SSH (now)

At IAG, the goal is: choose a configuration, hit run, go home. **Designed in [driver.md §2 and §4](driver.md#remote-remote-command-host).** In short:

```
python star.py remote setup  iag-a30
python star.py remote run    iag-a30 jobs/2k-snr10.toml
python star.py remote status iag-a30
python star.py remote fetch  iag-a30 2k-snr10
```

- **Remote is explicit and light:** the server runs the engine (detached, so it survives the SSH session) and answers status; it needs no Python. Pictures, videos and plots are made locally after `fetch`.
- **The local driver decides:** it reads the server's job folder over SSH and launches the engine there with the missing observers. The engine writes its own records, so nothing depends on the laptop staying connected.
- **The job logic is the same as locally:** fingerprints, resuming, backups, the attention message, written once against a `Host` interface.
- **Testing before IAG:** against `localhost` (an SSH server in WSL).
- **Depends on IAG:** SSH from inside the IAG network, whether `tmux` is installed, a scheduler or not ([gpus.md](gpus.md#questions-for-iag)).

## 8. Code organization and style

The rules from EP1/EP3 and [cpp_for_python.md](cpp_for_python.md):

- **Names carry the meaning;** small functions with verb-led names; almost no comments.
- **The engine computes, the driver orchestrates:** the engine reports facts in `KEY=VALUE` lines.
- **The reasoning lives in these docs,** not in the code.

### Engine (C++)

| File | New or changed |
|---|---|
| `options.cu` | `--observers` (radians), `--observers-per-pass`, `--memory-budget`, `--size`, `--frames`, `--output-dir` |
| `scene.cu` | `observer_of_imu` filled for one pass's observers: `fill_observer_table(scene, pass_observers)` |
| `image.cu` | layout `[observer][frame][iy][ix]`; `zero_image`; `save_observer` (one block of the image, temporary name then rename) |
| `photon.cuh` | `pixel_index` with the new layout; nothing else changes |
| `main.cu` | `run_passes` → `simulate_pass` → `simulate_chunks`; `print_pass_result`, `print_summary` |

The physics does not change; the photon loop does not change.

### Driver (Python)

`star.py` would grow too big, so it splits into a `driver/` package, one responsibility per module: the table is in [driver.md §8](driver.md#8-modules).

`JobSpec`, `Job`, `Plan`, `MachineProfile` as `@dataclass`es: named fields instead of dicts, so the code reads like the design.

## 9. Implementation order

**Superseded by [roadmap.md](roadmap.md)** (the one plan, phases A–I) and [driver.md §9](driver.md#9-implementation-order), which adds the engine's records, local `--detach`, and the sweeps for Amdahl and Gustafson. The engine part is unchanged:

1. **Engine, CPU:**
   - the `[observer][frame][iy][ix]` layout;
   - the observer list in radians, `--observers-per-pass`, `--memory-budget`;
   - the pass loop, one file per observer (temporary name, then rename), one result line per pass;
   - `--size`, `--frames`, `--output-dir`; `nmu` defaults to frames/π;
   - progress lines (at most every 10 s, checked once per chunk).
2. **Checks:**
   - passes vs one pass (bit for bit);
   - the existing checks (split, EP2, precision) updated for the new layout.
3. **Driver:**
   - the module split;
   - job folder, fingerprint, observer records, code hash;
   - `run` (new / complete / partial / interrupted), `--rerun` with the backup question, `--fresh`, the attention message;
   - `status`, `clean`, `plan` (memory exact; time from a CPU calibration for now).
4. **Remote and machine analysis:**
   - `remote run / status / fetch`, tested against `localhost` before IAG;
   - `probe / calibrate / predict` for this machine or one reached by SSH.
5. **With the GPU engine:** the same passes on the device, the GPU memory budget from `cudaMemGetInfo`, GPU calibration, the probe.
6. **Later:** `video` (mp4 through ffmpeg), presets (512, 1k, 2k, 4k).

## 10. Photons per pixel, SNR and N: always both ways

**Choose N and get the SNR, or choose an SNR and get N.** The driver always shows both.

### Photons per pixel

A pixel's count is Poisson: if it expects λ photons, it gets λ ± √λ. So:

```
SNR = λ / √λ = √λ          100 photons per pixel → SNR 10;  400 → SNR 20
```

"Photons per pixel" (λ) and SNR are the same information; SNR is the one that reads as quality (SNR 10 = 10% noise).

### λ from N

For one frame of one observer, at a point of the disk where the surface is seen at μ_disk:

```
λ(μ_disk) = N · (1/nmu) · (1/nphi) · (4/size²) · (1 + a·μ_disk) / (π·(1 + 2a/3))
             │      │         │           │                 └ limb darkening, normalized over the disk
             │      │         │           └ area of one pixel (the sky plane spans [−1, 1]²)
             │      │         └ the observer's azimuth bin (one frame)
             │      └ the observer's μ bin (equal solid angles, so equal shares)
             └ photons
```

- **On average over the disk** it reduces to the planning formula, N / (nmu · nphi · (π/4) · size²).
- **The center** (μ_disk = 1) gets (1 + a)/(1 + 2a/3) of the average; **the limb** (μ_disk → 0) gets 1/(1 + 2a/3).
  - With a = 1: center 1.2×, limb 0.6× the average.
- **Spots remove a small fraction** (0.4% at EP2's defaults); ignored in planning.

### N from a target SNR

The quality target is set **at the limb**, the dimmest part of the disk:

```
N = SNR² · nmu · nphi · (π/4) · size² · (1 + 2a/3)
```

| Image (a = 1, frames 300, nmu 95) | SNR 10 at the limb | SNR 20 at the limb |
|---|---|---|
| 1k | 3.9·10¹² | 1.6·10¹³ |
| 2k | 1.6·10¹³ | 6.3·10¹³ |

### Measured, not only predicted

After a run, the driver measures the SNR from the image itself: mean counts in a ring near the center and near the limb, compared with the prediction. A check that the formula and the run agree.

## 11. Frames and μ bins: `nmu` follows from the frames

- **`nphi` (frames)** cuts the sky in **azimuth**: each frame collects the photons within Δφ = 2π/nphi. Equivalently, the star turns by Δφ during one frame.
- **`nmu`** cuts the sky in **latitude** (μ = cos i): each observer collects the photons within Δμ = 2/nmu of its inclination.
- **`nmu` is about angles, not about observers.** Observers are the μ bins we keep; `nmu` sets how wide each one is.

Both widths blur the image, because a frame is an average over all directions in its bin:

- **Azimuth bin:** a feature at the star's equator moves by Δφ · (size/2) pixels during one frame.
- **Latitude bin, near the equator** (where Δi ≈ Δμ): a feature moves by Δμ · (size/2) pixels between the edges of the bin.

**To blur equally in both directions,** make the two angles equal at the equator, Δμ = Δφ:

```
nmu = nphi / π, rounded to the nearest odd number      150 frames → 47;  300 → 95;  360 → 115
```

- **`nmu` is derived from the frames by default,** and can be overridden.
- **Why odd [found while implementing]:** with an even `nmu`, the equator (μ = 0) is the *edge* between two bins.
  - The equator-on observer (i = π/2) then lands in a bin centered at 1.5916 rad with nmu = 48, not at π/2. Typing 1.5708 (slightly above π/2, so μ slightly negative) picked the bin below.
  - With an odd `nmu`, bin (nmu − 1)/2 is centered exactly on μ = 0, and the equator-on observer is exactly equator-on.
  - The poles are always bin edges (μ = ±1), so a pole-on observer is never exactly pole-on: its bin center is at i ≈ √(2/nmu) (0.207 rad with nmu = 47). That is the price of binning in μ.
- **Blur in pixels:** ≈ π · size / nphi in both directions. With 1k and 300 frames, ~11 pixels.
  - For the video this is natural: it is the distance a spot moves between frames, like a film camera's motion blur.
  - For sharp single frames, more frames are needed (and more photons).
- **Near the poles,** a μ bin is wider in angle (Δi ≈ √(2Δμ) at the pole), so pole-on observers are blurrier in inclination. Their azimuth is meaningless anyway (the star turns in the image plane).
- **Fewer μ bins means fewer photons for the same SNR** (N ∝ nmu), so deriving nmu from the frames (47 instead of EP2's 150 for 150 frames) also saves photons.

## 12. Observability: knowing what a run is doing

**Now, the simplest:**

- **Progress lines from the engine.**
  - At most every 10 seconds it prints `PROGRESS PASS=2/4 CHUNK=48000/97657 PHOTONS_PER_SECOND=... ELAPSED=... ETA=...`.
  - The clock is checked once per chunk, never per photon: ~10⁵ clock reads for 10⁸ photons, a few milliseconds in total.
- **A log per job:** the driver copies everything the engine prints (progress, pass lines, summary) into `log.txt` in the job folder.
- **Status:** reads the job folder: observers done, the last progress line. The same locally and through `remote status`.

**Later:**

- **Profiling a run:** an option to wrap it with `nsys` (timeline) or profile one pass's kernel with `ncu` ([../cuda_class/nsight.md](../cuda_class/nsight.md)).
- **GPU metrics over time:** utilization, clock, temperature, power, sampled with `nvidia-smi` during the run and saved next to the log. Useful for the laptop's clock drops when it heats.

## 13. Machine analysis: what can a machine do?

Commands that point to **this machine or one reached by SSH**, and predict how a job will run there before running it.

| Step | What it does | Output |
|---|---|---|
| **probe** | queries the hardware: GPU name, SMs, memory total/free, FP32:FP64 ratio, clock; RAM, disk, CPU cores | `machines/<host>-<device>.json` |
| **calibrate** | short runs that measure speeds: photons/s per precision (and per SM count on the GPU), zeroing, copy and disk bandwidth (the serial costs that grow with the image) | added to the same profile |
| **predict** | for a job (or the presets) and a profile: footprint, K, passes, N ↔ SNR, time, disk; or the largest image the machine fills in a time budget | a table, machine by machine |

- **Profiles are files,** so predictions for the A30 can be made from the laptop once its profile has been fetched. No connection is needed to plan.
- **Every job records the prediction it was launched with,** and compares it with the measured times when it finishes. Prediction vs measurement, as in EP3, now for every job.
- **Gustafson, machine by machine:** "the largest image filled in 2 hours", or "passes for the 2k job", side by side for the laptop and the A30.
- **New hardware:** probe and calibrate once, then everything predicts. An unmeasured GPU can get a rough forecast scaled from datasheet ratios, to be replaced by its calibration.

## 14. Open details

- [x] Output names: μ bin and inclination of the bin center (`observer_mu0042_i1.5708.bin`).
- [x] Precision in the fingerprint.
- [x] Rerun asks whether to back up; unattended defaults to backing up.
- [x] Folder names with the main parameters plus the fingerprint.
- [x] Presets: 512, 1k, 2k, 4k.
- [x] N and SNR always both ways; the SNR target is set at the limb.
- [x] `nmu` derived from the frames (the odd number nearest nphi/π), overridable.
- [x] `--memory-budget` units: `7GB`, `7G`, `7e9`, `512MB` (decimal, case-insensitive).
- [x] Square images only (`--size`).
- [x] Observability: progress lines and a log per job now; Nsight and GPU metrics later.
- [x] The driver's commands: [driver.md](driver.md).
- [ ] The rotation length (sets the frames at 30 fps): still open.

## 15. Implemented: the engine with passes (CPU, 2026-11-21)

### What the engine does now

```
./star -n 1e8 --size 512 --frames 300 --observers 1.5708,1.0472,0.5236,0 \
       --memory-budget 2GB --output-dir runs/job --fingerprint 1a2b3c4d --code-hash 3f2a91c
```

| New in the engine | File |
|---|---|
| `--observers` as inclinations in radians; μ bin, bin-center inclination and file name per observer; two observers in one bin are an error | `observers.cu` |
| `--size` (square), `--frames`, `nmu` derived as the odd number nearest frames/π (section 11) | `options.cu` |
| `--observers-per-pass K` or `--memory-budget` (`7GB`, `7G`, `7e9`, `512MB`); not both | `options.cu` |
| Image layout `[observer][frame][iy][ix]`, allocated once for K observers, zeroed at the start of each pass | `image.cu`, `photon.cuh` |
| Per observer, at the end of its pass: the image, then its record, each written as `.partial` and renamed | `image.cu`, `report.cu`, `files.cuh` |
| Three kinds of output line: `PROGRESS` (at most every 10 s), `PASS` (one per pass), `SUMMARY` (the last line) | `report.cu` |
| The pass loop | `main.cu` |

- **The physics did not change.** `trace_photon` only reads `slot_of_imu` (the observer's place in this pass) instead of the old observer table.
- **`SUMMARY` does not depend on K:** `PHOTONS` and `BLOCKED` are the same in every pass, `RECORDED` adds up over the passes, and `MISSED` = photons − blocked − recorded. So 1 pass or 4 passes print the same counts; only the times differ.
- **New phase time:** `ZERO_TIME`, the zeroing of the image at each pass (it used to be inside `SETUP_TIME`). The serial part of a run is now `SETUP_TIME + ZERO_TIME + SAVE_TIME`.
- **`stdout` is line-buffered,** so a driver reading through a pipe sees each line as soon as it is printed.

### Checks (N = 10⁷ unless noted)

| Check | Result |
|---|---|
| 4 observers in 1 pass vs 4 passes of 1 | counts the same, **images identical bit for bit** |
| two halves of the chunk range vs the full run | identical bit for bit |
| EP4 vs EP2 without spots, N = 2·10⁸, EP2's observers (bin centers of its μ bins) | χ²/dof = 0.9957 over 126 496 pixels (−1.1σ from 1) |
| FP32 vs FP64, N = 10⁸ | 1.2·10⁻⁴ of the photons land in another pixel; FP32 1.47× faster |

### What we learned

- **What a pass costs: the full trace every time, while the image writes are shared out.** In the first run, one pass recording 4 observers (96 MB) ran at 9.1·10⁶ photons/s, and one observer per pass (24 MB) at 1.03·10⁷. I first said "the smaller image fits the caches better". A measurement that separates the causes says otherwise (N = 2·10⁷, photon phase, 3 repeats):

  | Run | Time (s) |
  |---|---|
  | no image, 4 observers | 1.76, 1.76, 1.88 |
  | no image, 1 observer | 1.75, 1.74, 1.94 |
  | image, 4 observers in 1 pass (96 MB) | 2.07, 2.04, 2.12 |
  | image, 1 observer (24 MB) | 1.84, 1.96, 1.89 |
  | image, 4 passes of 1 observer (total) | 7.33, 7.58, 8.25 |

  - **Computing the pixel** for 8% of the photons (4 observers) or 2% (1 observer) makes no measurable difference: the trace (surface point, direction, spots, bin) dominates.
  - **Writing to the image** costs ~0.3 s for 1.7·10⁶ recorded photons, ~180 ns each: about one access to main memory per write, since a random pixel is rarely in cache.
  - **A single-observer pass is cheaper because it writes 1/4 as many photons,** not (as far as this shows) because 24 MB fits the 30 MB L3 better. That effect, if there is one, is inside the ±0.1 s noise.
  - **So passes cost ≈ passes × trace time:** 4 passes ≈ 3.7× one pass here. The writes are split between the passes; the tracing is repeated.
  - **For the GPU:** the same split matters. The trace is pure arithmetic and parallelizes; the writes become atomics in L2 and device memory, and their cost is what Nsight Compute will show.
- **The FP32 effect fell from 5·10⁻⁴ to 1.2·10⁻⁴** of the photons. The physics did not change; the observers did. EP2's layout put an observer in the last μ bin of 150 (i ≈ 0.12 rad), close to the pole. The new pole-on observer sits at i ≈ 0.21 rad (nmu = 47). That fits the FP32 hotspot found at pole-on observers ([precision.md](precision.md)): the projection divides by sin i, which is small there. To confirm with a dedicated run.
- **The equator lies on a bin edge when `nmu` is even** (section 11). That is why `nmu` is now odd by default.
