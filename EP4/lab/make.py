import math
import os

from lab import engine, environment, plan, runs, snr, star, videos
from lab.gpu_monitor import GpuMonitor

NOISY_SNR = 3
STAR_OPTIONS = ("seed", "limb", "spots", "observers", "size", "frames", "nmu", "chunk_size", "observers_per_pass")


def photons_and_snr(config, processor, precision, wanted_snr):
    nmu = engine.describe_engine(dict(config, photons=1), processor, precision)[0]["NMU"]
    if wanted_snr:
        return snr.photons_for_snr(wanted_snr, config["size"], config["frames"], nmu, config["limb"]), wanted_snr, nmu
    limb_mu = math.sqrt(1 - sum(snr.LIMB_RING) ** 2 / 4)
    counts = snr.expected_counts(config["photons"], config["size"], config["frames"], nmu, config["limb"], limb_mu)
    return config["photons"], math.sqrt(counts), nmu


# ============================================================ what the videos will be

def planned_views(config, processor, precision):
    observers = engine.describe_engine(dict(config, photons=1), processor, precision)[1]
    return [{"inclination": observer["INCLINATION_BIN_CENTER"], "record": {"SIZE": config["size"]}}
            for observer in observers]


def video_sizes(config, processor, precision):
    views = planned_views(config, processor, precision)
    sizes = {"each observer": (config["size"], config["size"])}
    sizes["observers_grid.mp4"] = videos.grid_size(views)
    sizes["observers_around.mp4"] = videos.around_layout(views)["size"]
    return sizes


def media_bytes(config, processor, precision, loops):
    sizes = video_sizes(config, processor, precision)
    each = len(config["observers"]) * videos.media_bytes(*sizes.pop("each observer"), config["frames"], loops)
    return each + sum(videos.media_bytes(width, height, config["frames"], loops) for width, height in sizes.values())


def show_prediction(config, processor, precision, expected_snr, nmu, each, loops):
    prediction = plan.predict(each)
    seconds = config["frames"] / videos.FRAMES_PER_SECOND
    print(f"\nThe pictures: {star.describe_star(config)}, nmu {nmu}, on {processor}{precision}")
    print(f"  N = {config['photons']:.3e} → SNR at the limb ring ≈ {expected_snr:.1f}"
          + ("  (noisy: raise --snr or -n)" if expected_snr < NOISY_SNR else ""))
    print(f"  predicted: {plan.duration(prediction['seconds'])} ({prediction['speed']:.3g} photons/s, "
          f"{prediction['speed_source']}); {prediction['passes']} pass(es), "
          f"{prediction['device_bytes'] / 1e9:.2f} GB on the device")
    print(f"  videos: one rotation = {config['frames']} frames = {seconds:g} s at {videos.FRAMES_PER_SECOND} fps, "
          f"repeated {loops}× = {seconds * loops:g} s each; one per observer, then all of them side by side, "
          "then around the star")
    for name, (width, height) in video_sizes(config, processor, precision).items():
        print(f"    {name}: {width}×{height}")
    print(f"  disk: {prediction['disk_bytes'] / 1e9:.2f} GB of raw images while it runs (deleted after), "
          f"~{media_bytes(config, processor, precision, loops) / 1e6:.0f} MB of mp4 kept")
    print("  pictures aren't measurements: no minimum N applies")


# ============================================================ making them

def observer_videos(views, observers_folder, config, nmu, output_folder, loops):
    observers = []
    for view in views:
        media = videos.observer_video(view, output_folder, loops)
        observers.append({"image": view["record"]["IMAGE"], "media": os.path.basename(media),
                          "snr": snr.snr_check(observers_folder, view["record"], config["photons"], nmu,
                                               config["limb"])})
        print(f"  {os.path.basename(media)}", flush=True)
    return observers


def combined_videos(views, output_folder, loops):
    made = [videos.grid_video(views, output_folder, loops), videos.around_video(views, output_folder, loops)]
    for path in made:
        print(f"  {os.path.basename(path)}", flush=True)
    return [os.path.basename(path) for path in made]


# ============================================================ the SNR ladder: the same star at several N

LADDER_INCLINATION = 1.0472


def ladder_photons(text):
    if ":" in text:
        low, high = (round(math.log10(float(value))) for value in text.split(":"))
        return [10 ** exponent for exponent in range(low, high + 1)]
    return sorted(int(float(value)) for value in text.split(","))


def ladder_label(photons):
    return f"n{photons:.0e}".replace("+", "")


def closest_view(views, inclination):
    return min(views, key=lambda view: abs(view["inclination"] - inclination))


def show_ladder(config, processor, precision, rungs, loops):
    total, peak = 0, 0
    print(f"\nThe SNR ladder: {star.describe_star(dict(config, photons=rungs[-1]))}, on {processor}{precision}")
    for photons in rungs:
        _, expected_snr, _ = photons_and_snr(dict(config, photons=photons), processor, precision, None)
        star_options = {name: config[name] for name in STAR_OPTIONS}
        prediction = plan.predict(plan.point(processor, precision, photons, config["sms"], **star_options))
        total += prediction["seconds"]
        peak = max(peak, prediction["disk_bytes"])
        print(f"  N = {photons:.0e}: SNR at the limb ≈ {expected_snr:.2f}, {plan.duration(prediction['seconds'])}, "
              f"{prediction['passes']} pass(es)")
    views = planned_views(config, processor, precision)
    around = videos.around_layout(views)["size"]
    ladder = videos.ladder_size([dict(views[0], labels=[]) for _ in rungs])
    media = len(rungs) * videos.media_bytes(*around, config["frames"], loops) + \
        videos.media_bytes(*ladder, config["frames"], loops)
    print(f"  total: {plan.duration(total)} of runs, then the videos; at most "
          f"{(peak + len(rungs) * peak / len(views)) / 1e9:.1f} GB of raw images on disk (deleted after)")
    print(f"  videos: per N the around video ({around[0]}×{around[1]}); the ladder ({ladder[0]}×{ladder[1]}), "
          f"the observer nearest i = {math.degrees(LADDER_INCLINATION):.0f}° at every N; ~{media / 1e6:.0f} MB, "
          f"each repeated {loops}×")


def ladder_rung(config, photons, processor, precision, nmu, run, loops):
    folder = run.path(ladder_label(photons))
    os.makedirs(folder, exist_ok=True)
    rung = dict(config, photons=photons)
    summary = engine.run_engine(rung, os.path.join(folder, "observers"), processor, precision)
    run.add_machine(summary)
    views = videos.views_of(os.path.join(folder, "observers"), engine.observer_records(os.path.join(folder, "observers")))
    kept = closest_view(views, LADDER_INCLINATION)
    snr_rows = snr.snr_check(os.path.join(folder, "observers"), kept["record"], photons, nmu, config["limb"])
    videos.around_video(views, folder, loops)
    for view in views:
        if view is not kept:
            os.remove(os.path.join(folder, "observers", view["record"]["IMAGE"]))
    limb = snr_rows[1]
    kept["labels"] = [f"N = {photons:.0e}", f"SNR limb {limb['snr_measured']:.2f} (predicted {limb['snr_predicted']:.2f})"]
    print(f"  N = {photons:.0e}: {summary['PHOTONS_TIME']:.3g} s, SNR at the limb {limb['snr_measured']:.2f}", flush=True)
    return {"photons": photons, "folder": os.path.basename(folder), "summary": summary, "snr": snr_rows,
            "observer": kept["record"]["IMAGE"]}, kept


def ladder_lines(config, rungs_done):
    lines = [star.describe_star(config), "", "The same star and seed at every N: each smaller run is exactly the first "
             "chunks of the larger ones, so the rungs are nested like partial sums.", "",
             "| N | photons (s) | SNR limb: measured / predicted | SNR center: measured / predicted | around video |",
             "|---|---|---|---|---|"]
    for rung in rungs_done:
        center, limb = rung["snr"]
        lines.append(f"| {rung['photons']:.0e} | {rung['summary']['PHOTONS_TIME']:.3g} | "
                     f"{limb['snr_measured']:.2f} / {limb['snr_predicted']:.2f} | "
                     f"{center['snr_measured']:.2f} / {center['snr_predicted']:.2f} | "
                     f"[{rung['folder']}/observers_around.mp4]({rung['folder']}/observers_around.mp4) |")
    return lines + ["", "The ladder: [snr_ladder.mp4](snr_ladder.mp4)"]


def ladder(config, processor, precision, rungs, loops=5, yes=False, plan_only=False):
    found = environment.checks_for(needs_gpu=processor == "gpu", needs_video=True)
    if not environment.show(found):
        return False
    environment.show_launch()
    try:
        nmu = photons_and_snr(dict(config, photons=rungs[0]), processor, precision, None)[2]
    except RuntimeError as error:
        print("\nThe engine refused this star: " + str(error).strip().splitlines()[-1])
        return False
    show_ladder(config, processor, precision, rungs, loops)
    if plan_only or not plan.confirmed(yes):
        return True
    label = f"ladder-{config['size']}-f{config['frames']}-{processor}{precision}-{ladder_label(rungs[0])}-{ladder_label(rungs[-1])}"
    asked = dict(config, processor=processor, precision=precision, ladder=rungs, loops=loops,
                 ladder_inclination=LADDER_INCLINATION)
    with runs.Run("make", label, asked) as run:
        rungs_done, kept = [], []
        for photons in rungs:
            rung, view = ladder_rung(config, photons, processor, precision, nmu, run, loops)
            rungs_done.append(rung)
            kept.append(view)
        videos.ladder_video(kept, run.folder, loops)
        for rung, view in zip(rungs_done, kept):
            os.remove(os.path.join(run.path(rung["folder"], "observers"), view["record"]["IMAGE"]))
        runs.save_json({"rungs": rungs_done, "loops": loops}, run.path("ladder.json"))
        top = rungs_done[-1]["snr"][1]
        headline = (f"{len(rungs)} rungs, N = {rungs[0]:.0e} … {rungs[-1]:.0e}: SNR at the limb "
                    f"{rungs_done[0]['snr'][1]['snr_measured']:.2f} … {top['snr_measured']:.1f}")
        stills = ["snr_ladder_frame_0.png"] + [f"{rung['folder']}/observers_around_frame_0.png" for rung in rungs_done]
        run.finish(headline, ladder_lines(config, rungs_done), stills)
    return True


def make(config, processor, precision, wanted_snr=None, loops=5, yes=False, plan_only=False):
    found = environment.checks_for(needs_gpu=processor == "gpu", needs_video=True)
    if not environment.show(found):
        return False
    environment.show_launch()
    try:
        photons, expected_snr, nmu = photons_and_snr(config, processor, precision, wanted_snr)
    except RuntimeError as error:
        print("\nThe engine refused this star: " + str(error).strip().splitlines()[-1])
        return False
    config = dict(config, photons=photons)
    star_options = {name: config[name] for name in STAR_OPTIONS}
    show_prediction(config, processor, precision, expected_snr, nmu,
                    plan.point(processor, precision, photons, config["sms"], **star_options), loops)
    if plan_only or not plan.confirmed(yes):
        return True
    label = (f"{config['size']}-f{config['frames']}-{processor}{precision}"
             + (f"-snr{wanted_snr:g}" if wanted_snr else ""))
    asked = dict(config, processor=processor, precision=precision, snr_at_the_limb=wanted_snr, loops=loops)
    with runs.Run("make", label, asked) as run:
        observers_folder = run.path("observers")
        with GpuMonitor(interval_ms=1000) as monitor:
            summary = engine.run_engine(config, observers_folder, processor, precision)
        run.add_machine(summary)
        views = videos.views_of(observers_folder, engine.observer_records(observers_folder))
        observers = observer_videos(views, observers_folder, config, nmu, run.folder, loops)
        combined = combined_videos(views, run.folder, loops)
        engine.delete_observer_images(observers_folder)
        runs.save_json({"summary": summary, "gpu_monitor": monitor.summary(), "observers": observers,
                        "combined_videos": combined, "loops": loops}, run.path("make.json"))
        stills = ["observers_around_frame_0.png", "observers_grid_frame_0.png"] + \
                 [observer["image"].removesuffix(".bin") + "_frame_0.png" for observer in observers]
        lines = [star.describe_star(config), "",
                 f"Videos: one rotation ({config['frames']} frames at {videos.FRAMES_PER_SECOND} fps), repeated "
                 f"{loops}×. All observers: [observers_around.mp4](observers_around.mp4), "
                 "[observers_grid.mp4](observers_grid.mp4).", "",
                 *star.phases_table([(f"{processor}{precision}", summary)]), "", *snr.snr_table(observers)]
        run.finish(snr.snr_headline(summary, observers), lines, stills)
    return True
