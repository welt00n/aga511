# ============================================================ describing a run

def describe_star(config):
    nmu = config["nmu"] if config["nmu"] is not None else "odd nearest frames/π"
    observers = ", ".join(f"{i:g}" for i in config["observers"])
    return (f"N = {config['photons']:.3g}, {config['size']}² pixels, {config['frames']} frames, nmu = {nmu}, "
            f"observers at {observers} rad, {config['spots']} spots")


def phases_header():
    return ["| run | photons/s | photons (s) | setup (s) | zero (s) | copy (s) | save (s) | passes | recorded | "
            "blocked |", "|---|---|---|---|---|---|---|---|---|---|"]


def phases_row(name, result):
    recorded = f"{result['RECORDED']} ({100 * result['RECORDED'] / result['PHOTONS']:.2f}%)"
    return (f"| {name} | {result['PHOTONS_PER_SECOND']:.3g} | {result['PHOTONS_TIME']:.3f} | "
            f"{result['SETUP_TIME']:.3f} | {result['ZERO_TIME']:.3f} | {result['COPY_TIME']:.3f} | "
            f"{result['SAVE_TIME']:.3f} | {result['PASSES']} | {recorded} | {result['BLOCKED']} |")


def phases_table(named_results):
    return phases_header() + [phases_row(name, result) for name, result in named_results]
