import argparse
import shlex
import sys

from lab import comparison, engine, make, measurement, parts, prompts, runs, testing


# ============================================================ option types

def photons(text):
    value = float(text)
    if value < 1 or value != int(value):
        raise argparse.ArgumentTypeError("the number of photons must be a whole number >= 1")
    return int(value)


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def bounded_size(low, high, why):
    def size(text):
        value = int(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"the size must be {low}–{high} pixels: {why}")
        return value
    return size


def inclinations(text):
    return [float(inclination) for inclination in text.split(",")]


def part_names(text):
    names = list(parts.PARTS) if text == "all" else text.split(",")
    unknown = [name for name in names if name not in parts.PARTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown part(s) {', '.join(unknown)}; parts: all, {', '.join(parts.PARTS)}")
    return names


def targets(text):
    return [target for target in text.split(",") if target]


def option(*flags, advanced=False, **settings):
    return {"flags": flags, "advanced": advanced, "settings": settings}


# ============================================================ the options

YES = option("--yes", action="store_true", advanced=True, help="don't ask before running")
PLAN = option("--plan", action="store_true", help="only check and predict: run nothing")
BOTH_HEMISPHERES = [0.0, 0.5236, 1.0472, 1.5708, 2.0944, 2.618, 3.1416]
STAR = [
    option("--size", type=bounded_size(16, 2048, "2k is the largest image (on the A30; ~3 h on the laptop)"),
           default=1024, help="square image side in pixels, up to 2048"),
    option("--frames", type=int, default=300, help="frames of the rotation"),
    option("--observers", type=inclinations, default=BOTH_HEMISPHERES,
           help="inclinations in radians from the north pole (0) to the south pole (3.1416)"),
    option("--sms", type=int, default=None, advanced=True, help="the GPU: work on this many SMs only"),
    option("--observers-per-pass", type=int, default=None, advanced=True,
           help="observers recorded per pass (default: as many as fit)"),
    option("--nmu", type=int, default=None, advanced=True, help="mu bins (default: the odd number nearest frames/pi)"),
    option("--seed", type=int, default=42, advanced=True),
    option("--limb", type=float, default=1.0, advanced=True, help="limb darkening a in I = I0 (1 + a mu)"),
    option("--spots", type=int, default=3, advanced=True),
    option("--chunk-size", type=int, default=1024, advanced=True, help="photons per random stream"),
]
MAKE = [
    option("--snr", type=float, default=10.0, help="the signal-to-noise ratio wanted at the limb (sets N)"),
    option("-n", "--photons", type=photons, default=None, advanced=True, help="N directly, instead of --snr"),
    option("--processor", choices=engine.PROCESSORS, default="gpu", help="where it runs"),
    option("--precision", type=int, choices=engine.PRECISIONS, default=32, help="bits of the floating point"),
    option("--loops", type=positive, default=5, help="times each video repeats the rotation"),
    option("--ladder", type=make.ladder_photons, default=None,
           help="an SNR ladder instead: N from:to one decade at a time (e.g. 1e9:1e12), or a list (1e9,1e10)"),
    *STAR, PLAN, YES,
]
MEASURE = [
    option("--parts", type=part_names, default=list(parts.PARTS), help="all, or some of " + ",".join(parts.PARTS)),
    option("--seconds", type=float, default=5.0, help="seconds of photons per run on all SMs (sets the N)"),
    option("--repeats", type=int, default=3, help="repeats of each run"),
    option("--size", type=bounded_size(1024, 2048, "measurements are made on high-resolution images"),
           default=1024, help="the measured star's image side in pixels, 1024 to 2048"),
    option("--frames", type=int, default=300, help="frames of the measured star's rotation"),
    option("--targets", type=targets, default=["80"], advanced=True,
           help="other GPUs to compile for (registers and occupancy only), e.g. 80 for the A30"),
    option("--continue", dest="continue_folder", default=None, advanced=True,
           help="a measurement's folder: run what is missing, then analyse"),
    option("--allow-invalid", action="store_true", advanced=True,
           help="run even below the minimum N (the report says it is not valid)"),
    PLAN, YES,
]
COMPARE = [option("folders", nargs="+", help="two or more measurement folders (runs/measure/...)")]
ARCH = option("--arch", default="native", help="native, or a compute capability such as 80 (the A30)")


# ============================================================ the commands

def star_config(args):
    return engine.default_star_config(args.photons or 1, seed=args.seed, limb=args.limb, spots=args.spots,
                                      observers=args.observers, size=args.size, frames=args.frames, nmu=args.nmu,
                                      chunk_size=args.chunk_size, observers_per_pass=args.observers_per_pass,
                                      sms=args.sms)


def run_test(args):
    return testing.test(args.yes, args.plan)


def run_make(args):
    if args.ladder:
        return make.ladder(star_config(args), args.processor, args.precision, args.ladder, args.loops, args.yes,
                           args.plan)
    wanted_snr = None if args.photons else args.snr
    return make.make(star_config(args), args.processor, args.precision, wanted_snr, args.loops, args.yes, args.plan)


def run_measure(args):
    return measurement.measure(args.parts, args.seconds, args.repeats, args.targets, args.yes, args.plan,
                               args.allow_invalid, args.continue_folder, args.size, args.frames)


def run_compare(args):
    return comparison.compare(args.folders)


def list_runs(args):
    print("\n".join(runs.write_index()))


def build(args):
    for binary in engine.build_all(args.arch):
        print(binary)


def command(description, options, handler):
    return {"description": description, "options": options, "handler": handler}


COMMANDS = {
    "test": command("does it work here? the environment, the builds and the correctness checks (~1 min)",
                    [PLAN, YES], run_test),
    "make": command("pictures and videos of the star, N set by the SNR wanted", MAKE, run_make),
    "measure": command("every measurement on this machine at valid N, with its report (notebook + PDF)", MEASURE,
                       run_measure),
    "compare": command("two or more measurements side by side (e.g. the laptop and the A30)", COMPARE, run_compare),
    "runs": command("every run so far, newest first", [], list_runs),
    "build": command("build every engine", [ARCH], build),
}


# ============================================================ the command line and the menu

def build_parser():
    parser = argparse.ArgumentParser(prog="main.py", description="EP4: the spotted star on the CPU and the GPU. "
                                                                 "Without a command, a menu asks.")
    commands = parser.add_subparsers(dest="command", metavar="command")
    for name, details in COMMANDS.items():
        sub = commands.add_parser(name, help=details["description"], description=details["description"])
        for each in details["options"]:
            sub.add_argument(*each["flags"], **each["settings"])
        sub.set_defaults(handler=details["handler"])
    return parser


def shown(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(f"{item:g}" if isinstance(item, float) else str(item) for item in value)
    return str(value)


def answer_option(each):
    flags, settings = each["flags"], each["settings"]
    question = flags[-1].lstrip("-") + (f": {settings['help']}" if "help" in settings else "")
    default = shown(settings.get("default"))
    if settings.get("action") == "store_true":
        return [flags[-1]] if prompts.ask_yes(question) else []
    if "choices" in settings:
        answer = prompts.ask_choice(question, [(str(choice), "") for choice in settings["choices"]], default)
    else:
        answer = prompts.ask_value(question, default, settings.get("type", str))
    if answer == default:
        return []
    values = answer.split() if settings.get("nargs") == "+" else [answer]
    return values if not flags[0].startswith("-") else [flags[-1], *values]


def menu():
    name = prompts.ask_choice("What do you want to do?",
                              [(name, details["description"]) for name, details in COMMANDS.items()], "test")
    options = COMMANDS[name]["options"]
    advanced = any(each["advanced"] for each in options) and prompts.ask_yes("Change the advanced options too?")
    argv = [name]
    for each in options:
        if advanced or not each["advanced"]:
            argv += answer_option(each)
    print(f"\n→ python main.py {shlex.join(argv)}\n")
    return argv


def main():
    parser = build_parser()
    try:
        argv = sys.argv[1:] or menu()
        runs.remember_command(argv)
        args = parser.parse_args(argv)
        if not getattr(args, "handler", None):
            parser.print_help()
            return 2
        return 1 if args.handler(args) is False else 0
    except KeyboardInterrupt:
        print("\nstopped")
        return 130


if __name__ == "__main__":
    sys.exit(main())
