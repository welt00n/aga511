import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lab import analysis, environment

FRAMES_PER_SECOND = 30
MP4_BYTES_PER_PIXEL_AND_FRAME = 0.15
GRID_WIDTH = 1024
LADDER_WIDTH = 1920
AROUND_RING = 1100
AROUND_LARGEST_VIEW = 360
AROUND_GAP = 1.45
MARGIN = 28
LABEL_HEIGHT = 34
BACKGROUND = (0, 0, 0)
INK = (214, 211, 204)
FAINT = (110, 108, 104)
STAR_FILL = (38, 37, 35)


# ============================================================ one observer's frames, read from disk as needed

def observer_frames(folder, record):
    path = os.path.join(folder, record["IMAGE"])
    return np.memmap(path, dtype=np.uint32, mode="r", shape=(record["FRAMES"], record["SIZE"], record["SIZE"]))


def brightest_level(frames):
    sample = np.asarray(frames[::max(1, len(frames) // 30)], dtype=np.float32)
    return float(np.percentile(sample, 99.9))


def views_of(folder, records):
    views = []
    for record in records:
        frames = observer_frames(folder, record)
        views.append({"record": record, "frames": frames, "brightest": brightest_level(frames),
                      "inclination": record["INCLINATION_BIN_CENTER"], "stem": record["IMAGE"].removesuffix(".bin")})
    return views


def picture(view, index, side=None):
    image = analysis.frame_to_picture(np.asarray(view["frames"][index], dtype=np.float32), view["brightest"])
    return image if side is None or side == image.width else image.resize((side, side), Image.BILINEAR)


def frame_count(views):
    return len(views[0]["frames"])


# ============================================================ writing an mp4, then repeating it

def even(value):
    value = int(math.ceil(value))
    return value + value % 2


def encode(pictures, width, height, path):
    command = [environment.tool("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(FRAMES_PER_SECOND), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path]
    video = subprocess.Popen(command, stdin=subprocess.PIPE)
    for image in pictures:
        video.stdin.write(image.tobytes())
    video.stdin.close()
    if video.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def repeat(one_cycle, path, loops):
    if loops == 1:
        os.replace(one_cycle, path)
        return
    command = [environment.tool("ffmpeg"), "-y", "-loglevel", "error", "-stream_loop", str(loops - 1),
               "-i", one_cycle, "-c", "copy", path]
    if subprocess.run(command).returncode != 0:
        raise RuntimeError(f"ffmpeg failed repeating {one_cycle}")
    os.remove(one_cycle)


def write_video(pictures, width, height, path, loops):
    one_cycle = path.removesuffix(".mp4") + ".one-cycle.mp4"
    encode(pictures, width, height, one_cycle)
    repeat(one_cycle, path, loops)
    return path


def media_bytes(width, height, frames, loops):
    return width * height * frames * loops * MP4_BYTES_PER_PIXEL_AND_FRAME


# ============================================================ labels

def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def inclination_label(radians):
    degrees = math.degrees(radians)
    side = (" (north pole-on)" if degrees < 10 else " (south pole-on)" if degrees > 170 else
            " (equator-on)" if abs(degrees - 90) < 1 else "")
    return f"i = {degrees:.0f}°{side}"


# ============================================================ one video per observer

def observer_video(view, output_folder, loops):
    picture(view, 0).save(os.path.join(output_folder, view["stem"] + "_frame_0.png"))
    side = view["record"]["SIZE"]
    pictures = (picture(view, index) for index in range(frame_count([view])))
    return write_video(pictures, even(side), even(side), os.path.join(output_folder, view["stem"] + ".mp4"), loops)


# ============================================================ all observers side by side

def grid_shape(count):
    columns = math.ceil(math.sqrt(count))
    return columns, math.ceil(count / columns)


def grid_side(views):
    columns, _ = grid_shape(len(views))
    return min(views[0]["record"]["SIZE"], GRID_WIDTH // columns)


def grid_size(views):
    columns, rows = grid_shape(len(views))
    side = grid_side(views)
    return even(columns * side), even(rows * side)


def grid_frame(views, index):
    columns, _ = grid_shape(len(views))
    side = grid_side(views)
    canvas = Image.new("RGB", grid_size(views), BACKGROUND)
    draw, label_font = ImageDraw.Draw(canvas), font(max(16, side // 20))
    for number, view in enumerate(views):
        x, y = (number % columns) * side, (number // columns) * side
        canvas.paste(picture(view, index, side), (x, y))
        draw.text((x + side // 40, y + side // 50), inclination_label(view["inclination"]), fill=INK, font=label_font)
    return canvas


def grid_video(views, output_folder, loops):
    grid_frame(views, 0).save(os.path.join(output_folder, "observers_grid_frame_0.png"))
    width, height = grid_size(views)
    pictures = (grid_frame(views, index) for index in range(frame_count(views)))
    return write_video(pictures, width, height, os.path.join(output_folder, "observers_grid.mp4"), loops)


# ============================================================ one observer at several N, side by side

def ladder_side(views):
    return even(min(views[0]["record"]["SIZE"], LADDER_WIDTH // len(views)))


def ladder_size(views):
    side = ladder_side(views)
    return even(side * len(views)), even(side + 2 * LABEL_HEIGHT)


def ladder_frame(views, index):
    side = ladder_side(views)
    canvas = Image.new("RGB", ladder_size(views), BACKGROUND)
    draw, label_font = ImageDraw.Draw(canvas), font(max(16, min(24, side // 14)))
    for number, view in enumerate(views):
        x = number * side
        canvas.paste(picture(view, index, side), (x, 2 * LABEL_HEIGHT))
        for line, text in enumerate(view["labels"]):
            draw.text((x + 10, 6 + line * (LABEL_HEIGHT - 6)), text, fill=INK, font=label_font)
    return canvas


def ladder_video(views, output_folder, loops, name="snr_ladder"):
    ladder_frame(views, 0).save(os.path.join(output_folder, name + "_frame_0.png"))
    width, height = ladder_size(views)
    pictures = (ladder_frame(views, index) for index in range(frame_count(views)))
    return write_video(pictures, width, height, os.path.join(output_folder, name + ".mp4"), loops)


# ============================================================ the views placed around the star, where they look from

def view_side(inclinations):
    angles = sorted(inclinations)
    gaps = [b - a for a, b in zip(angles, angles[1:])] or [math.pi]
    fits = AROUND_RING * 2 * math.sin(max(min(gaps), 1e-3) / 2) / AROUND_GAP
    return even(min(AROUND_LARGEST_VIEW, fits))


def direction(inclination):
    return math.sin(inclination), -math.cos(inclination)


def around_layout(views):
    inclinations = [view["inclination"] for view in views]
    side, star = view_side(inclinations), 0.12 * AROUND_RING
    centers = [(AROUND_RING * dx, AROUND_RING * dy) for dx, dy in map(direction, inclinations)]
    half = side / 2
    boxes = [(x - half, y - half, x + half, y + half + LABEL_HEIGHT) for x, y in centers]
    boxes.append((-1.7 * star, -1.7 * star, 1.7 * star, 1.7 * star))
    left, top = min(box[0] for box in boxes) - MARGIN, min(box[1] for box in boxes) - MARGIN
    right, bottom = max(box[2] for box in boxes) + MARGIN, max(box[3] for box in boxes) + MARGIN
    return {"origin": (-left, -top), "centers": centers, "side": side, "star": star,
            "size": (even(right - left), even(bottom - top))}


def around_background(views, layout):
    (ox, oy), star = layout["origin"], layout["star"]
    canvas = Image.new("RGB", layout["size"], BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.ellipse([ox - star, oy - star, ox + star, oy + star], fill=STAR_FILL, outline=INK, width=2)
    draw.ellipse([ox - star, oy - 0.28 * star, ox + star, oy + 0.28 * star], outline=FAINT, width=1)
    draw.line([(ox, oy - 1.6 * star), (ox, oy + 1.6 * star)], fill=FAINT, width=2)
    draw.text((ox + 8, oy - 1.6 * star - 4), "rotation axis", fill=FAINT, font=font(20))
    side = layout["side"]
    for view, (cx, cy) in zip(views, layout["centers"]):
        dx, dy = direction(view["inclination"])
        start = (ox + 1.05 * star * dx, oy + 1.05 * star * dy)
        end = (ox + cx - 0.55 * side * dx, oy + cy - 0.55 * side * dy)
        draw.line([start, end], fill=FAINT, width=1)
        draw.text((ox + cx - side / 2, oy + cy + side / 2 + 6), inclination_label(view["inclination"]),
                  fill=INK, font=font(20))
    return canvas


def around_frame(views, layout, background, index):
    canvas = background.copy()
    (ox, oy), side = layout["origin"], layout["side"]
    for view, (cx, cy) in zip(views, layout["centers"]):
        canvas.paste(picture(view, index, side), (int(ox + cx - side / 2), int(oy + cy - side / 2)))
    return canvas


def around_video(views, output_folder, loops):
    layout = around_layout(views)
    background = around_background(views, layout)
    around_frame(views, layout, background, 0).save(os.path.join(output_folder, "observers_around_frame_0.png"))
    pictures = (around_frame(views, layout, background, index) for index in range(frame_count(views)))
    width, height = layout["size"]
    return write_video(pictures, width, height, os.path.join(output_folder, "observers_around.mp4"), loops)
