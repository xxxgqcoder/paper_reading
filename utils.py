import time
import os
import shutil


def time_it(func):

    def wrapper(*kargs, **kwargs):
        begin = time.time_ns()
        ret = func(*kargs, **kwargs)
        elapse = (time.time_ns() - begin) // 1000000
        print(
            f"func {func.__name__} took {elapse // 60000}min {(elapse % 60000)//1000}sec {elapse%60000%1000}ms to finish"
        )

        return ret

    return wrapper


def save_image(src_path: str, dst_dir: str) -> None:
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copyfile(src_path, dst_path)


def safe_strip(raw: str) -> str:
    if raw is None or len(raw) == 0:
        return ''
    raw = str(raw)
    return raw.strip()
