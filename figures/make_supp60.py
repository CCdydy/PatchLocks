#!/usr/bin/env python3
"""Make the 60%-scale copies the supplement includes (PatchLocks_supp.tex uses
\\graphicspath{{Figures/_supp60/}{Figures/}}, so a file under _supp60/ shadows the
full-resolution one).

    python figures/make_supp60.py SRC.png DST.png

Recipe used for the paper: LANCZOS resize to int(0.6*w) x int(0.6*h), PNG optimize=True.
"""
import os
import sys

from PIL import Image


def main():
    src, dst = sys.argv[1], sys.argv[2]
    im = Image.open(src)
    out = im.resize((int(im.width * 0.6), int(im.height * 0.6)), Image.LANCZOS)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    out.save(dst, optimize=True)
    print("supp60", src, im.size, "->", dst, out.size)


if __name__ == "__main__":
    main()
