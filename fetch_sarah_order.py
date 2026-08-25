"""Fetch a completed CM SAF order, given the order number from its email.

SARAH is the independent validation reference of Section 3.3.3 - the fix for the
validation-independence gap, since every skill figure currently rests on ERA5
alone. It is not acquired by a plain GET. An order is placed on the CM SAF Web
User Interface with a spatial subset applied at extraction time (without it the
product is the whole Meteosat disk, 65S-65N by 65W-65E, roughly three hundred
times this domain), CM SAF extracts it, and an email supplies a per-order URL
and a shared read-only password that expires after seven days.

The password is therefore NOT stored here. Pass it in the environment:

    CMSAF_PASSWORD=... python fetch_sarah_order.py ORD68669 ORD68670

The record arrives in two halves that must be ordered with an identical
extraction spec, or they do not join cleanly at the boundary:

    CDR   1985-2020   the climate data record
    ICDR  2021-      the interim extension, 5-day latency

Every tar carries an MD5 in its email; pass it with --md5 to have it checked,
because a truncated archive on a slow link extracts far enough to look fine.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(ROOT, "data/raw/sarah")
BASE = "https://cmsaf.dwd.de/data"
USER = os.environ.get("CMSAF_USER", "routcm")


def md5sum(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(order, password, expect=None):
    os.makedirs(DEST, exist_ok=True)
    tar = os.path.join(DEST, "%s.tar" % order)
    url = "%s/%s/%s.tar" % (BASE, order, order)

    # curl rather than wget: wget is not present on macOS by default, and -C -
    # resumes rather than restarting a partial transfer.
    cmd = ["curl", "-sS", "-C", "-", "--retry", "3", "--retry-delay", "5",
           "-u", "%s:%s" % (USER, password), "-o", tar, url]
    print("fetching %s ..." % order)
    rc = subprocess.call(cmd)
    if rc != 0 or not os.path.exists(tar):
        print("  ! curl failed (exit %d). The order may have expired - they are"
              " kept for 7 days." % rc)
        return False

    got = md5sum(tar)
    if expect and got != expect:
        print("  ! MD5 mismatch: got %s, expected %s" % (got, expect))
        print("    Re-run to resume; delete the file first to start over.")
        return False
    print("  %.1f MB, md5 %s%s" % (os.path.getsize(tar) / 1048576.0, got,
                                   " (matches)" if expect else ""))

    with tarfile.open(tar) as t:
        members = [m for m in t.getmembers() if m.isfile()]
        t.extractall(DEST, filter="data")
    print("  extracted %d files into %s" % (len(members), DEST))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orders", nargs="+", help="order numbers, e.g. ORD68669")
    ap.add_argument("--md5", nargs="*", default=[],
                    help="expected MD5s, in the same order as the order numbers")
    args = ap.parse_args()

    password = os.environ.get("CMSAF_PASSWORD")
    if not password:
        sys.exit("Set CMSAF_PASSWORD to the password in the CM SAF order email.")

    ok = True
    for i, order in enumerate(args.orders):
        expect = args.md5[i] if i < len(args.md5) else None
        ok &= fetch(order, password, expect)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
