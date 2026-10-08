"""Keep Appendix A's description of the repository true.

Appendix A told the reader the repository contains "76 top-level Python scripts".
It had 114 by the eighth round and 82 after the consumed one-off scripts were
removed, so the figure has been wrong in both directions. The count is read from
`git ls-files` at run time rather than written as a literal, which is the same
discipline applied to every other number in the document: the sentence cannot
drift from the thing it describes.

The appendix also now says what the repository deliberately does not contain,
because a reader who finds no script that assembles the dissertation should be
told that is a decision rather than an omission.

    python fix_appendix_a_script_count.py
"""

import os
import re
import shutil
import subprocess
import sys

import docx

ROOT = os.path.dirname(os.path.abspath(__file__))
CH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS")
DOC = os.path.join(CH, "CR_Madukwe_Dissertation.docx")
LOCK = os.path.join(CH, "~$_Madukwe_Dissertation.docx")


def script_count():
    """Tracked, top-level Python scripts. Falls back to the filesystem."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
        n = len([f for f in out if "/" not in f and f.endswith(".py")])
        if n:
            return n
    except Exception:
        pass
    return len([f for f in os.listdir(ROOT) if f.endswith(".py")])


N = script_count()
# Two paragraphs describe the repository: Section 3.11 and Appendix A, written
# independently and so drifting independently. 3.11 said 76 scripts and "21
# invariant checks" while Appendix A said 76 and the suite had 28 tests. Both
# are matched here, and the test count is removed rather than corrected: it
# changes whenever a test is added, which is exactly the drift the brief guard
# already forbids quoting.
SENT = re.compile(
    r"The repository (contains|holds) (\d+|[a-z]+) top-level Python scripts")
TESTS = re.compile(r"a test suite of \d+ invariant checks")
NEW_TAIL = (
    " It contains only what produces the result: scripts that patched the "
    "document once and were consumed, and the orchestration written for "
    "particular reruns, were removed once spent and remain in the version "
    "history. There is deliberately no script that assembles the dissertation "
    "from scratch, because the merged document carries citation fields that "
    "exist only in it and regenerating the file would destroy them; the results "
    "chapters are replaced in place instead, and the field count is verified "
    "before and after.")


def main():
    if os.path.exists(LOCK):
        print("The dissertation is open in Word. Close it first.")
        return 2
    d = docx.Document(DOC)
    before = (d.element.xml.count("ZOTERO_ITEM"),
              d.element.xml.count('w:fldCharType="begin"'), len(d.sections))
    if not any(SENT.search(q.text) for q in d.paragraphs):
        print("NOT FOUND: the script-count sentence")
        return 1
    done = 0
    for par in [q for q in d.paragraphs if SENT.search(q.text)]:
        for r in par.runs:
            m = SENT.search(r.text)
            if not m:
                continue
            if m.group(2) != str(N):
                r.text = SENT.sub(
                    lambda mm: "The repository %s %d top-level Python scripts"
                    % (mm.group(1), N), r.text, count=1)
                done += 1
                print("  script count -> %d" % N)
            else:
                print("  script count already %d" % N)
            if TESTS.search(r.text):
                r.text = TESTS.sub("a test suite of invariant checks", r.text)
                done += 1
                print("  removed the quoted test count")
            break
        else:
            print("  a sentence spans runs; not edited")
            return 1
    appendix_a = next((q for q in d.paragraphs
                       if SENT.search(q.text) and "contains" in q.text), None)
    if appendix_a is not None and NEW_TAIL.strip()[:60] not in appendix_a.text:
        appendix_a.runs[-1].text = appendix_a.runs[-1].text.rstrip() + NEW_TAIL
        done += 1
        print("  added what the repository deliberately omits")
    if not done:
        print("nothing to change")
        return 0
    bak = DOC.replace(".docx", "_BACKUP_pre_appendixa.docx")
    shutil.copy(DOC, bak)
    d.save(DOC)
    dd = docx.Document(DOC)
    after = (dd.element.xml.count("ZOTERO_ITEM"),
             dd.element.xml.count('w:fldCharType="begin"'), len(dd.sections))
    if after != before:
        shutil.copy(bak, DOC)
        print("STRUCTURE CHANGED %s -> %s, restored" % (before, after))
        return 1
    print("%d changed; citations/fields/sections intact %s" % (done, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
