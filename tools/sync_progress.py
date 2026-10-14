#!/usr/bin/env python3
"""Rebuild progress.json from the notes files.

A day counts as done when the note file for it has words in it that are not
part of the blank template. That is the only completion signal that is visible
outside one browser, so it is the one the page trusts.

Run from the root of the gh-pages checkout:
    python3 tools/sync_progress.py
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(ROOT, "notes")

PROMPTS = [
    "Clarity, 1 to 5:",
    "What I took away:",
    "What lost me:",
    "Question for tomorrow:",
]


def template(day, title):
    lines = ["# Day %d notes: %s" % (day, title), ""]
    lines.append("Two minutes is plenty. Commit it and the site ticks over.")
    lines.append("")
    for p in PROMPTS:
        lines.append("**%s**" % p)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def boilerplate_lines():
    """Every line that can appear in a blank note, lowercased and stripped."""
    out = {"", "two minutes is plenty. commit it and the site ticks over."}
    for p in PROMPTS:
        out.add(("**%s**" % p).lower())
    return out


def answered(text):
    """True when the file carries words beyond the blank template."""
    boiler = boilerplate_lines()
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if line.lower() in boiler:
            continue
        # a prompt with an answer typed on the same line
        for p in PROMPTS:
            if line.lower().startswith(("**%s**" % p).lower()):
                line = line[len("**%s**" % p):].strip()
                break
        line = re.sub(r"[*_>\-\s]", "", line)
        if len(line) >= 2:
            return True
    return False


def first_commit_date(path):
    """When this note was first written, by git. Falls back to None."""
    rel = os.path.relpath(path, ROOT)
    try:
        out = subprocess.run(
            ["git", "log", "--follow", "--diff-filter=AM", "--format=%aI", "--", rel],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
    except Exception:
        return None
    return out[-1] if out else None


def main():
    idx_path = os.path.join(ROOT, "lessons", "index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    titles = {l["day"]: l.get("title", "") for l in idx.get("lessons", [])}

    os.makedirs(NOTES, exist_ok=True)
    verified = {}
    created = []

    for day in sorted(titles):
        path = os.path.join(NOTES, "day-%03d.md" % day)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(template(day, titles[day]))
            created.append(day)
            continue
        with open(path) as f:
            text = f.read()
        if answered(text):
            words = len([w for w in re.findall(r"[A-Za-z0-9']+", text)])
            m = re.search(r"\*\*Clarity, 1 to 5:\*\*\s*([1-5])", text)
            entry = {"words": words}
            if m:
                entry["clarity"] = int(m.group(1))
            when = first_commit_date(path)
            if when:
                entry["notedAt"] = when
            verified[str(day)] = entry

    out = {
        "generatedAt": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(timespec="seconds"),
        "lastDay": idx.get("lastDay"),
        "verified": verified,
    }
    with open(os.path.join(ROOT, "progress.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")

    print("notes created: %s" % (created or "none"))
    print("verified days: %d" % len(verified))
    if verified:
        cl = [v["clarity"] for v in verified.values() if "clarity" in v]
        if cl:
            print("mean clarity: %.2f over %d ratings" % (sum(cl) / len(cl), len(cl)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
