#!/usr/bin/env python3
"""Validate public Markdown scope, size, and accidental internal disclosures."""
from pathlib import Path
import argparse, ipaddress, re, subprocess, sys

README_LIMIT = 500
DOCUMENT_LIMIT = 1000
WORD = re.compile(r"\b[\w'-]+\b")
IPV4_CANDIDATE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
IPV6_CANDIDATE = re.compile(r"(?<![\w:])\[?(?=[0-9a-fA-F:]*:)[0-9a-fA-F:]{3,}\]?(?![\w:])")
FORBIDDEN = {
    "internal tool references": re.compile(r"\b(?:Codex|ChatGPT)\b", re.I),
    "prompt/developer discussion": re.compile(r"\b(?:internal prompt|developer instruction|prompt engineering)\b", re.I),
    "private absolute paths": re.compile(r"(?<![\w])/(?:home|data|cm|scratch|work|mnt)/", re.I),
    "credential setting": re.compile(r"(?:RTM_LLM_API_KEY\s*=|sk-[A-Za-z0-9]{20,})", re.I),
    "internal Git operation guidance": re.compile(r"\bgit\s+(?:add|commit|push|reset|clean)\b", re.I),
    "internal scheduler commands": re.compile(r"\b(?:sbatch|squeue|salloc|nvidia-smi)\b", re.I),
}

def private_network_address(value):
    """Recognize private, loopback, link-local, unspecified and reserved IP literals."""
    try:
        address = ipaddress.ip_address(value.strip("[]"))
    except ValueError:
        return False
    return (address.is_private or address.is_loopback or address.is_link_local
            or address.is_unspecified or address.is_reserved or address.is_multicast)

def private_address_lines(text):
    """Yield line numbers containing an IP literal that is not globally routable."""
    for line_no, line in enumerate(text.splitlines(), 1):
        candidates = set(IPV4_CANDIDATE.findall(line))
        candidates.update(IPV6_CANDIDATE.findall(line))
        if any(private_network_address(value) for value in candidates):
            yield line_no

def tracked_markdown(root):
    try:
        names = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.DEVNULL,
        ).decode().split("\0")
    except (OSError, subprocess.CalledProcessError):
        names = [str(p.relative_to(root)) for p in root.rglob("*.md") if ".git" not in p.parts]
    return sorted({name for name in names if name.lower().endswith(".md") and (root/name).is_file()})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args(); root=args.root.resolve(); errors=[]; files=tracked_markdown(root)
    if not files: errors.append("No public Markdown files found")
    for name in files:
        path=root/name; text=path.read_text(encoding="utf-8",errors="strict")
        words=len(WORD.findall(text)); limit=README_LIMIT if name=="README.md" else DOCUMENT_LIMIT
        if words>limit: errors.append(f"{name}: {words} words exceeds {limit}")
        for label,pattern in FORBIDDEN.items():
            for line_no,line in enumerate(text.splitlines(),1):
                if pattern.search(line): errors.append(f"{name}:{line_no}: {label}")
        for line_no in private_address_lines(text):
            errors.append(f"{name}:{line_no}: private or non-global network address")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)",text):
            if target.startswith(("http://","https://","mailto:","#")): continue
            local=(path.parent/target.split("#",1)[0]).resolve()
            if not local.exists(): errors.append(f"{name}: broken local link {target}")
    if errors:
        print("PUBLIC DOCUMENTATION CHECK FAILED")
        print("\n".join(f"- {e}" for e in errors));return 1
    print(f"PUBLIC DOCUMENTATION CHECK PASSED: {len(files)} Markdown files")
    for name in files:
        n=len(WORD.findall((root/name).read_text(encoding="utf-8")))
        print(f"{n:4} {name}")
    return 0

if __name__=="__main__": sys.exit(main())
