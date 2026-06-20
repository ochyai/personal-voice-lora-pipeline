#!/usr/bin/env python3
"""フォルダの中の文章ファイルを、種コーパス（seed_corpus.jsonl）に変換する.

note 記事(Markdown), ブログ, 書籍の下書き, 講演原稿, 対談の文字起こしなど、
.txt / .md / .docx / .pdf を1つのフォルダに集めて、これを実行するだけ。

  python collect/files_to_seed_jsonl.py --in ~/my-writings --out ~/voice-lora/raw/seed_corpus.jsonl

出力は1行1かたまりの JSONL:  {"text": "...", "source": "files:<拡張子>"}
400字未満の断片と重複（先頭512字ハッシュ）は自動で捨てます。
.docx は python-docx、.pdf は pdftotext(poppler-utils) があれば読みます。
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

MIN_CHARS = 400
MAX_CHARS = 16000


def chunk(text):
    text = (text or "").strip()
    if len(text) <= MAX_CHARS:
        return [text] if len(text) >= MIN_CHARS else []
    out, cur = [], ""
    for para in re.split(r"\n\n+", text):
        if len(cur) + len(para) + 2 > MAX_CHARS:
            if len(cur) >= MIN_CHARS:
                out.append(cur.strip())
            cur = para
        else:
            cur = cur + "\n\n" + para if cur else para
    if len(cur) >= MIN_CHARS:
        out.append(cur.strip())
    return out


def read_txt(p):
    return p.read_text(encoding="utf-8", errors="ignore")


def read_docx(p):
    try:
        from docx import Document
    except ImportError:
        print(f"  ! {p.name}: python-docx が無いのでスキップ（pip install python-docx）",
              file=sys.stderr)
        return ""
    doc = Document(str(p))
    return "\n\n".join(para.text.strip() for para in doc.paragraphs if para.text.strip())


def read_pdf(p):
    try:
        out = subprocess.run(["pdftotext", "-enc", "UTF-8", str(p), "-"],
                             capture_output=True, text=True, timeout=120)
        return out.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        print(f"  ! {p.name}: pdftotext が無いのでスキップ（brew/apt で poppler-utils）",
              file=sys.stderr)
        return ""


READERS = {".txt": read_txt, ".md": read_txt, ".markdown": read_txt,
           ".docx": read_docx, ".pdf": read_pdf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="文章ファイルの入ったフォルダ")
    ap.add_argument("--out", required=True, help="出力 JSONL（例: ~/voice-lora/raw/seed_corpus.jsonl）")
    ap.add_argument("--source", default="files", help="source タグの接頭辞")
    args = ap.parse_args()

    indir = Path(args.indir).expanduser()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    seen, n, total = set(), 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for p in sorted(indir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in READERS:
                continue
            text = READERS[p.suffix.lower()](p)
            for c in chunk(text):
                h = hashlib.sha256(c[:512].encode("utf-8", "ignore")).hexdigest()[:16]
                if h in seen:
                    continue
                seen.add(h)
                f.write(json.dumps(
                    {"text": c, "source": f"{args.source}:{p.suffix.lower().lstrip('.')}",
                     "id": h}, ensure_ascii=False) + "\n")
                n += 1
                total += len(c)
    print(f"書き出し: {out}  ({n:,} 件 / {total:,} 字 ≈ {total/2_000_000:.1f}M tok 日本語)")
    if n == 0:
        print("  ! 0件でした。フォルダの中身（.txt/.md/.docx/.pdf）と --in を確認してください。",
              file=sys.stderr)


if __name__ == "__main__":
    main()
