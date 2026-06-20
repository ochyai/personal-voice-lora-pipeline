#!/usr/bin/env python3
"""Gmail（Google Takeout の mbox）を、自分が送ったメールだけの JSONL に変換する.

Google Takeout（takeout.google.com）で「メール」を選ぶと .mbox がもらえます。
自分のアドレスを --me で渡すと、From が自分のメールだけを抜き出します。

  python collect/gmail_mbox_to_jsonl.py \\
      --mbox ~/Takeout/Mail/all.mbox \\
      --me you@example.com \\
      --out ~/voice-lora/raw/email.jsonl

出力は build_corpus.py の add_authored_jsonl が読める形:
  {"body": "...", "is_author_sent": true, "source": "email"}
引用行（> で始まる返信）や署名はある程度落とします。標準ライブラリのみで動きます。
"""
import argparse
import json
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path


def get_body(msg):
    """plain text 本文を取り出す（HTMLしか無ければタグを軽く除去）。"""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return part.get_content()
                except Exception:
                    pass
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                try:
                    return re.sub(r"<[^>]+>", " ", part.get_content())
                except Exception:
                    pass
        return ""
    try:
        body = msg.get_content()
    except Exception:
        return ""
    if msg.get_content_type() == "text/html":
        body = re.sub(r"<[^>]+>", " ", body)
    return body


def clean(body):
    body = re.sub(r"^>.*$", "", body, flags=re.M)               # 引用行
    body = re.sub(r"\nOn .*wrote:\s*$", "", body, flags=re.M)   # 英語の引用ヘッダ
    body = re.sub(r"\n-- \n.*$", "", body, flags=re.S)          # 署名（-- 以降）
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def main():
    import mailbox
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbox", required=True)
    ap.add_argument("--me", required=True, help="自分のメールアドレス（From 判定用）。カンマ区切りで複数可")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()

    me = [m.strip().lower() for m in args.me.split(",") if m.strip()]
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    box = mailbox.mbox(str(Path(args.mbox).expanduser()), factory=None)
    parser = BytesParser(policy=policy.default)
    n, scanned = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for raw_msg in box:
            scanned += 1
            # 旧式 mbox メッセージを、本文取得しやすい modern EmailMessage に再解析する
            try:
                msg = parser.parsebytes(raw_msg.as_bytes())
            except Exception:
                continue
            frm = (msg.get("From") or "").lower()
            if not any(addr in frm for addr in me):
                continue
            body = clean(get_body(msg))
            if len(body) < args.min_chars:
                continue
            f.write(json.dumps(
                {"body": body, "is_author_sent": True, "source": "email"},
                ensure_ascii=False) + "\n")
            n += 1
    print(f"走査 {scanned:,} 通 → 自分が送ったメール {n:,} 通を書き出し: {out}")
    if n == 0:
        print("  ! 0件。--me のアドレスが From と一致しているか確認してください。")


if __name__ == "__main__":
    main()
