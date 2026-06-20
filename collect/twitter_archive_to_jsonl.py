#!/usr/bin/env python3
"""X（旧Twitter）公式アーカイブを、自分のツイートだけの JSONL に変換する.

X の設定 →「アカウント」→「データのアーカイブをダウンロード」で zip を入手し、
展開して中の data/tweets.js（古い書き出しなら tweet.js）を渡します。

  python collect/twitter_archive_to_jsonl.py \\
      --tweets ~/twitter-archive/data/tweets.js \\
      --out ~/voice-lora/raw/twitter.jsonl

リツイート（RT）と他人への返信は既定で除外します（あなたの“地の文”だけ残す）。
短すぎるツイートも除外（--min-chars、既定60字）。URL や @メンション、#タグの
記号は軽く掃除します。出力:  {"text": "...", "source": "twitter"}
"""
import argparse
import json
import re
from pathlib import Path


def load_tweets_js(path):
    """tweets.js は `window.YTD.tweets.part0 = [ ... ]` という JS。前置きを剥がす。"""
    raw = Path(path).expanduser().read_text(encoding="utf-8", errors="ignore")
    i = raw.find("[")
    if i < 0:
        raise SystemExit("tweets.js の中に JSON 配列が見つかりません。ファイルを確認してください。")
    data = json.loads(raw[i:])
    # 各要素は {"tweet": {...}} または素の {...}
    return [d.get("tweet", d) for d in data]


def clean(text):
    text = re.sub(r"https?://\S+", "", text)          # URL
    text = re.sub(r"@\w+", "", text)                  # メンション
    text = re.sub(r"#(\w+)", r"\1", text)             # ハッシュタグの # だけ取る
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+\n", "\n", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tweets", required=True, help="tweets.js / tweet.js のパス")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=60)
    ap.add_argument("--keep-replies", action="store_true", help="返信ツイートも残す")
    args = ap.parse_args()

    tweets = load_tweets_js(args.tweets)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    n, skipped = 0, 0
    seen = set()
    with open(out, "w", encoding="utf-8") as f:
        for t in tweets:
            text = t.get("full_text") or t.get("text") or ""
            if text.startswith("RT @"):                     # リツイート
                skipped += 1
                continue
            if not args.keep_replies and t.get("in_reply_to_status_id_str"):
                skipped += 1
                continue
            text = clean(text)
            if len(text) < args.min_chars:
                skipped += 1
                continue
            if text in seen:
                continue
            seen.add(text)
            f.write(json.dumps({"text": text, "source": "twitter"},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"書き出し: {out}  (採用 {n:,} 件 / 除外 {skipped:,} 件)")
    print("  ※ 短いツイートは “声” の情報が薄いので、note やブログなど長文と一緒に使うのがおすすめです。")


if __name__ == "__main__":
    main()
