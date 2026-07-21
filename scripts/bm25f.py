"""BM25F lexical index over model metadata (name / tags / categories / description).

Kept in an importable module (rather than a script's ``__main__``) so a pickled index can be
reliably reloaded later by any consumer (e.g. the query harness) via ``bm25f.BM25FIndex``.
"""

import ast
import html
import json
import math
import re
from collections import Counter, defaultdict

import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9]+")
TAG_SPLIT_RE = re.compile(r"[\s/_-]+")


def safe_parse_annotation(s):
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return {}
    return {}


def strip_markup(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)      # images
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", r" \g<0> ", text)  # links
    text = re.sub(r"[#*_>`~]+", " ", text)                 # markdown formatting
    return text


def normalize_text(x):
    return strip_markup(x).lower()


def tokenize(x):
    return TOKEN_RE.findall(x)


def tokens_from_tag(tag_obj):
    parts = []
    for key in ("name", "slug"):
        val = (tag_obj.get(key) or "").lower()
        if not val:
            continue
        parts.append(val)
        parts.extend(TAG_SPLIT_RE.split(val))
    return tokenize(" ".join(p for p in parts if p))


# Module-level factories (instead of lambdas) so a built index is picklable.
def _int_dd():
    return defaultdict(int)


def _int_dd_dd():
    return defaultdict(_int_dd)


class BM25FIndex:
    def __init__(self, field_weights, field_b, k1=1.3, epsilon=1e-9):
        self.k1 = k1
        self.field_weights = field_weights
        self.field_b = field_b
        self.epsilon = epsilon
        self.N = 0
        self.field_lengths = defaultdict(_int_dd)
        self.avg_field_len = defaultdict(float)
        self.df = defaultdict(int)
        self.postings = defaultdict(_int_dd_dd)
        self.num_boosts = {}

    def _add_field_tokens(self, doc_id, field, tokens):
        self.field_lengths[field][doc_id] = len(tokens)
        for t, c in Counter(tokens).items():
            self.postings[t][doc_id][field] = c

    def add_document(self, doc_id, doc):
        self.N += 1
        name_toks = tokenize(normalize_text(doc.get("name", "")))
        desc_toks = tokenize(normalize_text(doc.get("description", "")))

        tag_tokens = []
        for t in (doc.get("tags") or []):
            if isinstance(t, dict):
                tag_tokens.extend(tokens_from_tag(t))
            else:
                tag_tokens.extend(tokenize(str(t).lower()))

        cat_tokens = []
        for c in (doc.get("categories") or []):
            if isinstance(c, dict):
                cat_tokens.extend(tokenize((c.get("name") or "").lower()))
            else:
                cat_tokens.extend(tokenize(str(c).lower()))

        fields = {
            "name": name_toks,
            "tags": tag_tokens,
            "categories": cat_tokens,
            "description": desc_toks,
        }
        for f, toks in fields.items():
            self._add_field_tokens(doc_id, f, toks)

    def finalize(self):
        seen = defaultdict(set)
        for term, doc_map in self.postings.items():
            for doc_id in doc_map.keys():
                seen[term].add(doc_id)
        for term, docs in seen.items():
            self.df[term] = len(docs)
        for f, lens in self.field_lengths.items():
            self.avg_field_len[f] = (sum(lens.values()) / len(lens)) if lens else 0.0

    def _idf(self, term):
        n_qi = self.df.get(term, 0)
        return math.log((self.N - n_qi + 0.5) / (n_qi + 0.5 + self.epsilon) + 1.0)

    def _accum_fielded_tf(self, term, doc_id):
        total = 0.0
        for f, w_f in self.field_weights.items():
            tf_f = self.postings.get(term, {}).get(doc_id, {}).get(f, 0)
            if not tf_f:
                continue
            b_f = self.field_b.get(f, 0.75)
            len_f = self.field_lengths[f].get(doc_id, 0)
            avglen_f = self.avg_field_len.get(f, 1.0) or 1.0
            norm = (1 - b_f) + b_f * (len_f / avglen_f)
            total += w_f * (tf_f / norm)
        return total

    def score(self, q_tokens, doc_id):
        s = 0.0
        for t in q_tokens:
            wdt = self._accum_fielded_tf(t, doc_id)
            if wdt <= 0:
                continue
            s += self._idf(t) * ((self.k1 + 1) * wdt) / (self.k1 + wdt)
        s += self.num_boosts.get(doc_id, 0.0)
        return s

    def search(self, query, top_k=20):
        q_tokens = tokenize(normalize_text(query))
        candidates = set()
        for t in q_tokens:
            candidates.update(self.postings.get(t, {}).keys())
        if not candidates:
            return []
        scored = [(self.score(q_tokens, d), d) for d in candidates]
        scored.sort(reverse=True)
        return scored[:top_k]


def build_index(meta_by_id):
    field_weights = {"name": 2.0, "tags": 1.5, "categories": 1.2, "description": 1.0}
    field_b = {"name": 0.6, "tags": 0.6, "categories": 0.6, "description": 0.75}
    idx = BM25FIndex(field_weights, field_b, k1=1.3)
    for doc_id, doc in meta_by_id.items():
        idx.add_document(doc_id, doc)
    idx.finalize()
    return idx


def build_metadata_idx(metadata_csv):
    """Build the BM25F index from a CSV. Uses an ``annotation`` JSON column when present,
    else falls back to a ``caption`` column as the description."""
    df = pd.read_csv(metadata_csv)
    has_annotation = "annotation" in df.columns
    has_caption = "caption" in df.columns

    meta_by_id = {}
    for _, row in df.iterrows():
        uid = str(row["uid"])
        if has_annotation and isinstance(row.get("annotation"), str):
            ann_raw = safe_parse_annotation(row["annotation"])
            ann = ann_raw.get(uid, {}) if isinstance(ann_raw, dict) else {}
            meta_by_id[uid] = {
                "uid": uid,
                "name": ann.get("name", "") or "",
                "description": ann.get("description", "") or "",
                "tags": ann.get("tags", []) or [],
                "categories": ann.get("categories", []) or [],
            }
        else:
            caption = row.get("caption", "") if has_caption else ""
            meta_by_id[uid] = {
                "uid": uid,
                "name": "",
                "description": caption if isinstance(caption, str) else "",
                "tags": [],
                "categories": [],
            }

    idx = build_index(meta_by_id)
    print(f"[info] Indexed {idx.N} documents with {len(idx.df)} unique terms.")
    return idx, meta_by_id


def bm25_search(collection, prompt, top_k=10):
    """Query a loaded ``(idx, meta_by_id)`` tuple; returns ranked uids."""
    idx, _ = collection
    hits = sorted(idx.search(prompt, top_k=top_k), key=lambda x: x[0], reverse=True)[:top_k]
    return [uid for _, uid in hits]
