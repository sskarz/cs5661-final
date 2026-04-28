#!/usr/bin/env python3
"""Parse AndroidControl GCS shards and emit a11y-augmented JSONL.

Reads GZIP-compressed TFRecord shards from data/androidcontrol_gcs/ (downloaded
via download_androidcontrol_gcs.py), deserializes per-screenshot
AndroidAccessibilityForest protobufs, filters to clickable+visible nodes, and
writes:

    data/androidcontrol_a11y/
        train.jsonl   (GCS-train: 13603 ep, ~190K rows)
        val.jsonl     (GCS-val:     137 ep,  ~1.7K rows)
        test.jsonl    (GCS-test:   1543 ep, ~19K rows)
        images/
            <episode_id>_<step_index>.png
        splits.json   (copied from gcs dir, for provenance)

Each JSONL row matches the existing data/androidcontrol/{train,test}.jsonl
schema PLUS an `a11y` field listing visible+clickable elements with normalized
bboxes [0,1].

Schema per row:
    {
      "messages": [{"role":"user", "content":[{"type":"image"}, {"type":"text", "text": <step_instruction>}]},
                   {"role":"assistant", "content":[{"type":"text", "text": <action json>}]}],
      "episode_id": int,
      "step_index": int,
      "total_steps": int,
      "granularity": "step" | "goal",
      "image": "images/<ep>_<si>.png",
      "goal": <high-level task instruction>,
      "a11y": [
          {"id": int, "bbox": [x0,y0,x1,y1] in [0,1],
           "text": str, "content_description": str,
           "view_id_resource_name": str, "class_name": str,
           "is_clickable": bool, "is_long_clickable": bool, "is_editable": bool,
           "depth": int}, ...
      ]
    }

Splits routed by episode_id ∈ GCS-{train,val,test}. The HF-mirror split is
ignored because it cross-leaks (89% of HF-test is in GCS-train; see TRAINING_LOG).

Dependencies:
    .venv/bin/python -m pip install android_env tensorflow-cpu
or simply:
    uv pip install android-env tensorflow-cpu
(android_env exposes the AndroidAccessibilityForest pb2 module; tensorflow is
used only for tf.train.Example parsing — could be replaced with a minimal
manual proto, but tf-cpu is reliable and ~200MB.)

Usage:
    uv run python scripts/parse_a11y_data.py \\
        --gcs-dir data/androidcontrol_gcs \\
        --output-dir data/androidcontrol_a11y \\
        --shards 0-19          # all shards
        --shards 0              # smoke (one shard)
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import struct
import sys
from io import BytesIO
from pathlib import Path

# --- Lazy imports — keep top-level fast for --help -----------------------------

_EXAMPLE_FDS_SOURCE = b"""
syntax = "proto2";
package tensorflow;
message BytesList { repeated bytes value = 1; }
message FloatList { repeated float value = 1 [packed = true]; }
message Int64List { repeated int64 value = 1 [packed = true]; }
message Feature {
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
  }
}
message Features { map<string, Feature> feature = 1; }
message Example { optional Features features = 1; }
"""


def _build_example_class():
    """Compile the tf.train.Example proto in-memory — no tensorflow dep needed.

    We use protoc via google.protobuf to register the schema once and return
    the Example message class via the default descriptor pool.
    """
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "_inline_example.proto"
    fdp.package = "tensorflow"
    fdp.syntax = "proto2"

    def add_msg(name, fields):
        msg = fdp.message_type.add()
        msg.name = name
        for fname, ftype, fnum, label, packed in fields:
            f = msg.field.add()
            f.name = fname
            f.number = fnum
            f.type = ftype
            f.label = label
            if packed:
                f.options.packed = True
        return msg

    LABEL_OPT = 1
    LABEL_REP = 3
    TYPE_BYTES = 12
    TYPE_FLOAT = 2
    TYPE_INT64 = 3
    TYPE_MSG = 11

    add_msg("BytesList", [("value", TYPE_BYTES, 1, LABEL_REP, False)])
    add_msg("FloatList", [("value", TYPE_FLOAT, 1, LABEL_REP, True)])
    add_msg("Int64List", [("value", TYPE_INT64, 1, LABEL_REP, True)])

    feat = fdp.message_type.add()
    feat.name = "Feature"
    for fname, fnum, type_name in [
        ("bytes_list", 1, "BytesList"),
        ("float_list", 2, "FloatList"),
        ("int64_list", 3, "Int64List"),
    ]:
        f = feat.field.add()
        f.name = fname
        f.number = fnum
        f.type = TYPE_MSG
        f.label = LABEL_OPT
        f.type_name = ".tensorflow." + type_name
        f.oneof_index = 0
    feat.oneof_decl.add().name = "kind"

    feats = fdp.message_type.add()
    feats.name = "Features"
    map_entry = feats.nested_type.add()
    map_entry.name = "FeatureEntry"
    map_entry.options.map_entry = True
    k = map_entry.field.add()
    k.name = "key"; k.number = 1; k.type = 9; k.label = LABEL_OPT  # TYPE_STRING=9
    v = map_entry.field.add()
    v.name = "value"; v.number = 2; v.type = TYPE_MSG; v.label = LABEL_OPT
    v.type_name = ".tensorflow.Feature"
    feature_field = feats.field.add()
    feature_field.name = "feature"
    feature_field.number = 1
    feature_field.type = TYPE_MSG
    feature_field.label = LABEL_REP
    feature_field.type_name = ".tensorflow.Features.FeatureEntry"

    ex = fdp.message_type.add()
    ex.name = "Example"
    fld = ex.field.add()
    fld.name = "features"; fld.number = 1; fld.type = TYPE_MSG
    fld.label = LABEL_OPT; fld.type_name = ".tensorflow.Features"

    # Use an isolated pool to avoid clashing with other libs (e.g. tensorflow,
    # google-cloud-*) that also register tensorflow.Example in the default pool.
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    ex_desc = pool.FindMessageTypeByName("tensorflow.Example")
    # protobuf 5+ uses GetMessageClass; older versions used MessageFactory.GetPrototype
    if hasattr(message_factory, "GetMessageClass"):
        return message_factory.GetMessageClass(ex_desc)
    return message_factory.MessageFactory(pool).GetPrototype(ex_desc)


# Module-level cache so each forked worker only compiles the proto once
# regardless of multiprocess start method (fork vs spawn).
_EXAMPLE_CLS = None
_FOREST_PB = None


def _import_protos():
    global _EXAMPLE_CLS, _FOREST_PB
    if _EXAMPLE_CLS is None:
        _EXAMPLE_CLS = _build_example_class()
    if _FOREST_PB is None:
        try:
            from android_env.proto.a11y import android_accessibility_forest_pb2 as forest_pb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Need android_env for AndroidAccessibilityForest proto. "
                f"Install with `uv pip install android-env`. ({e})"
            )
        _FOREST_PB = forest_pb
    return _EXAMPLE_CLS, _FOREST_PB


# --- Action canonicalization (mirrors prepare_androidcontrol.py) -----------------

TERMINAL_ACTION = {"action": "done"}


def canonicalize_action(raw_step: dict, img_w: int, img_h: int) -> dict:
    """Convert AndroidControl action -> our SFT schema (matches prepare script)."""
    raw_type = (raw_step.get("action_type") or "").lower().strip()
    out: dict = {"action": raw_type}
    if raw_type in {"click", "tap", "long_press"}:
        # Align with prepare_androidcontrol.py: collapse long_press → tap so the
        # legacy eval_androidcontrol.actions_match() works on these rows too.
        # AndroidControl's "long_press" is rare (<1% of train); not worth a
        # separate eval branch.
        out["action"] = "tap"
        x_px = raw_step.get("x")
        y_px = raw_step.get("y")
        if x_px is None or y_px is None:
            return {"action": "wait"}  # malformed; skip-equivalent
        out["x"] = round(max(0.0, min(1.0, float(x_px) / max(img_w, 1))), 4)
        out["y"] = round(max(0.0, min(1.0, float(y_px) / max(img_h, 1))), 4)
    elif raw_type in {"scroll", "swipe"}:
        out["action"] = "scroll"
        direction = raw_step.get("direction")
        if direction:
            out["direction"] = direction.lower()
    elif raw_type == "input_text" or raw_type == "type":
        out["action"] = "type"
        if raw_step.get("text") is not None:
            out["text"] = str(raw_step["text"])
    elif raw_type == "open_app":
        out["action"] = "open_app"
        if raw_step.get("app_name") is not None:
            out["app_name"] = str(raw_step["app_name"])
    elif raw_type in {"navigate_back", "navigate_home", "wait"}:
        pass
    else:
        out["action"] = raw_type or "wait"
    return out


# --- TFRecord reader (no TensorFlow dependency for the framing) -----------------

def read_tfrecords(path: Path):
    """Stream record bytes from a GZIP-compressed TFRecord file.

    TFRecord frame: len(8 LE uint64) + len_crc(4) + data(len) + data_crc(4).
    We skip CRC validation (~5x speedup; protobuf parse will fail loudly if
    a record is truncated).
    """
    with gzip.open(str(path), "rb") as f:
        while True:
            head = f.read(8)
            if not head:
                return
            if len(head) < 8:
                raise IOError(f"Truncated header in {path}")
            (length,) = struct.unpack("<Q", head)
            f.read(4)  # skip len-crc
            data = f.read(length)
            if len(data) < length:
                raise IOError(f"Truncated data in {path}")
            f.read(4)  # skip data-crc
            yield data


# --- A11y forest -> compact node list ----------------------------------------

def forest_to_nodes(forest_bytes: bytes, img_w: int, img_h: int, forest_pb) -> list[dict]:
    """Deserialize AndroidAccessibilityForest, return clickable+visible nodes."""
    forest = forest_pb.AndroidAccessibilityForest.FromString(forest_bytes)
    nodes: list[dict] = []
    for window in forest.windows:
        for node in window.tree.nodes:
            if not getattr(node, "is_visible_to_user", False):
                continue
            # Keep clickable, long_clickable, editable, OR has visible text /
            # content_description. Icon-only buttons (no text, only
            # content_description) are common Material targets and were
            # silently dropped before this filter was widened.
            useful = (
                node.is_clickable or node.is_long_clickable or node.is_editable
                or (node.text and node.text.strip())
                or (node.content_description and node.content_description.strip())
            )
            if not useful:
                continue
            r = node.bounds_in_screen
            x0, y0, x1, y1 = r.left, r.top, r.right, r.bottom
            # Reject zero-area or off-screen.
            if x1 <= x0 or y1 <= y0:
                continue
            if x1 <= 0 or y1 <= 0 or x0 >= img_w or y0 >= img_h:
                continue
            nodes.append({
                "id": int(node.unique_id),
                "bbox": [
                    round(max(0.0, min(1.0, x0 / img_w)), 4),
                    round(max(0.0, min(1.0, y0 / img_h)), 4),
                    round(max(0.0, min(1.0, x1 / img_w)), 4),
                    round(max(0.0, min(1.0, y1 / img_h)), 4),
                ],
                "text": (node.text or "").strip()[:80],
                "content_description": (node.content_description or "").strip()[:80],
                "view_id_resource_name": (node.view_id_resource_name or "").split("/")[-1][:60],
                "class_name": (node.class_name or "").split(".")[-1],
                "is_clickable": bool(node.is_clickable),
                "is_long_clickable": bool(node.is_long_clickable),
                "is_editable": bool(node.is_editable),
                "depth": int(getattr(node, "depth", 0)),
            })
    return nodes


# --- Per-shard worker --------------------------------------------------------

def process_shard(args_tuple):
    """Run inside a worker process: read one shard, emit per-split JSONL fragments."""
    (shard_path, out_dir, ep_to_split, save_pngs) = args_tuple
    Example, forest_pb = _import_protos()
    from PIL import Image

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    # Open per-split append handles for THIS worker; merge after.
    pid = os.getpid()
    handles = {
        sp: open(out_dir / f"_part_{sp}_pid{pid}.jsonl", "w")
        for sp in ("train", "val", "test")
    }
    n_in, n_out, n_skip_no_split, n_bad_action = 0, 0, 0, 0

    for raw in read_tfrecords(shard_path):
        ex = Example.FromString(raw)
        feats = ex.features.feature

        ep_id = int(feats["episode_id"].int64_list.value[0])
        split = ep_to_split.get(ep_id)
        if split is None:
            n_skip_no_split += 1
            continue

        goal = feats["goal"].bytes_list.value[0].decode("utf-8", errors="replace")
        screenshots = list(feats["screenshots"].bytes_list.value)
        widths = list(feats["screenshot_widths"].int64_list.value)
        heights = list(feats["screenshot_heights"].int64_list.value)
        a11y_blobs = list(feats["accessibility_trees"].bytes_list.value)
        actions_json = list(feats["actions"].bytes_list.value)
        step_instructions = list(feats["step_instructions"].bytes_list.value)

        n_steps = len(actions_json)
        n_in += 1

        # Each record has N screenshots, N a11y trees, N actions (or N+1
        # screenshots with a terminal state). The first N rows are real steps.
        for i in range(n_steps):
            try:
                act_raw = json.loads(actions_json[i].decode("utf-8", errors="replace"))
            except Exception:
                n_bad_action += 1
                continue
            w = int(widths[i]); h = int(heights[i])
            try:
                a11y = forest_to_nodes(a11y_blobs[i], w, h, forest_pb)
            except Exception:
                a11y = []
            action = canonicalize_action(act_raw, w, h)
            instr = step_instructions[i].decode("utf-8", errors="replace")

            # Match prepare_androidcontrol.py filename convention so the two
            # image dirs are interchangeable.
            img_rel = f"images/{ep_id:05d}_{i:02d}.png"
            if save_pngs:
                img_path = out_dir / img_rel
                if not img_path.exists():
                    try:
                        Image.open(BytesIO(screenshots[i])).convert("RGB").save(img_path, "PNG")
                    except Exception:
                        continue

            row = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": instr},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": json.dumps(action)},
                    ]},
                ],
                "episode_id": ep_id,
                "step_index": i,
                "total_steps": n_steps,
                "granularity": "step",
                "image": img_rel,
                "image_w": w,
                "image_h": h,
                "goal": goal,
                "a11y": a11y,
            }
            handles[split].write(json.dumps(row) + "\n")
            n_out += 1

    for h in handles.values():
        h.close()
    return {
        "shard": shard_path.name,
        "episodes_in": n_in,
        "rows_out": n_out,
        "skipped_no_split": n_skip_no_split,
        "skipped_bad_action": n_bad_action,
    }


# --- Main --------------------------------------------------------------------

def parse_shard_spec(spec: str, total: int = 20) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(s for s in out if 0 <= s < total)


def merge_partials(out_dir: Path) -> dict[str, int]:
    """Merge per-worker partials into one file per split.

    Sorts merged rows by (episode_id, step_index) so that downstream
    consumers (eval seeded shuffles, dataset slicing) see a stable order
    independent of how PIDs/workers were assigned during parse.
    """
    counts: dict[str, int] = {}
    for sp in ("train", "val", "test"):
        final = out_dir / f"{sp}.jsonl"
        rows: list[tuple[int, int, str]] = []
        for part in sorted(out_dir.glob(f"_part_{sp}_pid*.jsonl")):
            with open(part) as fin:
                for line in fin:
                    try:
                        r = json.loads(line)
                        rows.append((int(r["episode_id"]), int(r["step_index"]), line))
                    except Exception:
                        continue
            part.unlink()
        rows.sort(key=lambda t: (t[0], t[1]))
        with open(final, "w") as fout:
            for _, _, line in rows:
                fout.write(line)
        counts[sp] = len(rows)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-dir", type=Path, default=Path("data/androidcontrol_gcs"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/androidcontrol_a11y"))
    ap.add_argument("--shards", default="0-19")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pngs", action="store_true",
                    help="Skip screenshot extraction (a11y JSONL only).")
    args = ap.parse_args()

    splits_path = args.gcs_dir / "splits.json"
    if not splits_path.exists():
        sys.exit(f"Missing {splits_path}; run download_androidcontrol_gcs.py --splits-only first.")

    splits = json.load(open(splits_path))
    ep_to_split: dict[int, str] = {}
    for sp_name, ep_ids in splits.items():
        target = "val" if sp_name == "validation" else sp_name
        for ep in ep_ids:
            ep_to_split[int(ep)] = target

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Provenance copy of the splits file
    (args.output_dir / "splits.json").write_bytes(splits_path.read_bytes())
    # Clean stale partials from prior aborted runs — otherwise they'd be
    # merged in addition to fresh output and produce duplicates.
    for stale in args.output_dir.glob("_part_*_pid*.jsonl"):
        stale.unlink()

    shards = parse_shard_spec(args.shards)
    shard_paths = []
    for s in shards:
        p = args.gcs_dir / f"android_control-{s:05d}-of-00020"
        if not p.exists():
            print(f"[skip] missing shard {p}", file=sys.stderr)
            continue
        shard_paths.append(p)
    if not shard_paths:
        sys.exit("No shard files present.")

    print(f"[plan] {len(shard_paths)} shards, {args.workers} workers, splits routed.")
    save_pngs = not args.no_pngs

    work = [(p, args.output_dir, ep_to_split, save_pngs) for p in shard_paths]
    if args.workers <= 1:
        results = [process_shard(w) for w in work]
    else:
        with mp.Pool(args.workers) as pool:
            results = pool.map(process_shard, work)

    total_out = sum(r["rows_out"] for r in results)
    total_in = sum(r["episodes_in"] for r in results)
    skipped = sum(r["skipped_no_split"] for r in results)
    bad_act = sum(r["skipped_bad_action"] for r in results)
    print(f"[shards] {len(results)} processed | episodes={total_in} | rows={total_out} | "
          f"skipped_no_split={skipped} | skipped_bad_action={bad_act}")

    counts = merge_partials(args.output_dir)
    print(f"[merged] train={counts['train']}  val={counts['val']}  test={counts['test']}")


if __name__ == "__main__":
    main()
