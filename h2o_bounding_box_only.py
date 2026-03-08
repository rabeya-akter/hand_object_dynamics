#!/usr/bin/env python3
"""
H2O (cam4) action segment prompting using ONLY bounding_box/*.txt (100DOH outputs)
- Uses segment ranges (start_act:end_act) from the provided action label file (e.g., action_train.txt)
- Samples exactly N frames per segment, evenly from start_act to end_act
- Reads per-frame bounding boxes from: <seq>/cam4/bounding_box/<frame:06d>.txt
- Builds EgoHOD-style Hand Object Dynamics prompt using RAW pixel centers (cx, cy) (NO normalization)
- Optional Qwen inference + CSV/JSONL outputs
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


ACTION_LABELS = {
    0: "background",
    1: "grab book",
    2: "grab espresso",
    3: "grab lotion",
    4: "grab spray",
    5: "grab milk",
    6: "grab cocoa",
    7: "grab chips",
    8: "grab cappuccino",
    9: "place book",
    10: "place espresso",
    11: "place lotion",
    12: "place spray",
    13: "place milk",
    14: "place cocoa",
    15: "place chips",
    16: "place cappuccino",
    17: "open lotion",
    18: "open milk",
    19: "open chips",
    20: "close lotion",
    21: "close milk",
    22: "close chips",
    23: "pour milk",
    24: "take out espresso",
    25: "take out cocoa",
    26: "take out chips",
    27: "take out cappuccino",
    28: "put in espresso",
    29: "put in cocoa",
    30: "put in cappuccino",
    31: "apply lotion",
    32: "apply spray",
    33: "read book",
    34: "read espresso",
    35: "spray spray",
    36: "squeeze lotion",
}


# Order in bounding_box txt (lines 2..6)
ENTITY_NAMES = [
    "left hand",
    "right hand",
    "left hand object",
    "right hand object",
    "two hand object",
]


def format_action_options() -> str:
    return "\n".join([f"{k}: {ACTION_LABELS[k]}" for k in sorted(ACTION_LABELS.keys())])


def parse_action_label_file(action_label_txt: Path) -> List[Dict[str, Any]]:
    """
    Expected columns per line:
      id path action_label start_act end_act start_frame end_frame
    Example:
      1 subject1/h1/0 16 0 61 0 449
    """
    rows: List[Dict[str, Any]] = []
    for line in action_label_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("id "):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        rows.append(
            {
                "id": int(parts[0]),
                "path": parts[1],
                "gt_action_label": int(parts[2]),
                "start_act": int(parts[3]),
                "end_act": int(parts[4]),
                "start_frame": int(parts[5]),
                "end_frame": int(parts[6]),
            }
        )
    return rows


def map_path_to_local(subject_root: Path, path_from_txt: str) -> Optional[Path]:
    """
    action_label file uses: subject1/h1/0
    local may be:
      1) <root>/subject1/h1/0
      2) <root>/subject1_ego/h1/0
    """
    p1 = subject_root / path_from_txt
    if p1.exists():
        return p1

    m = re.match(r"^(subject\d+)/(.*)$", path_from_txt)
    if m:
        subj = m.group(1)
        rest = m.group(2)
        p2 = subject_root / f"{subj}_ego" / rest
        if p2.exists():
            return p2

    return None


def sample_frame_indices(start: int, end: int, num_samples: int) -> List[int]:
    """
    Sample exactly num_samples frame indices from [start, end], inclusive.
    Evenly spaced; may repeat indices if segment is shorter than num_samples.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if end < start:
        raise ValueError(f"Invalid segment range: start={start}, end={end}")
    if num_samples == 1:
        return [start]

    span = end - start
    return [start + round(i * span / (num_samples - 1)) for i in range(num_samples)]


def parse_bounding_box_txt(txt_path: Path) -> Dict[str, Any]:
    """
    Expects the format you generated (as in your screenshot):

    Line 1: W H
    Lines 2..6: each line = flag cx cy x1 y1 x2 y2 score
      order: LH, RH, LO, RO, THO
    Line 7: twohand_giou lo_idx ro_idx

    Returns:
      {
        "W": int, "H": int,
        "entities": [
          {"flag": int, "cx": float, "cy": float, "score": float},
          ... 5 entries
        ],
        "twohand_giou": float, "lo_idx": int, "ro_idx": int
      }
    """
    lines = [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]
    if len(lines) < 7:
        raise ValueError(f"Expected >=7 non-empty lines in {txt_path}, got {len(lines)}")

    # Line1: W H
    w_h = lines[0].split()
    if len(w_h) < 2:
        raise ValueError(f"Bad W H line in {txt_path}: {lines[0]}")
    W = int(float(w_h[0]))
    H = int(float(w_h[1]))

    entities = []
    for i in range(1, 6):  # lines[1]..lines[5]
        parts = lines[i].split()
        if len(parts) < 8:
            raise ValueError(f"Bad entity line {i+1} in {txt_path}: {lines[i]}")
        flag = int(float(parts[0]))
        cx = float(parts[1])
        cy = float(parts[2])
        score = float(parts[7])
        entities.append({"flag": flag, "cx": cx, "cy": cy, "score": score})

    # last line: giou lo_idx ro_idx
    tail = lines[6].split()
    if len(tail) < 3:
        raise ValueError(f"Bad tail line in {txt_path}: {lines[6]}")
    twohand_giou = float(tail[0])
    lo_idx = int(float(tail[1]))
    ro_idx = int(float(tail[2]))

    return {"W": W, "H": H, "entities": entities, "twohand_giou": twohand_giou, "lo_idx": lo_idx, "ro_idx": ro_idx}


def format_pair(cx: float, cy: float) -> str:
    # keep similar to your bounding_box file style (2 decimals)
    return f"({cx:.2f}, {cy:.2f})"


def format_pair_sequence(seq: List[Tuple[float, float]]) -> str:
    return "(" + ", ".join(format_pair(x, y) for (x, y) in seq) + ")"


def build_prompt(
    seqs: Dict[str, List[Tuple[float, float]]],
    num_frames: int,
) -> str:
    """
    Builds the EgoHOD-style Hand Object Dynamics prompt (raw pixel centers).
    """
    intro = (
        "You are an action_label detector assistant.\n\n"
        f"You are given Hand Object Dynamics across {num_frames} sampled frames.\n"
        "Each item is a 2D point (w, h) representing the CENTER of a detected bounding box in raw pixels.\n"
        "If an entity is missing in a frame, it is recorded as (0.00, 0.00).\n\n"
        "## action_label options\n"
        f"{format_action_options()}\n\n"
        "## Hand Object Dynamics\n"
    )

    block = (
        f"left hand:{format_pair_sequence(seqs['left hand'])}\n\n"
        f"right hand:{format_pair_sequence(seqs['right hand'])}\n\n"
        f"left hand object:{format_pair_sequence(seqs['left hand object'])}\n\n"
        f"right hand object:{format_pair_sequence(seqs['right hand object'])}\n\n"
        f"two hand object:{format_pair_sequence(seqs['two hand object'])}\n"
    )

    outro = (
        "\nYour task is to determine the most likely action label.\n\n"
        "Instructions:\n"
        "- Focus on how hands and objects move over time.\n"
        "- Use both hands and object-contact trajectories.\n"
        "- Choose exactly one action_label ID from the provided list.\n"
        "- Return only the predicted action_label ID.\n"
        "- Do not output extra text, reasoning, JSON, or formatting.\n\n"
        "Output:\n"
        "[action_label]\n"
    )

    return intro + block + outro


def extract_last_valid_json(text: str) -> Optional[Dict[str, Any]]:
    dec = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, consumed = dec.raw_decode(text[i:])
                if isinstance(obj, dict) and "action_label" in obj:
                    candidates.append(obj)
                i += consumed
                continue
            except Exception:
                pass
        i += 1
    return candidates[-1] if candidates else None


def extract_prediction(text: str) -> Tuple[Optional[int], Optional[float], Optional[str], Optional[Dict[str, Any]]]:
    """
    Preferred model output: plain integer, e.g. 16
    Also supports fallback JSON: {"action_label":16,"confidence":...,"rationale":...}
    """
    text = text.strip()
    if not text:
        return None, None, None, None

    obj = extract_last_valid_json(text)
    if isinstance(obj, dict) and "action_label" in obj:
        try:
            pred_label = int(obj["action_label"])
        except Exception:
            pred_label = None
        pred_conf = obj.get("confidence", None)
        pred_rat = obj.get("rationale", None)
        return pred_label, pred_conf, pred_rat, obj

    m = re.search(r"-?\d+", text)
    if m:
        try:
            val = int(m.group(0))
            return val, None, None, {"action_label": val}
        except Exception:
            pass

    return None, None, None, None


def run_qwen_one(
    prompt: str,
    tok,
    model,
    max_new_tokens: int,
    temperature: float,
    max_input_tokens: Optional[int] = None,
) -> Tuple[str, Optional[int], Optional[float], Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns:
      raw_text, pred_label, pred_conf, pred_rationale, parsed_obj, error
    """
    import torch

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    if max_input_tokens is not None and input_ids.shape[1] > max_input_tokens:
        return "", None, None, None, None, f"prompt_too_long_tokens={input_ids.shape[1]}"

    attention_mask = torch.ones_like(input_ids, device=model.device)

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    gen_only = gen[0][input_ids.shape[1]:]
    txt = tok.decode(gen_only, skip_special_tokens=True).strip()

    pred_label, pred_conf, pred_rat, parsed_obj = extract_prediction(txt)
    return txt, pred_label, pred_conf, pred_rat, parsed_obj, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_root", type=str, required=True,
                    help="Folder that contains subject1_ego/... (your local ego dataset root).")
    ap.add_argument("--action_label", type=str, required=True,
                    help="Path to action label file (e.g., action_train.txt)")
    ap.add_argument("--out_jsonl", type=str, default="h2o_bbox_prompts_preds.jsonl")
    ap.add_argument("--out_csv", type=str, default="h2o_bbox_preds.csv")
    ap.add_argument("--max_prompts", type=int, default=-1,
                    help="Limit number of segments to process. -1 = all")
    ap.add_argument("--preview_n", type=int, default=0,
                    help="Print first N prompts (and parsed answer if --run_qwen).")

    # NOTE: EgoHOD’s template uses 16 points (0..15). Default=16.
    ap.add_argument("--num_sampled_frames", type=int, default=16,
                    help="Sample exactly this many frames from each segment, evenly from start_act to end_act.")

    # Qwen inference options
    ap.add_argument("--run_qwen", action="store_true",
                    help="Actually run the model and store predictions.")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_input_tokens", type=int, default=-1,
                    help="If >0, skip Qwen call when input tokens exceed this number.")
    args = ap.parse_args()

    subject_root = Path(args.subject_root)
    action_label_txt = Path(args.action_label)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = parse_action_label_file(action_label_txt)
    if not rows:
        raise RuntimeError(f"No rows parsed from: {action_label_txt}")

    if args.max_prompts != -1:
        rows = rows[: args.max_prompts]

    tok = model = None
    if args.run_qwen:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

    num_sampled_frames = max(1, args.num_sampled_frames)
    max_input_tokens = None if args.max_input_tokens <= 0 else args.max_input_tokens

    csv_fieldnames = [
        "id", "path", "start_act", "end_act",
        "gt_action_label",
        "pred_action_label", "confidence", "rationale",
        "correct", "error",
    ]

    written = 0
    previews_left = args.preview_n
    parsed_count = 0
    correct_count = 0
    skipped_or_error = 0

    with out_jsonl.open("w", encoding="utf-8") as fjson, out_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=csv_fieldnames)
        writer.writeheader()

        pbar = tqdm(rows, desc="Processing segments", unit="seg")
        for r in pbar:
            local_seq = map_path_to_local(subject_root, r["path"])
            if local_seq is None:
                skipped_or_error += 1
                writer.writerow({
                    "id": r["id"], "path": r["path"],
                    "start_act": r["start_act"], "end_act": r["end_act"],
                    "gt_action_label": r["gt_action_label"],
                    "pred_action_label": "", "confidence": "", "rationale": "",
                    "correct": "", "error": "sequence_not_found",
                })
                continue

            bb_dir = local_seq / "cam4" / "bounding_box"
            if not bb_dir.exists():
                skipped_or_error += 1
                writer.writerow({
                    "id": r["id"], "path": r["path"],
                    "start_act": r["start_act"], "end_act": r["end_act"],
                    "gt_action_label": r["gt_action_label"],
                    "pred_action_label": "", "confidence": "", "rationale": "",
                    "correct": "", "error": "bounding_box_dir_missing",
                })
                continue

            start, end = r["start_act"], r["end_act"]
            sampled_indices = sample_frame_indices(start, end, num_sampled_frames)
            pbar.set_postfix_str(f"{r['path']} [{start}:{end}] frames={len(sampled_indices)}")

            # Prepare sequences of (w,h) for each entity
            seqs: Dict[str, List[Tuple[float, float]]] = {name: [] for name in ENTITY_NAMES}

            try:
                for frame_idx in sampled_indices:
                    fp = bb_dir / f"{frame_idx:06d}.txt"
                    if not fp.exists():
                        raise FileNotFoundError(f"Missing bounding_box file: {fp}")

                    rec = parse_bounding_box_txt(fp)
                    ents = rec["entities"]  # 5 entities in fixed order

                    for name, ent in zip(ENTITY_NAMES, ents):
                        if int(ent["flag"]) == 1:
                            seqs[name].append((float(ent["cx"]), float(ent["cy"])))
                        else:
                            # missing => (0,0) (keeps fixed-length trajectory)
                            seqs[name].append((0.0, 0.0))

            except Exception as e:
                skipped_or_error += 1
                writer.writerow({
                    "id": r["id"], "path": r["path"],
                    "start_act": start, "end_act": end,
                    "gt_action_label": r["gt_action_label"],
                    "pred_action_label": "", "confidence": "", "rationale": "",
                    "correct": "", "error": f"bbox_read_error:{type(e).__name__}",
                })
                continue

            prompt = build_prompt(seqs=seqs, num_frames=len(sampled_indices))

            qwen_raw: Optional[str] = None
            qwen_parsed: Optional[Dict[str, Any]] = None
            pred_label = None
            pred_conf = None
            pred_rat = None
            err = ""

            if args.run_qwen:
                qwen_raw, pred_label, pred_conf, pred_rat, qwen_parsed, run_err = run_qwen_one(
                    prompt=prompt,
                    tok=tok,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    max_input_tokens=max_input_tokens,
                )
                if run_err:
                    err = run_err
                    skipped_or_error += 1

            correct = ""
            if pred_label is not None:
                parsed_count += 1
                is_correct = int(int(pred_label) == int(r["gt_action_label"]))
                correct_count += is_correct
                correct = str(is_correct)

            out_rec = {
                "id": r["id"],
                "path": r["path"],
                "local_path": str(local_seq),
                "start_act": start,
                "end_act": end,
                "gt_action_label": r["gt_action_label"],
                "sampled_frame_indices": sampled_indices,
                "hand_object_dynamics": {k: seqs[k] for k in ENTITY_NAMES},
                "prompt": prompt,
            }
            if args.run_qwen:
                out_rec["qwen_raw_text"] = qwen_raw
                out_rec["qwen_parsed"] = qwen_parsed
                out_rec["error"] = err

            fjson.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            writer.writerow({
                "id": r["id"],
                "path": r["path"],
                "start_act": start,
                "end_act": end,
                "gt_action_label": r["gt_action_label"],
                "pred_action_label": "" if pred_label is None else pred_label,
                "confidence": "" if pred_conf is None else pred_conf,
                "rationale": "" if pred_rat is None else pred_rat,
                "correct": correct,
                "error": err,
            })

            written += 1

            if previews_left > 0:
                print("\n" + "=" * 80)
                print(f"PROMPT id={r['id']} path={r['path']} frames[{start}:{end}] gt={r['gt_action_label']}")
                print(f"SAMPLED FRAME INDICES: {sampled_indices}")
                print("=" * 80)
                print(prompt)
                if args.run_qwen:
                    print("\n--- QWEN RAW ---\n", qwen_raw)
                    print("\n--- QWEN PARSED ---\n", qwen_parsed)
                    if err:
                        print("\n--- ERROR ---\n", err)
                previews_left -= 1

        pbar.close()

    print(f"[OK] Wrote JSONL: {out_jsonl.resolve()}")
    print(f"[OK] Wrote CSV:   {out_csv.resolve()}")

    if args.run_qwen:
        parse_rate = (parsed_count / written) if written > 0 else 0.0
        acc_on_parsed = (correct_count / parsed_count) if parsed_count > 0 else 0.0
        print(f"[STATS] segments_written={written}")
        print(f"[STATS] parsed={parsed_count} (parse_rate={parse_rate:.3f})")
        print(f"[STATS] accuracy_on_parsed={acc_on_parsed:.3f}")
        if skipped_or_error > 0:
            print(f"[STATS] skipped_or_error={skipped_or_error} (see CSV 'error' column)")


if __name__ == "__main__":
    main()