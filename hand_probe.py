"""Live HandReader probe: score every hand slot against thresholds."""
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

from clash_jev.hand import HandReader, EXTRA_CARD_SHAPES, _slot_box, _shape, _detail, _reference, _is_lit, _is_empty
from clash_jev.hand import _HAND_MATCH, _DETAIL_MATCH, _LOWER_HALF_MATCH, _MATCH_MARGIN, _LOWER_HALF_MARGIN

SERIAL = "RFCX91J8LSD"


def grab() -> "cv2.typing.MatLike":
    out = Path(tempfile.gettempdir()) / "opencode" / "hand_probe.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = f'adb -s {SERIAL} exec-out screencap -p > "{out}"'
    subprocess.run(cmd, shell=True, check=True)
    frame = cv2.imread(str(out))
    if frame is None:
        raise SystemExit("screencap failed")
    return frame


def main() -> None:
    frame = grab()
    reference = _reference(frame)
    bank = HandReader().bank
    extra = "yes" if EXTRA_CARD_SHAPES.exists() else "no"
    print(f"frame={frame.shape[1]}x{frame.shape[0]} bank={len(bank.names)} extra_shapes={extra}")
    print(f"thresholds detail={_DETAIL_MATCH} hand={_HAND_MATCH} lower={_LOWER_HALF_MATCH} margin={_MATCH_MARGIN}/{_LOWER_HALF_MARGIN}")

    reader = HandReader()
    hand = reader.read(frame)
    print("reader:", ", ".join(f"s{c.slot}:{c.name}{'*' if c.ready else ''}" for c in hand))
    print("next:", reader.next_card)

    print("\nper-slot scores:")
    for slot in range(4):
        if _is_empty(reference, slot):
            print(f"  slot {slot}: EMPTY")
            continue
        d_name, d_score, d_lead = bank.match(_detail(reference, _slot_box(slot)), detail=True)
        s_name, s_score, s_lead = bank.match(_shape(reference, _slot_box(slot)))
        l_name, l_score, l_lead = bank.match(_shape(reference, _slot_box(slot)), lower_half=True)
        lit = _is_lit(reference, slot)
        ok_d = d_score >= _DETAIL_MATCH and d_lead >= _MATCH_MARGIN
        ok_s = s_score >= _HAND_MATCH and s_lead >= _MATCH_MARGIN
        ok_l = l_score >= _LOWER_HALF_MATCH and l_lead >= _LOWER_HALF_MARGIN
        print(
            f"  slot {slot} lit={int(lit)} detail={d_name}:{d_score:.3f}/{d_lead:.3f}{'OK' if ok_d else ''} "
            f"shape={s_name}:{s_score:.3f}/{s_lead:.3f}{'OK' if ok_s else ''} "
            f"lower={l_name}:{l_score:.3f}/{l_lead:.3f}{'OK' if ok_l else ''}"
        )
        shape_vec = _shape(reference, _slot_box(slot))
        scores = bank.matrix @ shape_vec
        by_name: dict[str, float] = {}
        for row, name in enumerate(bank.row_names):
            if bank.deck is not None and name not in bank.deck:
                continue
            value = float(scores[row])
            if name not in by_name or value > by_name[name]:
                by_name[name] = value
        runner = sorted(by_name.items(), key=lambda t: t[1], reverse=True)[:3]
        print("         top3:", ", ".join(f"{n}={sc:.3f}" for n, sc in runner))


if __name__ == "__main__":
    sys.exit(main())
