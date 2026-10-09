"""Backup voice demo: the same call scenario for 3 contrasting sellers, fully in audio.

Scenario: calm pick-up -> aggression -> calming down -> calm -> confusion -> language
switch -> interest -> agreement. The Behaviour Analyst and Persona Director decide every
persona switch live from the audio.

    python -m vani_persona.ps04.simulate [--sellers new_receptive time_pressed english_first]
        -> artifacts/ps04/demo_audio/<seller>.wav        (whole call: bot + simulated seller)
        -> artifacts/ps04/demo_audio/<seller>_switch_log.json
        -> artifacts/ps04/demo_audio/README.md          (transcripts + switch logs)

The seller's lines are scripted, spoken by a second Sarvam voice and sent through Saaras
exactly like a live caller. Everything the bot does (signals, persona changes, replies) is
decided live from that audio, so the switches are real reactions, not scripted ones.
"""

from __future__ import annotations

import argparse
import json

from vani_persona import REPO_ROOT
from vani_persona.ps04 import sarvam
from vani_persona.ps04.live import CallSession, join_wavs
from vani_persona.ps04.live_personas import PERSONAS

OUT = REPO_ROOT / "artifacts" / "ps04" / "demo_audio"
SELLER_VOICE = "aditya"

# Same eight beats for every seller: calm pick-up, aggression, calming down, calm,
# confusion, language switch, interest, agreement.
SCENARIOS = {
    "new_receptive": [
        ("hi-IN", "हाँ जी, बोलिए, कौन?"),
        ("hi-IN", "अरे यार, कितनी बार call करोगे? दिमाग खराब कर दिया है।"),
        ("hi-IN", "अच्छा ठीक है, बोलो क्या बात है।"),
        ("hi-IN", "हाँ, सुन रहा हूँ, आगे बताइए।"),
        ("hi-IN", "मतलब? कौन सी meeting? समझ नहीं आया।"),
        ("od-IN", "ମୁଁ ଓଡ଼ିଆରେ କଥା ହେବି, ଆପଣ ଓଡ଼ିଆରେ କୁହନ୍ତୁ।"),
        ("od-IN", "ଠିକ ଅଛି, ମିଟିଂ କେବେ ହୋଇପାରିବ? କେତେ ବେଳେ ଆସିବେ?"),
        ("od-IN", "କାଲି ବାରଟା ବେଳେ ଠିକ ଅଛି।"),
    ],
    "time_pressed": [
        ("hi-IN", "हाँ बोलो।"),
        ("hi-IN", "अरे यार, कितनी बार call करोगे? दिमाग खराब कर दिया है।"),
        ("hi-IN", "अच्छा ठीक है, बोलो क्या बात है।"),
        ("hi-IN", "हाँ, सुन रहा हूँ, आगे बताइए।"),
        ("hi-IN", "मतलब? कौन सी meeting? समझ नहीं आया।"),
        ("en-IN", "Okay, please tell me in English, I will understand better."),
        ("en-IN", "Fine, that sounds useful. When can your executive come?"),
        ("en-IN", "Tomorrow at twelve works for me."),
    ],
    "english_first": [
        ("en-IN", "Hello, yes, who is this?"),
        ("en-IN", "Why do you people keep calling me again and again? This is irritating."),
        ("en-IN", "Okay fine, tell me what it is."),
        ("en-IN", "Yes, I am listening, go on."),
        ("en-IN", "Sorry, I didn't understand. What meeting is this?"),
        ("ta-IN", "எனக்கு தமிழில் பேசுங்கள், ஆங்கிலம் சரியாக புரியவில்லை."),
        ("ta-IN", "சரி, நல்லது. மீட்டிங் எப்போது வைக்கலாம்? எத்தனை மணிக்கு வருவீர்கள்?"),
        ("ta-IN", "நாளை பன்னிரண்டு மணிக்கு சரி."),
    ],
}


def run(key: str, spec: dict) -> dict:
    s = CallSession(spec)
    clips, lines = [], []
    r = s.open()
    clips.append(r["audio"])
    lines.append(f"*Starts as **{r['persona']['name']}**: {r['start_reason']}*")
    lines.append(f"**VANI** [{r['persona']['name']}] ({r['settings']['speaker']}, pace {r['settings']['pace']}, "
                 f"{r['settings']['language']}): {r['reply']}")
    for lang, line in SCENARIOS[key]:
        if s.ended:
            break
        seller_wav, _ = sarvam.synthesize(line, lang, SELLER_VOICE, 1.05, 0.6, 16000)
        clips.append(seller_wav)
        r = s.seller_audio(seller_wav)
        if r.get("wait"):  # scripted lines are complete; no one is going to carry on
            r = s.flush()
        a = r.get("analysis") or {}
        lines.append(f"**Seller** (Saaras: {r.get('language')}): {r.get('transcript')}  \n"
                     f"<sub>Analyst: {a.get('state')} {a.get('intensity')} {a.get('trend')} ({a.get('source')}) · "
                     f"Director: {r.get('decision', {}).get('action')}</sub>")
        for e in r.get("switches", []):
            ch = ", ".join(f"{k}: {v[0]} → {v[1]}" for k, v in e["change"].items() if k != "focus") or "—"
            head = (f"{e['from_name']} → {e['to_name']}" if e["type"] == "persona" else e["type"])
            lines.append(f"> ⚡ **{head}** ({e['signal']}, {e['detector']}): {ch}. *{e['reason']}*")
        if r.get("audio"):
            clips.append(r["audio"])
            st = r["settings"]
            lines.append(f"**VANI** [{r['persona']['name']}] ({st['speaker']}, pace {st['pace']}, {st['language']}): "
                         f"{r['reply']}  <sub>{r['timings'].get('reply_total_ms')} ms</sub>")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{key}.wav").write_bytes(join_wavs(clips, gap_ms=400))
    log = json.loads(s.save().read_text(encoding="utf-8"))
    (OUT / f"{key}_switch_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"key": key, "summary": spec["summary"], "lines": lines, "outcome": s.outcome,
            "n_switches": len(s.switch_log),
            "timeline": " → ".join(PERSONAS[t["persona"]].name for t in s.timeline)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sellers", nargs="+", default=list(SCENARIOS))
    args = ap.parse_args()
    demo = json.loads((REPO_ROOT / "artifacts" / "ps04" / "demo_sellers.json").read_text(encoding="utf-8"))
    md = ["# Voice demo (simulated seller voice, live bot)", "",
          "Seller lines are scripted and spoken by a Sarvam voice; every bot decision is made live from the audio.", ""]
    for key in args.sellers:
        res = run(key, demo[key])
        md += [f"## {key.replace('_', ' ')}", "", f"*{res['summary']}* · {res['n_switches']} switches · "
               f"outcome: {res['outcome'] or 'none'} · audio: `{key}.wav`", "",
               f"**Persona timeline:** {res['timeline']}", ""] + [x + "\n" for x in res["lines"]]
        print(f"{key}: {res['timeline']} | {res['n_switches']} switches, outcome {res['outcome']}")
    (OUT / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
