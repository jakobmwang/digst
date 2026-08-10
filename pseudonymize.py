"""Pseudonymize raw.jsonl using Qwen3 via Ollama for entity extraction.

Resumable pipeline: tracks progress in progress.json, maintains consistent
pseudonym mappings in pseudonym_map.json.

Usage:
    python export/pseudonymize.py [--limit N]
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:30b-a3b-instruct-16k"

BASE_DIR = Path(__file__).parent
RAW_FILE = BASE_DIR / "raw.jsonl"
CLEAN_FILE = BASE_DIR / "clean.jsonl"
MAP_FILE = BASE_DIR / "pseudonym_map.json"
PROGRESS_FILE = BASE_DIR / "progress.json"

# ── Danish name pools ────────────────────────────────────────────────────────

MALE_FIRST_NAMES = [
    "Anders", "Bjørn", "Erik", "Gustav", "Jonas", "Lars", "Niels", "Peter",
    "Søren", "Ulrik", "William", "Aksel", "Carl", "Emil", "Georg", "Iver",
    "Knud", "Magnus", "Oscar", "Rasmus", "Thomas", "Viktor", "Bent", "Dan",
    "Finn", "Henrik", "Jens", "Leif", "Nis", "Poul", "Svend", "Uffe",
    "Aage", "Claus", "Egon", "Gunnar", "Ivan", "Kurt", "Mogens", "Olaf",
    "Stig", "Valdemar", "Christian", "Frederik", "Hans", "Jakob", "Mikkel",
    "Ole", "Torben", "Viggo",
]

FEMALE_FIRST_NAMES = [
    "Anna", "Camilla", "Dorte", "Freja", "Hanne", "Ida", "Karen", "Mette",
    "Olivia", "Rita", "Tine", "Vibeke", "Xenia", "Yrsa", "Zara", "Bodil",
    "Dagny", "Frida", "Helga", "Julie", "Lise", "Nora", "Pia", "Sigrid",
    "Ulla", "Agnete", "Cecilie", "Else", "Gitte", "Inge", "Kirsten", "Margit",
    "Oda", "Rosa", "Tove", "Vera", "Birgit", "Dina", "Flora", "Hilda",
    "Johanne", "Lone", "Naja", "Asta", "Ruth", "Thyra", "Bente", "Grethe",
    "Lene", "Sonja",
]


LAST_NAMES = [
    "Andersen", "Bak", "Christensen", "Dahl", "Eriksen", "Frederiksen",
    "Grøn", "Holm", "Iversen", "Jensen", "Kirkegaard", "Larsen", "Madsen",
    "Nielsen", "Olsen", "Pedersen", "Rasmussen", "Sørensen", "Thomsen",
    "Vestergaard", "Wind", "Aaberg", "Bjerre", "Clemmensen", "Damgaard",
    "Engel", "Friis", "Gade", "Hjorth", "Ipsen", "Juul", "Knudsen",
    "Lund", "Munk", "Nørgaard", "Overgaard", "Pape", "Ravn", "Skjødt",
    "Trane", "Ulrich", "Villadsen", "Wahl", "Yde", "Zilmer", "Bach",
    "Dall", "Falk", "Gram", "Hald", "Jakobsen", "Krogh", "Lindberg",
    "Mølgaard", "Nissen", "Odgaard", "Pilgaard", "Rode", "Storm", "Tang",
]

STREET_NAMES = [
    "Enebærvej", "Kirsebærvænget", "Birkevej", "Ellevej", "Granvænget",
    "Hyldestien", "Kastanjevej", "Lindestien", "Mosevej", "Nørregade",
    "Østervej", "Pilevænget", "Rosenvej", "Skovstien", "Tjørnevej",
    "Uldumvej", "Viborgvej", "Ågade", "Bøgevej", "Dalvej",
    "Egevej", "Fuglevej", "Havnegade", "Industrivej", "Jupitervej",
    "Kløvervej", "Lyngvej", "Møllevej", "Nattergalevej", "Odinsvej",
    "Præstevænget", "Rugvej", "Solsikkevej", "Tulipanvej", "Uranusvej",
    "Valmuestien", "Åkandevej", "Blåbærvej", "Cedervej", "Drosselvej",
]

# ── Pseudonym generation ────────────────────────────────────────────────────

_rng = random.Random(42)  # Deterministic for reproducibility within a session
_used_names: set[str] = set()



def _generate_person_name(gender: str = "u") -> str:
    """Generate a unique Danish pseudonym, matching gender."""
    if gender == "m":
        pool = MALE_FIRST_NAMES
    elif gender == "f":
        pool = FEMALE_FIRST_NAMES
    else:
        pool = MALE_FIRST_NAMES + FEMALE_FIRST_NAMES

    for _ in range(1000):
        name = f"{_rng.choice(pool)} {_rng.choice(LAST_NAMES)}"
        if name not in _used_names:
            _used_names.add(name)
            return name
    # Fallback: add number suffix
    name = f"{_rng.choice(pool)} {_rng.choice(LAST_NAMES)}-{_rng.randint(1,999)}"
    _used_names.add(name)
    return name


def _generate_address() -> str:
    """Generate a fake Danish address."""
    street = _rng.choice(STREET_NAMES)
    number = _rng.randint(1, 200)
    suffix = _rng.choice(["", "", "", "a", "b", "c"])
    return f"{street} {number}{suffix}"


def _generate_email(pseudonym: str) -> str:
    """Generate email based on person pseudonym."""
    parts = pseudonym.lower().replace("ø", "oe").replace("æ", "ae").replace("å", "aa")
    parts = parts.split()
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[-1]}@example.dk"
    return f"{parts[0]}@example.dk"


def _generate_phone() -> str:
    """Generate a random 8-digit Danish phone number."""
    return str(_rng.randint(20000000, 99999999))


# ── Entity normalization ─────────────────────────────────────────────────────

def normalize_entity(text: str) -> str:
    """Normalize entity text for consistent map lookup."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text


# ── Pseudonym map management ────────────────────────────────────────────────

def load_map() -> dict:
    if MAP_FILE.exists():
        return json.loads(MAP_FILE.read_text(encoding="utf-8"))
    return {}


def save_map(pmap: dict):
    MAP_FILE.write_text(json.dumps(pmap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_pseudonym(pmap: dict, text: str, entity_type: str, gender: str = "u") -> str:
    """Get or create a consistent pseudonym for an entity."""
    norm = normalize_entity(text)
    key = f"{entity_type}::{norm}"
    if key in pmap:
        return pmap[key]

    if entity_type == "person_name":
        pseudo = _generate_person_name(gender)
    elif entity_type == "address":
        pseudo = _generate_address()
    elif entity_type == "email":
        # Deterministic: derive from hash of original email
        import hashlib
        h = int(hashlib.sha256(norm.encode()).hexdigest(), 16)
        first = MALE_FIRST_NAMES[h % len(MALE_FIRST_NAMES)] if h % 2 == 0 else FEMALE_FIRST_NAMES[h % len(FEMALE_FIRST_NAMES)]
        last = LAST_NAMES[(h >> 8) % len(LAST_NAMES)]
        pseudo_name = f"{first} {last}"
        pseudo = _generate_email(pseudo_name)
    elif entity_type == "phone":
        pseudo = _generate_phone()
    else:
        pseudo = f"[REDACTED-{entity_type}]"

    pmap[key] = pseudo
    return pseudo


# ── Progress management ─────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"last_processed_line": 0, "total_lines": 0}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")


# ── Ollama entity extraction ────────────────────────────────────────────────

SYSTEM_PROMPT = """Du er en dansk NER-model. Du modtager tekst fra kommunale dagsordener og referater.
Returnér KUN valid JSON – ingen forklaring, ingen markdown-blokke."""

USER_PROMPT_TEMPLATE = """Find alle personnavne, personrelaterede adresser, emails og telefonnumre i følgende tekst.

Regler:
- Personnavne: Fulde navne (fornavn + efternavn). Inkludér politikere, embedsmænd, borgere. Undlad titler (borgmester, direktør) men tag navnet.
- Adresser: KUN adresser der er knyttet til en person (bopæl, ansøgers adresse, borgers adresse). F.eks. "borger Kim Hansen, Grønnegade 12" → tag "Grønnegade 12". Adresser der er emnet for en politisk beslutning (byggeri, renovering, lokalplan, nedrivning) skal IKKE tagges – de er ikke persondata.
- Emails: Alle email-adresser.
- Telefonnumre: Danske telefonnumre (8 cifre, evt. med +45).

IKKE entities:
- Adresser der er emnet for en sag (f.eks. "Grønnegade 12 skal renoveres", "lokalplan for Vestergade 45")
- Kommunenavne, bydelsnavne, stednavne (Aarhus, Brabrand, Gellerup)
- Institutioner (Aarhus Kommune, Børn og Unge, Teknik og Miljø)
- Udvalgsnavne (Byrådet, Teknisk Udvalg)
- Partinavne (Socialdemokratiet, Venstre)
- Bygninger/lokaler (Rådhuset, Byrådssalen, Dokk1)
- Lovhenvisninger (§ 14, stk. 2)
- Sagsnumre

Returnér JSON i dette format:
{{"entities": [{{"text": "eksakt tekst fra input", "type": "person_name|address|email|phone", "gender": "m|f"}}]}}

gender-feltet er kun relevant for person_name (m=mand, f=kvinde). Udelad for andre typer.

Hvis ingen entities findes, returnér: {{"entities": []}}

TEKST:
{text}"""


def extract_entities(text: str, max_retries: int = 3) -> list[dict]:
    """Call Qwen3 via Ollama to extract entities from text."""
    # Truncate very long texts to avoid overwhelming the model
    if len(text) > 12000:
        text = text[:12000]

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 2048,
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]

            # Strip thinking tags if present (Qwen3 sometimes wraps in <think>...</think>)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Try to extract JSON from the response
            # Sometimes the model wraps in ```json ... ```
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            # Sometimes there's text before/after the JSON
            brace_start = content.find("{")
            brace_end = content.rfind("}") + 1
            if brace_start >= 0 and brace_end > brace_start:
                content = content[brace_start:brace_end]

            result = json.loads(content)
            entities = result.get("entities", [])

            # Validate entity format
            valid = []
            for e in entities:
                if isinstance(e, dict) and "text" in e and "type" in e:
                    if e["type"] in ("person_name", "address", "email", "phone"):
                        valid.append(e)
            return valid

        except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt+1}/{max_retries}: {exc}", file=sys.stderr)
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed after {max_retries} attempts: {exc}", file=sys.stderr)
                return []

    return []


# ── Text replacement ────────────────────────────────────────────────────────

def _make_pattern(original: str, entity_type: str) -> re.Pattern:
    """Build a type-appropriate regex pattern for an entity."""
    if entity_type == "person_name" or entity_type == "address":
        # Whitespace-tolerant: "Hans Jensen" matches "Hans  Jensen" or "Hans\nJensen"
        tokens = original.split()
        escaped = [re.escape(t) for t in tokens]
        inner = r"\s+".join(escaped)
        return re.compile(r"(?<!\w)" + inner + r"(?!\w)")
    elif entity_type == "email":
        return re.compile(r"(?<![\w.+\-])" + re.escape(original) + r"(?![\w.+\-])")
    elif entity_type == "phone":
        return re.compile(r"(?<!\d)" + re.escape(original) + r"(?!\d)")
    else:
        return re.compile(re.escape(original))


def apply_pseudonyms(text: str, entities: list[dict], pmap: dict) -> str:
    """Replace entities in text with their pseudonyms using type-aware boundary matching."""
    if not isinstance(text, str):
        return text

    # Phase 1: Apply entities found by the model in this call
    if entities:
        sorted_entities = sorted(entities, key=lambda e: len(e["text"]), reverse=True)
        for entity in sorted_entities:
            original = entity["text"]
            etype = entity["type"]
            pattern = _make_pattern(original, etype)
            if pattern.search(text):
                gender = entity.get("gender", "u")
                pseudo = get_pseudonym(pmap, original, etype, gender=gender)
                text = pattern.sub(pseudo, text)

    # Phase 2: Dictionary pass — match all previously seen originals from the map
    # Sort by original length descending to avoid partial replacements
    # Skip entries where the original is itself a known pseudonym (prevent double-replace)
    pseudo_values = set(pmap.values())
    known = []
    for key, pseudo in pmap.items():
        etype, _, original = key.partition("::")
        if original and original not in pseudo_values:
            known.append((original, pseudo, etype))
    known.sort(key=lambda x: len(x[0]), reverse=True)

    for original, pseudo, etype in known:
        pattern = _make_pattern(original, etype)
        if pattern.search(text):
            text = pattern.sub(pseudo, text)

    return text


# ── Main pipeline ────────────────────────────────────────────────────────────

def count_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Pseudonymize raw.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Process only N lines (0=all)")
    args = parser.parse_args()

    if not RAW_FILE.exists():
        print(f"Error: {RAW_FILE} not found. Run export_raw.py first.", file=sys.stderr)
        sys.exit(1)

    # Load state
    pmap = load_map()
    progress = load_progress()
    total_lines = count_lines(RAW_FILE)
    progress["total_lines"] = total_lines
    start_line = progress["last_processed_line"]

    # Pre-populate used names from existing map
    for key, val in pmap.items():
        etype = key.split("::")[0]
        if etype == "person_name":
            _used_names.add(val)

    print(f"Total lines: {total_lines}")
    print(f"Resuming from line: {start_line}")
    if args.limit:
        print(f"Limit: {args.limit} lines")

    # Open output file in append mode (resume-safe)
    mode = "a" if start_line > 0 else "w"
    processed = 0
    errors = 0
    t0 = time.time()

    with open(RAW_FILE, "r", encoding="utf-8") as fin, \
         open(CLEAN_FILE, mode, encoding="utf-8") as fout:

        for line_num, line in enumerate(fin):
            if line_num < start_line:
                continue

            if args.limit and processed >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Skipping invalid JSON at line {line_num}", file=sys.stderr)
                errors += 1
                continue

            # Build text for entity extraction
            title = record.get("title", "") or ""
            content = record.get("content_md", "") or ""
            sted = record.get("sted", "") or ""
            combined_text = f"{title}\n\n{content}".strip()

            # Extract entities
            entities = []
            if combined_text:
                entities = extract_entities(combined_text)

            # Apply pseudonyms to text fields (always run — dictionary pass catches known names)
            record["title"] = apply_pseudonyms(title, entities, pmap)
            record["content_md"] = apply_pseudonyms(content, entities, pmap)
            record["sted"] = apply_pseudonyms(sted, entities, pmap)

            # Write cleaned record
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            progress["last_processed_line"] = line_num + 1
            save_progress(progress)

            # Save map periodically
            if processed % 10 == 0:
                save_map(pmap)

            # Progress reporting
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_lines - line_num - 1) / rate if rate > 0 else 0
            entity_count = len(entities)

            if processed % 10 == 0 or entity_count > 0:
                print(
                    f"  [{line_num+1}/{total_lines}] "
                    f"{processed} done, {entity_count} entities, "
                    f"{rate:.1f} lines/s, ETA {eta/60:.0f}min",
                    file=sys.stderr,
                )

    # Final save
    save_map(pmap)
    save_progress(progress)

    elapsed = time.time() - t0
    print(f"\nPass 1 done. Processed {processed} lines in {elapsed:.0f}s ({errors} errors)")
    print(f"Pseudonym map has {len(pmap)} entries")

    # Pass 2: re-apply complete map to catch names discovered after a line was processed
    print("Pass 2: applying complete pseudonym map to all lines...")
    tmp_file = CLEAN_FILE.with_suffix(".tmp")
    patched = 0
    with open(CLEAN_FILE, "r", encoding="utf-8") as fin, \
         open(tmp_file, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            for field in ("title", "content_md", "sted"):
                val = record.get(field)
                if isinstance(val, str):
                    replaced = apply_pseudonyms(val, [], pmap)
                    if replaced != val:
                        patched += 1
                    record[field] = replaced
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_file.replace(CLEAN_FILE)
    print(f"Pass 2 done. Patched {patched} additional fields.")
    print(f"Output: {CLEAN_FILE}")


if __name__ == "__main__":
    main()
