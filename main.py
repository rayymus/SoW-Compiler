import json
import os
import re
import time

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from groq import Groq


SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly"
]

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-safeguard-20b")

SOURCE = ""
SUBJECTS = ["Mathematics H2", "Physics H2", "Computing H2", "Economics H2"]
ACAD_LEVEL = "JC2 (2025 Cohort)"

CHUNK_CHAR_LIMIT = 8000
SLEEP_SECONDS = 2.0


def extract_id(url: str) -> str:
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        raise ValueError(f"Could not parse ID from URL: {url}")
    return match.group(1)


def doc_to_text(doc: dict) -> str:
    lines = []
    for content in doc.get("body", {}).get("content", []):
        if "paragraph" in content:
            for elem in content["paragraph"].get("elements", []):
                text = elem.get("textRun", {}).get("content", "").strip()
                if text:
                    lines.append(text)
        elif "table" in content:
            for row in content["table"].get("tableRows", []):
                cells = []
                for cell in row.get("tableCells", []):
                    cell_text = []
                    for cell_content in cell.get("content", []):
                        if "paragraph" in cell_content:
                            for elem in cell_content["paragraph"].get("elements", []):
                                text = elem.get("textRun", {}).get("content", "").strip()
                                if text:
                                    cell_text.append(text)
                    cells.append(" ".join(cell_text))
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def chunk_text(text: str, max_chars: int) -> list[str]:
    chunks = []
    current = []
    size = 0
    for line in text.splitlines():
        line = f"{line}\n"
        if size + len(line) > max_chars and current:
            chunks.append("".join(current).strip())
            current = [line]
            size = len(line)
        else:
            current.append(line)
            size += len(line)

    if current:
        chunks.append("".join(current).strip())
    return chunks


def extract_first_json_array(raw: str) -> str | None:
    start = raw.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return None


def clean_json_like(raw: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", raw)


def parse_json_array(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            items = data.get("items")
            if isinstance(items, list):
                return items
    except json.JSONDecodeError:
        pass

    snippet = extract_first_json_array(raw)
    if snippet:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            cleaned = clean_json_like(snippet)
            return json.loads(cleaned)

    raise ValueError("Model did not return valid JSON.")


def dedupe_items(items: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for item in items:
        tasks = item.get("tasks")
        key = (
            item.get("subject"),
            item.get("term"),
            item.get("week"),
            json.dumps(tasks, sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def extract_chunk(client: Groq, subject: str, source_text: str) -> list[dict]:
    system = (
        f"Extract a Scheme of Work as JSON, under {ACAD_LEVEL}. "
        "Return ONLY a JSON array of objects with keys: "
        "subject, term, week, tasks. "
        "tasks must ALWAYS be a JSON array of strings; if there is only one task, use a single-item list. "
        "Use double quotes for all keys and strings. Do not use code fences. "
        "Use only the chunk content. If there are no items for this subject, return []. "
        "term and week should be numeric and note that Term 1 Week 1 starts on 5 January and each term has 10 weeks. "
        "There is a week long holiday AFTER the end of term 1 and term 3 and there is a 4 week-long holiday AFTER the end of term 2 "
        "so for each subject's scheme of work, it should start at term 1 week 2."
    )
    user = f"Subject: {subject}\n\nChunk:\n{source_text}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content
    return parse_json_array(raw)


def extract_sow(client: Groq, subject: str, source_text: str) -> list[dict]:
    items = []
    for i, chunk in enumerate(chunk_text(source_text, CHUNK_CHAR_LIMIT)):
        if i > 0 and SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)
        chunk_items = extract_chunk(client, subject, chunk)
        for item in chunk_items:
            item_subject = str(item.get("subject", "")).strip().lower()
            if item_subject and item_subject != subject.strip().lower():
                continue
            items.append(item)
    return dedupe_items(items)


def extract_subject_sections(text: str, subjects: list[str]) -> dict[str, str]:
    subject_map = {s.lower(): s for s in subjects}
    sections = {s: [] for s in subjects}
    current = None
    for line in text.splitlines():
        line_norm = line.strip().lower()
        matched = None
        for subject_norm, subject in subject_map.items():
            if subject_norm and subject_norm in line_norm:
                matched = subject
                break
        if matched:
            current = matched
        if current:
            sections[current].append(line)
    return {subject: "\n".join(lines).strip() for subject, lines in sections.items()}


def main() -> None:
    if not SOURCE:
        raise RuntimeError("Add the link to SOURCE")

    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    docs_service = build("docs", "v1", credentials=creds)

    doc_id = extract_id(SOURCE)
    doc = docs_service.documents().get(documentId=doc_id).execute()
    text = doc_to_text(doc)

    sections = extract_subject_sections(text, SUBJECTS)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    output = {}
    for subject in SUBJECTS:
        subject_text = sections.get(subject) or text
        output[subject] = extract_sow(client, subject, subject_text)

    print(json.dumps(output, indent=2))

    #  Output
    with open("output.md", "w") as f:
        for term in range(1, 4+1):
            for week in range(1, 10+1):
                f.write(f"## Term {term} Week {week}\n")
                for subject in SUBJECTS:
                    for item in output[subject]:
                        if (tasks := item["tasks"]) and item["term"] == term and item["week"] == week:
                            for task in tasks:
                                f.write(f"{item["subject"]} - {task}\n")

                f.write("\n")


if __name__ == "__main__":
    main()
