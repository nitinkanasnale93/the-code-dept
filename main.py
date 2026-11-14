from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import os
import json
import re
import time

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Set it in CMD first.")

client = genai.Client(api_key=API_KEY, http_options={"timeout": 30000})
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


def guess_file_info(language):
    names = {
        "python": (".py", "script.py"),
        "javascript": (".js", "script.js"),
        "html": (".html", "index.html"),
        "css": (".css", "style.css"),
        "java": (".java", "Main.java"),
        "c++": (".cpp", "main.cpp"),
        "c#": (".cs", "Program.cs"),
        "go": (".go", "main.go"),
        "ruby": (".rb", "script.rb"),
        "php": (".php", "index.php"),
        "swift": (".swift", "main.swift"),
        "kotlin": (".kt", "main.kt"),
        "typescript": (".ts", "script.ts"),
        "json": (".json", "data.json"),
        "text": (".txt", "output.txt"),
        "markdown": (".md", "README.md"),
    }
    return names.get(str(language).lower(), (".txt", "untitled.txt"))


def safe_error(error):
    message = str(error)
    if API_KEY:
        message = message.replace(API_KEY, "[REDACTED]")
    return re.sub(r"AIza[A-Za-z0-9_-]{30,}", "[REDACTED]", message)


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(text[start:])
        return result


def normalize_result(result):
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    files = result.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Gemini did not return a nonempty files array")
    details = result.get("overall_details")
    if not isinstance(details, dict):
        details = {}

    total_lines = 0
    extensions = set()
    languages = []
    for file in files:
        if not isinstance(file, dict):
            raise ValueError("Invalid file object")
        code = file.get("code", "")
        if not isinstance(code, str):
            raise ValueError("File code must be text")
        language = str(file.get("language") or "text")
        extension, default_name = guess_file_info(language)
        filename = file.get("suggested_filename")
        if not isinstance(filename, str) or not filename.strip():
            filename = default_name
        filename = os.path.basename(filename.replace("\\", "/")) or default_name
        file["code"] = code
        file["language"] = language
        file["suggested_filename"] = filename
        file["lines"] = len(code.splitlines()) if code else 0
        total_lines += file["lines"]
        extensions.add(os.path.splitext(filename)[1] or extension)
        languages.append(language)

    details["lines"] = total_lines
    details["file_extension"] = "Mixed" if len(extensions) > 1 else next(iter(extensions))
    details.setdefault("language", ", ".join(dict.fromkeys(languages)))
    for field in (
        "complexity", "space_complexity", "purpose", "input_output_examples",
        "usage_instructions", "error_handling_details", "assumptions", "target_environment"
    ):
        value = details.get(field)
        if field == "input_output_examples" and isinstance(value, (dict, list)):
            details[field] = json.dumps(value, ensure_ascii=False, indent=2)
        elif not isinstance(value, str):
            details[field] = "Not specified"
    for field in ("complexity", "space_complexity"):
        value = details[field].strip()
        if value in ("1", "O(1)"):
            details[field] = "O(1)"
        # Do not guess other complexities; they depend on the generated algorithm.
    if not isinstance(details.get("dependencies"), list):
        details["dependencies"] = []
    if not isinstance(details.get("num_functions_classes"), int):
        details["num_functions_classes"] = 0
    if not isinstance(details.get("best_practices_followed"), bool):
        details["best_practices_followed"] = False
    result["overall_details"] = details
    return result

@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/generate-code", methods=["POST"])
def generate_code():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("prompt"), str) or not data["prompt"].strip():
        return jsonify({"error": "Send JSON with a nonempty 'prompt'."}), 400

    user_request = data["prompt"].strip()
    prompt = f'''You are an experienced software developer. Generate complete, runnable code for the user request.
Return exactly ONE valid JSON object and NOTHING ELSE. Do not wrap it in Markdown fences.
The JSON must follow this structure:
{{
  "files": [{{"code": "full source code", "language": "java", "suggested_filename": "Main.java"}}],
  "overall_details": {{
    "language": "java", "lines": 0, "complexity": "O(1)",
    "space_complexity": "O(1)", "purpose": "What this code does",
    "dependencies": [], "file_extension": ".java",
    "input_output_examples": "Example input and output",
    "usage_instructions": "How to run",
    "num_functions_classes": 1, "error_handling_details": "Error handling",
    "assumptions": "Assumptions", "best_practices_followed": true,
    "target_environment": "Java JDK"
  }}
}}
Rules:
- Include as many files as the project needs (for example index.html, style.css, script.js).
- Match each filename to its language. For Java, public class name must match the .java filename.
- In the JSON output, encode line breaks inside code strings as escaped newline characters.
- Escape backslashes and quotation marks correctly according to JSON rules.
- Avoid unnecessary try/catch blocks in simple examples.
- Use Big-O notation for complexities, e.g. O(1), O(n), O(sqrt(n)); do not return numeric complexity scores.
- The input_output_examples field MUST be a string, not an object or array.
- Include real source code in the code strings, not Markdown.
User request: {user_request}
'''

    for attempt in range(3):
        try:
            current_prompt = prompt
            if attempt:
                current_prompt += "\nIMPORTANT: Previous response was not valid JSON. Double-check ALL string escaping and return strictly valid JSON only."
            response = client.interactions.create(model=MODEL_NAME, input=current_prompt)
            output = getattr(response, "output_text", None)
            if not isinstance(output, str) or not output.strip():
                raise ValueError("Gemini returned no text")
            result = normalize_result(extract_json(output))
            return jsonify(result), 200
        except (json.JSONDecodeError, ValueError) as error:
            print(f"Gemini output error (attempt {attempt + 1}/3): {safe_error(error)}")
            if attempt == 2:
                return jsonify({"error": "Gemini returned invalid code JSON after 3 attempts. Try a smaller request."}), 502
        except Exception as error:
            code = getattr(error, "code", None)
            status = code if isinstance(code, int) and 400 <= code <= 599 else 502
            print("Gemini API error:", safe_error(error))
            return jsonify({"error": f"Gemini API request failed (HTTP {status}). Check the backend terminal."}), status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)
