import json
import subprocess
from pathlib import Path

import pytest

from docsrag.ingest.projects import collect, is_secret, notebook_text

AWS = "AKIA" + "ABCDEFGHIJKLMNOP"  # split so this test file itself doesn't look like a leak
PEM = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----"


@pytest.mark.parametrize("name", [".env", ".env.example", ".env.local", "backend/.env", ".envrc",
                                  "id_rsa", "deploy.pem", "credentials.json", "client_secret.json",
                                  "token.json", ".npmrc"])
def test_secret_files_are_rejected_by_name(name):
    # Never indexed, whatever they contain: .env.example often holds real values too.
    assert is_secret(name, "harmless")


@pytest.mark.parametrize("text", [f"KEY = '{AWS}'", PEM, 'api_key = "abcdef1234567890xyz"',
                                  "OPENAI = 'sk-" + "a" * 40 + "'", "gh = 'ghp_" + "a" * 36 + "'"])
def test_secret_content_is_rejected_even_in_normal_files(text):
    # Committed secrets are git-tracked, so a name check alone would let them through.
    assert is_secret("src/config.py", text)


def test_ordinary_code_is_not_a_secret():
    assert not is_secret("src/auth.py", "def login(password: str):\n    token = make_token(user)\n")


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_root(tmp_path: Path) -> Path:
    app = tmp_path / "myapp"
    (app / "src").mkdir(parents=True)
    (app / "README.md").write_text("# MyApp\n\nTracks train delays.\n")
    (app / "package.json").write_text(json.dumps({"dependencies": {"next": "15"}, "devDependencies": {"vitest": "1"}}))
    (app / "src/auth.ts").write_text("export function login() { return session() }\n")
    (app / "src/leak.py").write_text(f"KEY = '{AWS}'\n")
    (app / ".env").write_text("DB_PASSWORD=hunter2hunter2\n")
    (app / "package-lock.json").write_text("{}")
    (app / "src/big.js").write_text("x" * 200_000)
    (app / "src/logo.ts").write_bytes(b"\xff\xfe\x00binary")
    (app / ".gitignore").write_text("dist/\n")
    (app / "dist").mkdir()
    (app / "dist/out.js").write_text("compiled()")
    git(app, "init", "-q")
    git(app, "add", "-A", "-f", ".env", ".")  # a committed .env must still be rejected
    git(app, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "Add login flow")

    loose = tmp_path / "sketch"  # non-git folder
    (loose / "node_modules/dep").mkdir(parents=True)
    (loose / "node_modules/dep/index.js").write_text("module.exports = 1")
    (loose / "main.py").write_text("print('hi')\n")

    (tmp_path / "LLaVA").mkdir()
    (tmp_path / "LLaVA/model.py").write_text("class Llava: pass\n")
    return tmp_path


def test_collect_keeps_real_files_and_drops_secrets_junk_and_skipped_projects(tmp_path):
    docs, skipped = collect(make_root(tmp_path), skip={"LLaVA"})
    ids = {d["id"] for d in docs}
    assert {"projects:myapp/src/auth.ts", "projects:myapp/README.md", "projects:myapp/package.json",
            "projects:sketch/main.py"} <= ids
    for bad in ["myapp/.env", "myapp/src/leak.py", "myapp/package-lock.json", "myapp/src/big.js",
                "myapp/src/logo.ts", "myapp/dist/out.js", "sketch/node_modules/dep/index.js", "LLaVA/model.py"]:
        assert f"projects:{bad}" not in ids, bad
    # Skipped secrets are reported by path so they can be audited -- never their content.
    assert set(skipped) == {"myapp/.env", "myapp/src/leak.py"}
    assert not any("hunter2" in d["body"] or AWS in d["body"] for d in docs)


def test_file_docs_carry_a_label_so_chunks_know_their_project_and_path(tmp_path):
    docs, _ = collect(make_root(tmp_path), skip=set())
    auth = next(d for d in docs if d["id"] == "projects:myapp/src/auth.ts")
    assert auth["source_type"] == "projects"
    assert auth["metadata"] == {"project": "myapp", "path": "src/auth.ts", "kind": "code",
                                "label": "myapp/src/auth.ts"}


def test_overview_answers_big_picture_questions(tmp_path):
    # "What does myapp do / what stack / what was I working on" need README + deps + history in one doc.
    docs, _ = collect(make_root(tmp_path), skip=set())
    ov = next(d for d in docs if d["id"] == "projects:myapp/__overview__")
    for fact in ["Tracks train delays", "next", "vitest", "Add login flow", "src/", ".ts"]:
        assert fact in ov["body"], fact
    assert "hunter2" not in ov["body"]


def test_notebook_keeps_cell_sources_and_drops_outputs():
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["# Price model\n"]},
        {"cell_type": "code", "source": ["fit(X, y)\n"], "outputs": [{"text": ["HUGE OUTPUT"]}]},
    ]}
    text = notebook_text(json.dumps(nb))
    assert "# Price model" in text and "fit(X, y)" in text
    assert "HUGE OUTPUT" not in text


@pytest.mark.parametrize("rel", ["app/Http/Controllers/AuthController.php",
                                 "resources/views/home.blade.php", "resources/js/App.vue", ".cursor/rules/x.mdc"])
def test_laravel_and_vue_sources_are_indexed(rel):
    # collabdocs and habbo-agency-portal are Laravel/Vue apps: without these their code is invisible.
    from docsrag.ingest.projects import _allowed
    assert _allowed(rel)


def test_vendored_dependencies_are_not_indexed():
    # Composer's vendor/ is third-party code that would drown out the app's own files.
    from docsrag.ingest.projects import _allowed
    assert not _allowed("vendor/laravel/framework/src/Auth.php")


@pytest.mark.parametrize("rel", ["docs/superpowers/plans/plan.md", "lib/glfw-3.4.bin.MACOS/docs/html/index.html"])
def test_docs_folders_are_hidden(rel):
    # Toprak's choice: docs/ (specs/plans, vendored library docs) stays out of the index.
    from docsrag.ingest.projects import _allowed
    assert not _allowed(rel)


@pytest.mark.parametrize("text", [
    "    environment:\n      POSTGRES_PASSWORD: devpass123\n",              # compose literal
    "DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD:-fuutball}@db/x",  # fallback default
    "url = 'postgresql://app:hunter22@localhost/db'",                       # inline URL credentials
    "JWT_SECRET=abcd1234\n",
])
def test_env_style_credentials_are_rejected(text):
    # An eval leaked fuutball's compose default DB password: unquoted/env-style values must count too.
    assert is_secret("docker-compose.yml", text)


@pytest.mark.parametrize("text", [
    "      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}\n",                    # reference, no value
    "DATABASE_URL=postgresql://${USER}:${PASS}@db/x",
    "PASSWORD_MIN_LENGTH = 8\n",
    "API_TOKEN = None\n",
    "SECRET_KEY = os.environ['SECRET_KEY']\n",
    "def login(password: str): ...\n",
])
def test_credential_references_are_not_secrets(text):
    # Over-blocking these would hide most auth/config code, which is exactly what gets asked about.
    assert not is_secret("settings.py", text)


def test_eval_answer_key_is_never_indexed(tmp_path):
    # docsrag indexes itself; its questions.yaml would hand retrieval the gold answers.
    (tmp_path / "docsrag/evals").mkdir(parents=True)
    (tmp_path / "docsrag/evals/questions.yaml").write_text("- id: x\n")
    (tmp_path / "docsrag/evals/questions.local.yaml").write_text("- id: y\n")
    (tmp_path / "docsrag/main.py").write_text("print(1)\n")
    docs, _ = collect(tmp_path, skip=set())
    ids = {d["id"] for d in docs}
    assert "projects:docsrag/evals/questions.yaml" not in ids
    # Gitignored, but a project without .git is walked without .gitignore: skip it explicitly.
    assert "projects:docsrag/evals/questions.local.yaml" not in ids


# Built by concatenation so these fixtures don't read as real keys in this file.
KNOWN_FORMATS = {
    "stripe": "STRIPE = '" + "sk_" + "live_" + "a1B2c3D4" * 3 + "'",
    "huggingface": "HF = '" + "hf_" + "aB3dE5fG7h" * 3 + "xyzw'",
    "openai project": "K = '" + "sk-" + "proj-" + "Ab1Cd2Ef3G" * 4 + "'",
    "jwt": "bearer = '" + "eyJhbGciOiJIUzI1NiJ9" + ".eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U'",
    "sendgrid": "SG_ = '" + "SG." + "aB3dE5fG7hJ9kL1mN3pQ5r" + "." + "sT7uV9wX1yZ3aB5cD7eF9gH1iJ3kL5mN7oP9qR1sT3u'",
    "npm": "//registry.npmjs.org/:_authToken=" + "npm_" + "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3zA5",
    "pypi": "password = " + "pypi-" + "AgEIcHlwaS5vcmcCJGFiY2RlZjEyLTM0NTYtNzg5MC1hYmNkLWVmMTIzNDU2Nzg5MAAC",
    "twilio": "TWILIO = '" + "SK" + "0123456789abcdef" * 2 + "'",
}


@pytest.mark.parametrize("name", KNOWN_FORMATS)
def test_more_provider_key_formats_are_rejected(name):
    assert is_secret("app/settings.py", KNOWN_FORMATS[name])


@pytest.mark.parametrize("text", [
    'GEMINI_KEY = "Xq7mP2vL9nR4tK8wZ3bY"',                       # unknown provider, telling name
    '{"auth": "q8ZtP3mXr7Lk2Vn9BtR4"}',                          # JSON config
    'client = Client("Zx81QmP0aLk2Vn9BtR7cWs4YhG6dJe3F")',       # no telling name at all
])
def test_unknown_format_keys_are_caught_by_randomness(text):
    # Future projects will use providers we have no pattern for; random-looking literals give them away.
    assert is_secret("src/client.py", text)


@pytest.mark.parametrize("text", [
    "commit = '9948e17345e530ac28b1d883efa265916b9de790'",       # git SHA (hex only)
    "id = '123e4567-e89b-12d3-a456-426614174000'",               # UUID
    'msg = "This sentence is long but it is ordinary English text."',
    'path = "frontend/src/features/tactics/TacticsPage.tsx"',
    "token_refresh_interval_seconds = 3600",
    'API_KEY_HEADER = "X-Api-Key-Header-Name"',                 # telling name, but not random
    'className="flex items-center justify-between rounded-lg px-4"',
])
def test_ordinary_long_strings_are_not_flagged(text):
    # Over-blocking hides normal code; these are the shapes that dominate real projects.
    assert not is_secret("src/app.py", text)
