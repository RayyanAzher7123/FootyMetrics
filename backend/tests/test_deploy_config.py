from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_compose_starts_the_file_backed_stack():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    backend, rest = text.split("frontend:", maxsplit=1)
    assert "DATABASE_URL" not in text
    assert "depends_on" not in backend
    assert "condition: service_healthy" in rest
    assert 'profiles: ["postgres"]' in text


def test_nginx_proxies_the_api():
    nginx = (ROOT / "frontend" / "nginx.conf").read_text(encoding="utf-8")
    assert "location /api/" in nginx
    assert "proxy_pass http://backend:8000/api/;" in nginx


def test_image_build_skips_local_environments():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    frontend_ignored = (ROOT / "frontend" / ".dockerignore").read_text(encoding="utf-8")
    assert ".venv" in ignored
    assert "frontend" in ignored
    assert "node_modules" in frontend_ignored
