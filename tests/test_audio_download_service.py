from pathlib import Path

from app.services.audio_download_service import download_audio


class FakeResponse:
    status_code = 200
    headers = {"Content-Length": "5", "Content-Type": "audio/mpeg"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield b"audio"


def test_download_audio_fetches_signed_url_exactly_as_provided(monkeypatch, tmp_path):
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr("app.services.audio_download_service.requests.get", fake_get)

    signed_url = "https://example.com/wp-json/wpab/v1/audio-download?token=a%2Fb&expires=123"
    result = download_audio(signed_url, Path(tmp_path), "job-1")

    assert captured["url"] == signed_url
    assert "headers" not in captured["kwargs"]
    assert result.source_path.read_bytes() == b"audio"
