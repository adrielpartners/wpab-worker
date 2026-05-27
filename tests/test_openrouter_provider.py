import base64

from app.services.providers.transcription.openrouter_provider import OpenRouterProvider


class FakeResponse:
    status_code = 200
    text = '{"text":"ok"}'

    def raise_for_status(self):
        return None

    def json(self):
        return {"text": "ok"}


def test_openrouter_provider_uses_json_base64_payload(monkeypatch, tmp_path):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    chunk = tmp_path / "chunk.mp3"
    chunk.write_bytes(b"audio bytes")
    monkeypatch.setattr("requests.post", fake_post)

    provider = OpenRouterProvider("sk-test", "https://openrouter.ai/api/v1")
    text = provider.transcribe_chunk(chunk, "openai/whisper-large-v3", "job-1")

    assert text == "ok"
    assert captured["url"] == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["json"]["model"] == "openai/whisper-large-v3"
    assert captured["json"]["input_audio"]["format"] == "mp3"
    assert captured["json"]["input_audio"]["data"] == base64.b64encode(b"audio bytes").decode("ascii")
