# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""cache_salt contract for adapters whose conditioning bypasses the token ids.

vLLM's prefix cache hashes the prompt token ids folded with ``cache_salt``.
The adapters under test build placeholder-token prompts and carry the real
voice conditioning (reference waveform, speaker identity, uploaded-voice
generation) in ``additional_information`` — a side channel the block hash
never sees. Without a salt, two requests whose placeholder tokens match share
one cache entry, so with prefix caching enabled a same-text clone can reuse
KV computed from a *different* voice (and Ming-flash, whose whole prompt is a
single ``[0]`` token, would collide across unrelated texts).

The shipped deploy yamls keep ``enable_prefix_caching: false`` for these
models, so today the gap is config-guarded; every sibling AR adapter
(Fish Speech, Qwen3-TTS, Audio8, MOSS, Gepard, Higgs v2/v3, IndexTTS-2) still
salts unconditionally as defense in depth — VoxCPM2's own yaml comment
contemplates turning the cache on. These tests pin the four remaining
adapters to that convention.
"""

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext, conditioning_cache_salt
from vllm_omni.entrypoints.openai.tts_adapters.breeze_tts_2 import BreezeTTS2Adapter
from vllm_omni.entrypoints.openai.tts_adapters.ming_flash_omni_tts import MingFlashOmniTTSAdapter
from vllm_omni.entrypoints.openai.tts_adapters.ming_tts import MingTTSAdapter
from vllm_omni.entrypoints.openai.tts_adapters.voxcpm2 import VoxCPM2Adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REF = "data:audio/wav;base64,UklGRg=="


class _StubServer:
    """Back-reference server exposing what ``build`` touches.

    ``_resolve_ref_audio`` follows the serving_speech contract:
    ``(wav_samples, sample_rate, cache_key)`` where the key is content-aware,
    so a same-path local rewrite resolves to a new key.
    """

    _tts_executor = None
    _max_instructions_length = 2048

    def __init__(self, cache_key: str | None = "key:aaa"):
        self.cache_key = cache_key
        self.resolve_calls: list = []
        self.uploaded_speakers: dict = {}
        self.created_at = 1234

    async def _resolve_ref_audio(self, ref_audio):
        self.resolve_calls.append(ref_audio)
        return [0.0, 0.1], 16000, self.cache_key

    def _voice_created_at(self, voice):
        return self.created_at

    def _load_uploaded_audio(self, voice):
        return None

    def _get_uploaded_audio_data(self, voice):
        return REF


# ---------------------------------------------------------------------------
# VoxCPM2
# ---------------------------------------------------------------------------


def _voxcpm2_adapter(monkeypatch, server) -> VoxCPM2Adapter:
    import vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker as talker

    ctx = SimpleNamespace(
        server=server,
        engine_client=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(), model="vox")),
    )
    adapter = VoxCPM2Adapter(ctx)
    adapter.capabilities = SimpleNamespace(precomputed_speakers={})
    adapter._encode = lambda text: []  # no tokenizer download in unit tests
    monkeypatch.setattr(
        talker,
        "build_voxcpm2_prompt",
        lambda **kwargs: {"type": "token", "prompt_token_ids": [1] * 8},
    )
    return adapter


def test_voxcpm2_build_salts_inline_ref_audio(monkeypatch):
    """The content-aware resolve key must reach the salt, not be discarded."""
    server = _StubServer(cache_key="key:aaa")
    adapter = _voxcpm2_adapter(monkeypatch, server)
    request = OpenAICreateSpeechRequest(input="hello", ref_audio=REF, ref_text="ref")

    prepared = asyncio.run(adapter.build(request, [], False))

    assert server.resolve_calls == [REF]
    assert prepared.tts_params["ref_audio_cache_key"] == "key:aaa"
    assert prepared.prompt["cache_salt"] == conditioning_cache_salt(request, {"ref_audio_cache_key": "key:aaa"})

    # Same request strings, rewritten local file (new resolve key) -> new salt.
    server_b = _StubServer(cache_key="key:bbb")
    prepared_b = asyncio.run(
        _voxcpm2_adapter(monkeypatch, server_b).build(
            OpenAICreateSpeechRequest(input="hello", ref_audio=REF, ref_text="ref"), [], False
        )
    )
    assert prepared_b.prompt["cache_salt"] != prepared.prompt["cache_salt"]


def test_voxcpm2_build_folds_uploaded_voice_created_at(monkeypatch):
    """Delete/re-upload of the same voice name must re-key the cache."""
    server = _StubServer()
    server.uploaded_speakers = {"alice": {}}
    adapter = _voxcpm2_adapter(monkeypatch, server)
    request = OpenAICreateSpeechRequest(input="hello", voice="alice")

    prepared = asyncio.run(adapter.build(request, [], False))

    assert prepared.prompt["additional_information"]["voice_created_at"] == 1234
    assert prepared.tts_params["voice_created_at"] == 1234
    assert prepared.prompt["cache_salt"] == conditioning_cache_salt(request, {"voice_created_at": 1234})

    server.created_at = 9999  # re-uploaded voice
    prepared_b = asyncio.run(adapter.build(OpenAICreateSpeechRequest(input="hello", voice="alice"), [], False))
    assert prepared_b.prompt["cache_salt"] != prepared.prompt["cache_salt"]


# ---------------------------------------------------------------------------
# Breeze-TTS-2
# ---------------------------------------------------------------------------


def _breeze_adapter(server) -> BreezeTTS2Adapter:
    ctx = SpeechServingContext(
        server=server,
        engine_client=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096, model="breeze")),
    )
    adapter = BreezeTTS2Adapter(ctx)

    async def fake_build_async(request, sampling, reference):
        return {"type": "token", "prompt_token_ids": [0] * 8, "additional_information": {}}

    adapter._build_async = fake_build_async
    return adapter


def test_breeze_build_salts_reference_audio():
    server = _StubServer(cache_key="key:aaa")
    adapter = _breeze_adapter(server)
    request = OpenAICreateSpeechRequest(input="hello", voice="S1", ref_audio=REF, ref_text="ref")

    prepared = asyncio.run(adapter.build(request, [SimpleNamespace()], False))

    assert server.resolve_calls == [REF]
    assert prepared.tts_params["ref_audio_cache_key"] == "key:aaa"
    assert prepared.prompt["cache_salt"] == conditioning_cache_salt(request, {"ref_audio_cache_key": "key:aaa"})

    server_b = _StubServer(cache_key="key:bbb")
    prepared_b = asyncio.run(
        _breeze_adapter(server_b).build(
            OpenAICreateSpeechRequest(input="hello", voice="S1", ref_audio=REF, ref_text="ref"),
            [SimpleNamespace()],
            False,
        )
    )
    assert prepared_b.prompt["cache_salt"] != prepared.prompt["cache_salt"]


def test_breeze_build_tolerates_missing_resolve_key():
    """A None key (mocked resolvers) must not leak into tts_params."""
    server = _StubServer(cache_key=None)
    adapter = _breeze_adapter(server)
    request = OpenAICreateSpeechRequest(input="hello", ref_audio=REF, ref_text="ref")

    prepared = asyncio.run(adapter.build(request, [SimpleNamespace()], False))

    assert "ref_audio_cache_key" not in prepared.tts_params
    assert prepared.prompt["cache_salt"] == conditioning_cache_salt(request, {})


# ---------------------------------------------------------------------------
# Ming-flash-omni TTS (single [0] placeholder token for everything)
# ---------------------------------------------------------------------------


def _ming_flash_adapter(server) -> MingFlashOmniTTSAdapter:
    return MingFlashOmniTTSAdapter(SimpleNamespace(server=server, engine_client=SimpleNamespace()))


def test_ming_flash_salt_separates_identical_placeholder_prompts():
    """Every request shares one token id; the salt is the only discriminator."""
    adapter = _ming_flash_adapter(_StubServer())
    r1 = OpenAICreateSpeechRequest(input="hello world")
    r2 = OpenAICreateSpeechRequest(input="goodbye moon")

    p1 = asyncio.run(adapter.build(r1, [], False))
    p2 = asyncio.run(adapter.build(r2, [], False))

    assert p1.prompt["prompt_token_ids"] == p2.prompt["prompt_token_ids"] == [0]
    assert p1.prompt["cache_salt"] != p2.prompt["cache_salt"]
    assert p1.prompt["cache_salt"] == conditioning_cache_salt(r1, {})


def test_ming_flash_salt_covers_voice_and_speaker_embedding():
    adapter = _ming_flash_adapter(_StubServer())
    base = OpenAICreateSpeechRequest(input="hello", voice="DB30")
    emb = OpenAICreateSpeechRequest(input="hello", voice="DB30", speaker_embedding=[0.1, 0.2, 0.3])
    other_voice = OpenAICreateSpeechRequest(input="hello", voice="DB31")

    p_base = asyncio.run(adapter.build(base, [], False))
    p_emb = asyncio.run(adapter.build(emb, [], False))
    p_voice = asyncio.run(adapter.build(other_voice, [], False))

    assert len({p_base.prompt["cache_salt"], p_emb.prompt["cache_salt"], p_voice.prompt["cache_salt"]}) == 3


# ---------------------------------------------------------------------------
# Ming-TTS (dense)
# ---------------------------------------------------------------------------


def _ming_tts_adapter(server) -> MingTTSAdapter:
    adapter = MingTTSAdapter(SpeechServingContext(server=server))

    def fake_build_prompt(request, ref_audio_data=None, voice_name=None, voice_created_at=0):
        additional: dict = {}
        if voice_name:
            additional["voice_name"] = voice_name
            additional["voice_created_at"] = int(voice_created_at)
        return {"type": "token", "prompt_token_ids": [7] * 8, "additional_information": additional}

    adapter._build_ming_dense_prompt = fake_build_prompt
    return adapter


def test_ming_tts_salt_folds_inline_ref_key_without_polluting_model_params():
    server = _StubServer(cache_key="key:aaa")
    adapter = _ming_tts_adapter(server)
    request = OpenAICreateSpeechRequest(input="hi", ref_audio=REF, ref_text="ref")

    prepared = asyncio.run(adapter.build(request, [], False))

    assert prepared.prompt["cache_salt"] == conditioning_cache_salt(request, {"ref_audio_cache_key": "key:aaa"})
    # The resolve key feeds the salt only; the model-side dict is unchanged.
    assert "ref_audio_cache_key" not in prepared.tts_params

    server_b = _StubServer(cache_key="key:bbb")
    prepared_b = asyncio.run(
        _ming_tts_adapter(server_b).build(
            OpenAICreateSpeechRequest(input="hi", ref_audio=REF, ref_text="ref"), [], False
        )
    )
    assert prepared_b.prompt["cache_salt"] != prepared.prompt["cache_salt"]


def test_ming_tts_salt_folds_uploaded_voice_created_at():
    server = _StubServer(cache_key="key:aaa")
    server.uploaded_speakers = {"alice": {"ref_text": "uploaded ref"}}
    adapter = _ming_tts_adapter(server)
    request = OpenAICreateSpeechRequest(input="hi", voice="alice")

    prepared = asyncio.run(adapter.build(request, [], False))

    # build() adopts the uploaded clip (resolved through the same contract)
    # and voice_created_at travels in additional_information == tts_params.
    assert server.resolve_calls == [REF]
    assert prepared.tts_params["voice_created_at"] == 1234
    expected = conditioning_cache_salt(request, {"voice_created_at": 1234, "ref_audio_cache_key": "key:aaa"})
    assert prepared.prompt["cache_salt"] == expected

    server.created_at = 9999
    prepared_b = asyncio.run(adapter.build(OpenAICreateSpeechRequest(input="hi", voice="alice"), [], False))
    assert prepared_b.prompt["cache_salt"] != prepared.prompt["cache_salt"]
