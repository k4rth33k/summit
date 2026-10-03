import pytest

from summit.teacher.fireworks import (
    FireworksEchoClient,
    FireworksEchoError,
    align_echo_logprobs,
)


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


def make_client(payload=None, status=200):
    c = FireworksEchoClient(model="m", api_key="k", max_retries=1)
    calls = []

    class FakeHTTP:
        def post(self, url, json=None):
            calls.append(json)
            return FakeResp(payload, status)

    c._client = FakeHTTP()
    c._calls = calls
    return c


def echo_payload(token_ids):
    # legacy LogProbs: first entry None, then one float per token + 1 generated
    lps = [None] + [-0.1 * i for i in range(1, len(token_ids) + 2)]
    return {
        "choices": [
            {
                "text": "x",
                "logprobs": {
                    "tokens": [str(t) for t in token_ids] + ["999"],
                    "token_logprobs": lps,
                    "token_ids": list(token_ids) + [999],
                },
                "prompt_token_ids": list(token_ids),
            }
        ]
    }


def test_score_alignment():
    tokens = [10, 11, 12, 13]
    c = make_client(echo_payload(tokens))
    lps = c.score(tokens)
    assert len(lps) == 4
    assert lps[0] is None
    assert lps[1] == pytest.approx(-0.1)
    # request must use echo + pre-tokenized prompt
    body = c._calls[0]
    assert body["prompt"] == tokens
    assert body["echo"] is True and body["max_tokens"] == 1 and body["logprobs"] == 0


def test_score_token_mismatch_raises():
    c = make_client(echo_payload([1, 2, 3]))
    with pytest.raises(FireworksEchoError, match="do not match"):
        c.score([9, 9, 9])


def test_score_short_logprobs_raises():
    payload = {"choices": [{"logprobs": {"token_logprobs": [None, -0.5]}}]}
    c = make_client(payload)
    with pytest.raises(FireworksEchoError, match="short logprobs"):
        c.score([1, 2, 3, 4, 5])


def test_score_batch():
    c = make_client(echo_payload([5, 6]))
    out = c.score_batch([[5, 6], [5, 6]])
    assert len(out) == 2 and all(len(r) == 2 for r in out)


def test_first_token_zero_normalized_to_none():
    # Fireworks returns 0.0 (not None) at position 0 — must normalize.
    tokens = [7, 8]
    payload = echo_payload(tokens)
    payload["choices"][0]["logprobs"]["token_logprobs"][0] = 0.0
    c = make_client(payload)
    assert c.score(tokens)[0] is None


def test_missing_api_key():
    import os

    os.environ.pop("FIREWORKS_API_KEY", None)
    with pytest.raises(FireworksEchoError):
        FireworksEchoClient(model="m", api_key=None)


def test_align_echo_logprobs_preserves_padded_sequence_coordinates():
    # Three active tokens at padded input columns 2, 3, 4. NexRL's four-wide
    # target tensor corresponds to input columns 1..4, so scores belong in
    # target columns 1, 2, 3 rather than being packed at 0, 1, 2.
    aligned = align_echo_logprobs(
        [0, 0, 1, 1, 1], [None, -2.0, -3.0], output_width=4
    )
    assert aligned == [0.0, 0.0, -2.0, -3.0]


def test_align_echo_logprobs_rejects_length_mismatch():
    with pytest.raises(FireworksEchoError, match="length mismatch"):
        align_echo_logprobs([1, 1, 1], [None, -1.0], output_width=2)
