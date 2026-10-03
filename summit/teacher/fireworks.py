"""Low-level Fireworks client for teacher sequence scoring.

OPD needs the teacher's logprobs on *student-generated* token sequences.
Fireworks' `/v1/completions` supports the legacy `echo` mode: send the
sequence as a (pre-tokenized) prompt with `max_tokens=1, echo=true,
logprobs=0` and read back per-token logprobs for the prompt itself.

Pre-tokenized integer prompts + `return_token_ids` avoid any
detokenize/retokenize drift between student and teacher (same Qwen3.5
tokenizer family, but we never re-tokenize anyway).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class FireworksEchoError(RuntimeError):
    pass


def align_echo_logprobs(
    attention_mask: list[int],
    token_logprobs: list[float | None],
    output_width: int,
) -> list[float]:
    """Align unpadded echo scores with NexRL's next-token target tensor.

    Fireworks returns one score per unpadded input token. NexRL keeps the
    original padded sequence coordinates and its ``responses`` tensor is
    normally one element narrower than ``input_ids`` (targets for positions
    1..N). Packing only scored response tokens at column zero silently
    misaligns them with ``scoring_attention_mask``.
    """
    active_columns = [i for i, enabled in enumerate(attention_mask) if enabled]
    if len(active_columns) != len(token_logprobs):
        raise FireworksEchoError(
            "attention/logprob length mismatch: "
            f"{len(active_columns)} active tokens, {len(token_logprobs)} scores"
        )
    if output_width < 0 or output_width > len(attention_mask):
        raise FireworksEchoError(
            f"invalid output width {output_width} for input width {len(attention_mask)}"
        )

    first_output_column = len(attention_mask) - output_width
    aligned = [0.0] * output_width
    for column, value in zip(active_columns, token_logprobs):
        output_column = column - first_output_column
        if 0 <= output_column < output_width and value is not None:
            aligned[output_column] = float(value)
    return aligned


class FireworksEchoClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.fireworks.ai/inference",
        timeout_s: float = 300.0,
        max_retries: int = 5,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        if not self.api_key:
            raise FireworksEchoError("FIREWORKS_API_KEY not set")
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=timeout_s, write=30.0, pool=30.0),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def score(self, token_ids: list[int]) -> list[float | None]:
        """Return per-token logprobs for `token_ids` (len == len(token_ids)).

        Index 0 is None (no prior context), matching Tinker's
        `compute_logprobs` convention.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": [int(t) for t in token_ids],
            "max_tokens": 1,
            "echo": True,
            "logprobs": 0,
            "temperature": 0.0,
            "return_token_ids": True,
        }
        resp = self._post_with_retries("/v1/completions", body)
        choice = resp["choices"][0]
        lp = choice.get("logprobs") or {}
        token_logprobs = lp.get("token_logprobs")
        if token_logprobs is None:
            raise FireworksEchoError(f"no logprobs in response: {str(resp)[:500]}")
        out: list[float | None] = [
            None if v is None else float(v) for v in token_logprobs[: len(token_ids)]
        ]
        # Convention: position 0 has no prior context. Fireworks returns 0.0
        # there (Tinker/Weaver return None) — normalize to None.
        if out:
            out[0] = None
        if len(out) < len(token_ids):
            raise FireworksEchoError(
                f"short logprobs: got {len(out)} for {len(token_ids)} prompt tokens"
            )
        # Sanity: when token ids are returned, they must match what we sent.
        echoed_ids = lp.get("token_ids") or choice.get("prompt_token_ids")
        if echoed_ids:
            echoed = [int(t) for t in echoed_ids[: len(token_ids)]]
            if echoed != [int(t) for t in token_ids]:
                raise FireworksEchoError("echoed token ids do not match the request")
        return out

    def score_batch(
        self, batch: list[list[int]], max_workers: int = 8
    ) -> list[list[float | None]]:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(self.score, batch))

    def _post_with_retries(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.post(f"{self.base_url}{path}", json=body)
            except Exception as e:  # noqa: BLE001 - network-level failure, retry
                last_err = e
            else:
                if 200 <= r.status_code < 300:
                    return r.json()
                # 4xx (other than 429) is a hard failure — do not retry.
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise FireworksEchoError(f"HTTP {r.status_code}: {r.text[:500]}")
                last_err = FireworksEchoError(f"HTTP {r.status_code}: {r.text[:300]}")
            wait = min(2**attempt * 2.0, 60.0)
            logger.warning(
                "fireworks %s failed (%s); retry %d in %.0fs", path, last_err, attempt + 1, wait
            )
            time.sleep(wait)
        raise FireworksEchoError(f"request failed after {self.max_retries} retries: {last_err}")
