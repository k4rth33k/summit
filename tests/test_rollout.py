from summit.rollout.mini_swe import repair_prompt_tokens


def test_repair_fills_from_batchencoding():
    resp = {"choices": [], "nexrl_train": {"prompt_tokens": [], "response_tokens": [1, 2]}}

    class FakeBatchEncoding(dict):
        pass

    out = repair_prompt_tokens(resp, lambda: FakeBatchEncoding(input_ids=[10, 11, 12]))
    assert out["nexrl_train"]["prompt_tokens"] == [10, 11, 12]


def test_repair_noop_when_present():
    resp = {"choices": [], "nexrl_train": {"prompt_tokens": [5], "response_tokens": [1]}}
    out = repair_prompt_tokens(resp, lambda: (_ for _ in ()).throw(AssertionError))
    assert out["nexrl_train"]["prompt_tokens"] == [5]


def test_repair_batched_ids():
    resp = {"choices": [], "nexrl_train": {"prompt_tokens": []}}
    out = repair_prompt_tokens(resp, lambda: [[7, 8]])
    assert out["nexrl_train"]["prompt_tokens"] == [7, 8]
