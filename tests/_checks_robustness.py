from _testlib import *  # noqa: E402,F401,F403  -- harness + shared fixtures

print("\n=== UNIT: robustness additions ===")
_v_beta = cl._parse_version("1.2.3-beta")
check("parse version with beta suffix", _v_beta == (1, 2, 3), detail=f"got {_v_beta}")
_v_dev = cl._parse_version("1.2.4.dev0")
check("parse version with dev suffix", _v_dev == (1, 2, 4), detail=f"got {_v_dev}")
check("parse version with normal string", cl._parse_version("2.0.1") == (2, 0, 1))

# Verify Ollama request payload structure
_ollama_payload_struct: dict = {}
_orig_urlopen_struct = _urlreq.urlopen
_urlreq.urlopen = _fake_urlopen_factory(
    {"response": "ok", "prompt_eval_count": 2, "eval_count": 1}, _ollama_payload_struct
)
try:
    cl._call_ollama("local", "sys_p", "user_p", timeout=5, max_output_tokens=192)
finally:
    _urlreq.urlopen = _orig_urlopen_struct
check("ollama payload separate prompt", _ollama_payload_struct.get("prompt") == "user_p")
check("ollama payload separate system", _ollama_payload_struct.get("system") == "sys_p")

# Verify global reasoning stripping for cloud / OpenAI compat / deepseek calls
_body_with_think = {
    "choices": [{"message": {"content": "<think>Let me think...</think>final answer"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 10},
}
_compat_payload_think: dict = {}
_urlreq.urlopen = _fake_urlopen_factory(_body_with_think, _compat_payload_think)
try:
    os.environ["OPENROUTER_API_KEY"] = "test"
    _or_resp = cl._call_openrouter(
        "openai/gpt-5.4-nano", "s", "p", timeout=5, max_output_tokens=128
    )
    check(
        "openrouter strips reasoning block",
        _or_resp["text"] == "final answer",
        detail=f"got {_or_resp['text']!r}",
    )
finally:
    _urlreq.urlopen = _orig_urlopen_struct
    del os.environ["OPENROUTER_API_KEY"]

_urlreq.urlopen = _fake_urlopen_factory(_body_with_think)
_old_deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
try:
    os.environ["DEEPSEEK_API_KEY"] = "test"
    _ds_resp = cl._call_deepseek(
        "deepseek/deepseek-v4-flash", "s", "p", timeout=5, max_output_tokens=128
    )
    check(
        "deepseek strips reasoning block",
        _ds_resp["text"] == "final answer",
        detail=f"got {_ds_resp['text']!r}",
    )
finally:
    _urlreq.urlopen = _orig_urlopen_struct
    if _old_deepseek_key is None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    else:
        os.environ["DEEPSEEK_API_KEY"] = _old_deepseek_key

_json_internal_fences = '```json\n{"code": "```python\\nprint(1)\\n```"}\n```'
_parsed_fences = cl._try_parse_json(_json_internal_fences)
check(
    "parse JSON with internal code fences",
    _parsed_fences == {"code": "```python\nprint(1)\n```"},
    detail=f"got {_parsed_fences}",
)
