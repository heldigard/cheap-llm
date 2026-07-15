from _livetestlib import *  # noqa: E402,F401,F403  -- harness + gate

# =================================================================
# LIVE: real cascade calls
# =================================================================
if LIVE:
    print("\n=== LIVE: real cascade (Ollama + OpenRouter) ===")
    if not (HAVE_OR or HAVE_OLLAMA):
        skip("LIVE", "all live tests", "no OPENROUTER_API_KEY and no Ollama")
    else:
        # L1: configured T1 local model directly reachable through Ollama
        if HAVE_OLLAMA:
            try:
                r = cl._call_ollama(
                    cl.DEFAULT_LOCAL_PRIMARY, CLASSIFY_SYS, CLASSIFY_PROMPT, timeout=15
                )
                txt = r.get("text", "")
                check(
                    "LIVE",
                    "T1 local model@ollama reachable + non-empty",
                    bool(txt) and len(txt) > 5,
                    detail=f"lat={r.get('latency', 0):.1f}s out_tok={r.get('output_tokens', 0)}",
                )
            except Exception as e:
                check(
                    "LIVE",
                    "T1 local model@ollama reachable",
                    False,
                    detail=f"{type(e).__name__}: {str(e)[:70]}",
                )
        else:
            skip("LIVE", "T1 local model@ollama reachable", "Ollama down")

        # L2: full cascade, prefer_local=True (the default every caller uses)
        try:
            t0 = time.perf_counter()
            out = cl.cheap_complete(
                system=CLASSIFY_SYS,
                prompt=CLASSIFY_PROMPT,
                schema_hint=["category", "reason"],
                timeout_total=20,
                prefer_local=True,
            )
            wall = time.perf_counter() - t0
            check(
                "LIVE",
                "cascade prefer_local=True resolves",
                out.get("model") is not None and out.get("json_valid") is True,
                detail=f"model={out.get('model')} tier={out.get('tier')} wall={wall:.1f}s",
            )
            check(
                "LIVE",
                "cascade returns valid category field",
                bool(_safe_get_category(out.get("text", ""))),
                detail=f"category={_safe_get_category(out.get('text', ''))}",
            )
        except Exception as e:
            check(
                "LIVE",
                "cascade prefer_local=True resolves",
                False,
                detail=f"{type(e).__name__}: {str(e)[:70]}",
            )

        # L3: full cascade, prefer_local=False (cloud-first)
        try:
            out = cl.cheap_complete(
                system=CLASSIFY_SYS,
                prompt=CLASSIFY_PROMPT,
                schema_hint=["category", "reason"],
                timeout_total=20,
                prefer_local=False,
            )
            check(
                "LIVE",
                "cascade prefer_local=False resolves on cloud",
                out.get("model") is not None and out.get("provider") != "ollama",
                detail=(
                    f"model={out.get('model')} provider={out.get('provider')} "
                    f"cost=${out.get('cost', 0):.6f}"
                ),
            )
        except Exception as e:
            check(
                "LIVE",
                "cascade prefer_local=False resolves",
                False,
                detail=f"{type(e).__name__}: {str(e)[:70]}",
            )

        # L4: cache hit — repeat identical call → cached=True, no provider call
        try:
            cl._cache_put  # ensure present
            cl.cheap_complete(
                system="Cache probe.",
                prompt="identical-cache-key-prompt-xyz",
                schema_hint=["category"],
                timeout_total=20,
                prefer_local=False,
            )
            # record how many attempts the first call made, then second must be cached
            out2 = cl.cheap_complete(
                system="Cache probe.",
                prompt="identical-cache-key-prompt-xyz",
                schema_hint=["category"],
                timeout_total=20,
                prefer_local=False,
            )
            check(
                "LIVE",
                "repeat call hits cache",
                out2.get("cached") is True and out2.get("latency", 99) < 0.05,
                detail=f"cached={out2.get('cached')} lat={out2.get('latency', 0):.3f}s",
            )
        except Exception as e:
            check(
                "LIVE", "repeat call hits cache", False, detail=f"{type(e).__name__}: {str(e)[:70]}"
            )

        # L5: scrub confirmed on the LIVE path (critical-fix regression).
        # Spy on _call_provider: assert a planted secret never reaches it,
        # while the call still resolves normally.
        try:
            seen: dict = {}
            orig = cl._call_provider

            def spy(
                model, provider, system, prompt, timeout, max_output_tokens, require_json=False
            ):
                seen["prompt"] = prompt
                seen["system"] = system
                return orig(
                    model,
                    provider,
                    system,
                    prompt,
                    timeout,
                    max_output_tokens,
                    require_json,
                )

            cl._call_provider = spy
            try:
                out = cl.cheap_complete(
                    system=CLASSIFY_SYS,
                    prompt="DEBUG: Authorization: Bearer eyJhbGc.iO.SflKx; "
                    "db=postgres://admin:Hunter2Secret@db:5432 — classify this",
                    schema_hint=["category", "reason"],
                    timeout_total=20,
                    prefer_local=True,
                )
            finally:
                cl._call_provider = orig
            leaked = any(
                s in (seen.get("prompt", "") + seen.get("system", ""))
                for s in ("eyJhbGc", "Hunter2Secret")
            )
            check(
                "LIVE",
                "secret scrubbed before live provider call",
                not leaked and out.get("model") is not None,
                detail=f"leaked={leaked} resolved={out.get('model') is not None}",
            )
        except Exception as e:
            check(
                "LIVE",
                "secret scrubbed before live provider call",
                False,
                detail=f"{type(e).__name__}: {str(e)[:70]}",
            )

        # L6: ZenMux reachable independently (failover path is real)
        if HAVE_ZM:
            try:
                r = cl._call_zenmux(
                    cl.TOP3_CASCADE[0][0], CLASSIFY_SYS, CLASSIFY_PROMPT, timeout=15
                )
                check(
                    "LIVE",
                    "ZenMux failover tier reachable",
                    bool(r.get("text")),
                    detail=(
                        f"provider={r.get('provider')} cost_est=${r.get('api_cost', 0) or 0:.6f}"
                    ),
                )
            except Exception as e:
                check(
                    "LIVE",
                    "ZenMux failover tier reachable",
                    False,
                    detail=f"{type(e).__name__}: {str(e)[:70]}",
                )
        else:
            skip("LIVE", "ZenMux failover tier reachable", "ZENMUX_API_KEY not set")
else:
    print("\n=== LIVE: (--e2e-only, skipped) ===")
