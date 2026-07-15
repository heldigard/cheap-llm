"""Allow ``python3 -m cheap_llm`` to run the CLI."""

from .cli import main

raise SystemExit(main())
