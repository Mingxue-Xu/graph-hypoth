"""The local web app: start pipeline runs, confirm hypotheses in the browser, browse the graph.

``runs`` drives the CLI's own ``run_synthesist`` on a worker thread and replaces the stdin
confirmation prompt with a browser-answered gate; ``graph_view`` projects a run directory into the
JSON the canvas draws; ``app`` is the FastAPI application and ``server`` the ``graph-hypoth-web``
entry point. Only ``app`` and ``server`` need the ``web`` extra (FastAPI + uvicorn).
"""
