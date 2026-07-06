r"""SQL builders for a CourtListener Postgres replica — pure string functions, unit-tested.

Key correctness note baked in here: whitespace collapsing uses a *standard* string ``'\s+'``, NOT
``E'\s+'``. In a PostgreSQL ``E''`` string, ``\s`` is an unknown escape and is silently reduced to a
literal ``s``, so newlines are NOT collapsed — which truncates snippets when the client splits on
newlines. With ``standard_conforming_strings=on`` (the default), ``'\s+'`` is passed verbatim to the
regex engine. The tests assert we never emit the ``E'`` form.
"""

from __future__ import annotations


def snippet_sql(col: str, n: int = 3000) -> str:
    r"""SQL expression yielding the first ``n`` chars of ``col`` with whitespace collapsed.

    Uses ``left()`` for a *partial* TOAST detoast (only the first chunks are read) — never
    ``length(col)``, which would force decompressing the whole column.
    """
    return rf"regexp_replace(left({col}, {n}), '\s+', ' ', 'g')"


def pool_sql(thr: int, limit: int, table: str = "search_opinion") -> str:
    """Held-out opinion snippets: ids past the training cut. ORDER BY id DESC = NEWEST opinions,
    which have the cleanest ``plain_text`` (oldest ids are scanned/short)."""
    snip = snippet_sql("plain_text")
    return f"SELECT id, {snip} FROM {table} WHERE id > {thr} ORDER BY id DESC LIMIT {limit};"


def citation_edge_sql(thr: int, limit: int) -> str:
    """Phase-1 citation edge ids only (no text, no join) — fast, index-friendly. DESC = newest."""
    return (
        "SELECT citing_opinion_id, cited_opinion_id FROM search_opinionscited "
        f"WHERE citing_opinion_id > {thr} AND cited_opinion_id > {thr} "
        f"ORDER BY id DESC LIMIT {limit};"
    )


def cocitation_edge_sql(thr: int, limit: int) -> str:
    """Phase-1 co-citation edge ids: pairs that cite a common opinion (same-doctrine proxy)."""
    return (
        "SELECT o1.citing_opinion_id, o2.citing_opinion_id FROM search_opinionscited o1 "
        "JOIN search_opinionscited o2 ON o2.cited_opinion_id=o1.cited_opinion_id "
        "AND o2.citing_opinion_id > o1.citing_opinion_id "
        f"WHERE o1.citing_opinion_id > {thr} AND o2.citing_opinion_id > {thr} "
        f"ORDER BY o1.id DESC LIMIT {limit};"
    )


def snippets_by_id_sql(ids: list[int]) -> str:
    """Batch snippet fetch via indexed PK lookups (``id = ANY(array)``)."""
    arr = "{" + ",".join(str(int(i)) for i in sorted(set(ids))) + "}"
    snip = snippet_sql("plain_text")
    return f"SELECT id, {snip} FROM search_opinion WHERE id = ANY('{arr}'::int[]);"
