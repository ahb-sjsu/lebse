from lebse import sql


def test_snippet_uses_standard_string_not_E():
    # The core correctness bug: E'\s+' silently becomes literal 's' and leaves newlines.
    s = sql.snippet_sql("plain_text")
    assert r"'\s+'" in s
    assert "E'" not in s
    assert "left(plain_text, 3000)" in s


def test_snippet_partial_detoast_never_length():
    # Must use left() (partial detoast), never length() (full-column detoast of 100GB).
    s = sql.snippet_sql("plain_text", n=2000)
    assert "left(" in s
    assert "length(" not in s
    assert "2000" in s


def test_pool_sql_newest_first_and_limited():
    s = sql.pool_sql(9400, 6000)
    assert "id > 9400" in s
    assert "ORDER BY id DESC" in s  # newest = cleanest plain_text
    assert "LIMIT 6000" in s


def test_citation_edge_sql_is_int_only_no_join():
    s = sql.citation_edge_sql(9400, 4000)
    assert "search_opinionscited" in s
    assert "JOIN search_opinion " not in s  # phase 1 is int-only; text fetched separately
    assert "citing_opinion_id > 9400" in s and "cited_opinion_id > 9400" in s


def test_cocitation_edge_sql_self_join_on_common_cited():
    s = sql.cocitation_edge_sql(9400, 4000)
    assert "o2.cited_opinion_id=o1.cited_opinion_id" in s
    assert "o2.citing_opinion_id > o1.citing_opinion_id" in s  # unordered pair, once


def test_snippets_by_id_uses_indexed_any_array():
    s = sql.snippets_by_id_sql([5, 1, 5, 3])
    assert "id = ANY('{1,3,5}'::int[])" in s  # deduped + sorted
