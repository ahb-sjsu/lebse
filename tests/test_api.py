import lebse


def test_version_exposed():
    assert isinstance(lebse.__version__, str)
    assert lebse.__version__.count(".") >= 2


def test_default_model_id():
    assert lebse.DEFAULT_MODEL == "ahbond/lebse"


def test_import_is_torch_free():
    # `import lebse` must not pull torch/sentence-transformers (kept lazy in lebse.load).
    import sys

    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


def test_cli_entrypoints_importable():
    from lebse.eval import build_parser as eval_parser
    from lebse.train import build_parser as train_parser

    assert eval_parser().prog == "lebse-eval"
    assert train_parser().prog == "lebse-train"
