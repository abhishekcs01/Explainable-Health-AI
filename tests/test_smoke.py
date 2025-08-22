
def test_imports():
    import xai_health
    from xai_health import config, data, features, model, explain, recommendations, ui, utils
    assert hasattr(config, "DATA_PATH")
