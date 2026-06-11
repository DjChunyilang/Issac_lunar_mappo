from __future__ import annotations


def test_skrl_import_is_not_skipped() -> None:
    import skrl
    from skrl.multi_agents.torch.mappo import MAPPO

    assert getattr(skrl, "__version__", None)
    assert MAPPO.__name__ == "MAPPO"
