"""Tests for environment.observation.Observer: print_layer_info (L137-143)."""


class TestObserverPrintLayerInfo:
    """Observer.print_layer_info の出力テスト."""

    def test_print_layer_info_runs_without_error(self, minimal_env):
        # 例外なく実行できることを確認
        minimal_env.observer.print_layer_info()

    def test_print_layer_info_outputs_text(self, minimal_env, capsys):
        minimal_env.observer.print_layer_info()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_layer_info_contains_static_label(self, minimal_env, capsys):
        minimal_env.observer.print_layer_info()
        captured = capsys.readouterr()
        assert "WALL" in captured.out

    def test_print_layer_info_compact_env(self, compact_env, capsys):
        compact_env.observer.print_layer_info()
        captured = capsys.readouterr()
        assert len(captured.out) > 0
