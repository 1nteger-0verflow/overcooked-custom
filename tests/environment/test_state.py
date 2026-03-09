"""Tests for environment.state.State.__str__ (L37-52)."""


class TestStateStr:
    """State.__str__ の文字列出力テスト."""

    def test_str_returns_string(self, minimal_state):
        assert isinstance(str(minimal_state), str)

    def test_str_contains_agent_info(self, minimal_state):
        assert "エージェント" in str(minimal_state)

    def test_str_contains_customer_info(self, minimal_state):
        assert "客席" in str(minimal_state)

    def test_str_contains_wait_line_info(self, minimal_state):
        assert "案内待ち" in str(minimal_state)

    def test_str_compact_returns_string(self, compact_state):
        assert isinstance(str(compact_state), str)

    def test_str_compact_contains_two_agents(self, compact_state):
        result = str(compact_state)
        assert "agent0" in result
        assert "agent1" in result
