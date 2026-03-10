"""Tests for operation.controller_factory and operation.controller."""

import pytest
from omegaconf import OmegaConf

from operation.agent_controller import AgentController
from operation.controller import Controller
from operation.controller_factory import create_controller
from operation.keyboard_input import KeyboardInput
from operation.random_input import RandomInput


def _op(kind: str, **kwargs):
    # controller_factory は dict の最初のキーをコントローラタイプとして使う
    return OmegaConf.create({kind: kwargs or {}})


class TestCreateControllerRouting:
    def test_keyboard_type(self):
        ctrl = create_controller(_op("keyboard"), agent_id=0, num_actions=9, verbose=False, confirm=False)
        assert isinstance(ctrl, KeyboardInput)

    def test_random_type(self):
        ctrl = create_controller(_op("random"), agent_id=0, num_actions=9, verbose=False, confirm=False)
        assert isinstance(ctrl, RandomInput)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown operation type"):
            create_controller(_op("nonexistent"), agent_id=0, num_actions=9, verbose=False, confirm=False)

    def test_returns_agent_controller_subclass(self):
        ctrl = create_controller(_op("random"), agent_id=0, num_actions=9, verbose=False, confirm=False)
        assert isinstance(ctrl, AgentController)


class TestAgentControllerClassMethod:
    def test_create_via_class_method(self):
        op = _op("keyboard")
        ctrl = AgentController.create_controller(op, agent_id=0, num_actions=9, verbose=False, confirm=False)
        assert isinstance(ctrl, KeyboardInput)

    def test_random_via_class_method(self):
        op = _op("random")
        ctrl = AgentController.create_controller(op, agent_id=1, num_actions=6, verbose=False, confirm=False)
        assert isinstance(ctrl, RandomInput)
        assert ctrl.get_action() in range(6)


class TestControllerIntegration:
    """Controller (複数エージェント統合) の基本テスト."""

    @pytest.fixture
    def ui(self):
        return {"random": None, "keyboard": None}

    @pytest.fixture
    def player(self):
        return ["random"]

    def test_operate_returns_actions(self, minimal_env, ui, player):
        ctrl = Controller(minimal_env, ui, player, verbose=False, confirm=False)
        actions = ctrl.operate()
        assert len(actions) == minimal_env.num_agents

    def test_is_auto_true_for_all_random(self, minimal_env, ui, player):
        ctrl = Controller(minimal_env, ui, player, verbose=False, confirm=False)
        assert ctrl.is_auto()
