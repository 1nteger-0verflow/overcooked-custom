from environment.actions import Actions
from operation.agent_controller import AgentController

ACTION_MAPPING = {
    "left": Actions.LEFT,
    "right": Actions.RIGHT,
    "up": Actions.UP,
    "down": Actions.DOWN,
    " ": Actions.INTERACT,
    "tab": Actions.STAY,
}


class KeyboardInput(AgentController):
    def __init__(self, agent_id: int, verbose: bool, confirm: bool):
        self.agent_id = agent_id
        self.verbose = verbose
        self.confirm = confirm
        self.done = False
        self.reset()

    @property
    def is_auto(self) -> bool:
        return False

    @property
    def is_done(self) -> bool:
        return self.done

    def reset(self):
        self.done = False
        self.action = Actions.STAY

    def input_key(self, key: str) -> bool:
        if self.done:
            return False
        match key:
            case key if key in ACTION_MAPPING:
                self.action = ACTION_MAPPING[key]
            case key if key in [str(x) for x in range(1, 10)]:
                self.action = Actions.PICK_PLACE_BASE + int(key) - 1
            case key:
                return False
        if not self.confirm:
            self.done = True
        return True

    def input_observation(self, obs):
        self.reset()

    def get_action(self):
        return self.action
