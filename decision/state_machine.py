# decision/state_machine.py
from enum import Enum, auto

class FSMState(Enum):
    FOLLOW_LANE = auto()
    PREPARE_LANE_CHANGE_LEFT = auto()
    LANE_CHANGE_LEFT = auto()
    OVERTAKE_CRUISE = auto()
    PREPARE_RETURN_RIGHT = auto()
    LANE_CHANGE_RIGHT = auto()
    EMERGENCY_BRAKE = auto()

LEFT_CHANGE_STATES = {
    FSMState.PREPARE_LANE_CHANGE_LEFT,
    FSMState.LANE_CHANGE_LEFT,
}

RIGHT_CHANGE_STATES = {
    FSMState.PREPARE_RETURN_RIGHT,
    FSMState.LANE_CHANGE_RIGHT,
}

LANE_CHANGE_STATES = LEFT_CHANGE_STATES | RIGHT_CHANGE_STATES

OVERTAKE_STATES = {
    FSMState.PREPARE_LANE_CHANGE_LEFT,
    FSMState.LANE_CHANGE_LEFT,
    FSMState.OVERTAKE_CRUISE,
    FSMState.PREPARE_RETURN_RIGHT,
    FSMState.LANE_CHANGE_RIGHT,
}

def is_left_change_state(state):
    return state in LEFT_CHANGE_STATES

def is_right_change_state(state):
    return state in RIGHT_CHANGE_STATES

def is_lane_change_state(state):
    return state in LANE_CHANGE_STATES

def is_overtake_state(state):
    return state in OVERTAKE_STATES

def state_name(state):
    if isinstance(state, FSMState):
        return state.name
    return str(state)
