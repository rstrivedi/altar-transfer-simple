# Added by RST: Configuration for scripted resident agents
"""Tunables for resident behavior in normative allelopathic harvest.

Residents enforce the posted rule, harvest/replant permitted color, and patrol.
All parameters must be identical across control and treatment conditions.
"""

# Zap mechanics (must match substrate configuration)
ZAP_RANGE = 3  # Beam length from substrate (beamLength in ColorZapper)
ZAP_COOLDOWN = 4  # Frames between zaps (cooldownTime in Zapper component)

# Harvest and plant mechanics
HARVEST_RADIUS = 3  # Local search radius for ripe berries
HARVEST_FREQUENCY = 2  # Policy frequency control: only try to harvest every N steps (creates gaps for zapping)
PLANT_FREQUENCY = 2  # Policy frequency control: only try to plant every N steps (creates gaps for zapping)

# Patrol behavior
PATROL_PERSISTENCE = 8  # Frames to continue in same direction before changing
PATROL_DIRECTIONS = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT']  # Available patrol actions

# Tiebreak policy
CHOICE_TIEBREAK = "nearest_then_lowest_id"  # When multiple targets eligible, pick nearest; tie -> lowest agent_id

# Reproducibility
DEFAULT_SEED = 42  # Default seed for patrol randomness

# Action indices (match allelopathic_harvest ACTION_SET)
# (NOOP, FORWARD, BACKWARD, STEP_LEFT, STEP_RIGHT, TURN_LEFT, TURN_RIGHT, FIRE_ZAP, FIRE_ONE, FIRE_TWO, FIRE_THREE)
ACTION_NOOP = 0
ACTION_FORWARD = 1
ACTION_BACKWARD = 2  # FIXED: Was 3
ACTION_STEP_LEFT = 3  # FIXED: Was 4
ACTION_STEP_RIGHT = 4  # FIXED: Was 2
ACTION_TURN_LEFT = 5
ACTION_TURN_RIGHT = 6
ACTION_FIRE_ZAP = 7
ACTION_PLANT_RED = 8  # FIRE_ONE
ACTION_PLANT_GREEN = 9  # FIRE_TWO
ACTION_PLANT_BLUE = 10  # FIRE_THREE

# Color indices (match Phase 1: 1=RED, 2=GREEN, 3=BLUE, 0=GREY)
COLOR_GREY = 0
COLOR_RED = 1
COLOR_GREEN = 2
COLOR_BLUE = 3

# Map permitted color index to plant action
PLANT_ACTION_MAP = {
    COLOR_RED: ACTION_PLANT_RED,
    COLOR_GREEN: ACTION_PLANT_GREEN,
    COLOR_BLUE: ACTION_PLANT_BLUE,
}
