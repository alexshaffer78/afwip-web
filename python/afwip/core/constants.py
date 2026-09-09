from enum import Enum, auto

class Side(Enum):
    US = "US"
    PRC = "PRC"

class Phase(Enum):
    ATO_SETUP = "ATO_SETUP" # Posture selection, squadron/enabler deployment
    BID_INITIATIVE = "BID_INITIATIVE"  # Sacrifice enablers, roll D4, resolve ties
    PLAY_INTEL = "PLAY_INTEL" # Roll to peek at opponent enabler cards
    PLAYER_TURN = "PLAYER_TURN"  # Alternating turns until both pass
    END_CLEANUP = "END_CLEANUP"  # Score VPs, reset board, check campaign end

class BandID(Enum):
    US_STANDOFF = "US_STANDOFF"
    US_AIRBASE = "US_AIRBASE"
    US_CONTINGENCY_LOCATION = "US_CONTINGENCY_LOCATION"
    BAND_A = "US_BAND_1, PRC_BAND_5"
    BAND_B = "US_BAND_2, PRC_BAND_4"
    BAND_C = "US_BAND_3, PRC_BAND_3"
    BAND_D = "US_BAND_4, PRC_BAND_2"
    BAND_E = "US_BAND_5, PRC_BAND_1"
    PRC_AIRBASE = "PRC_AIRBASE"
    PRC_STANDOFF = "PRC_STANDOFF"
    

class TokenType(Enum):
    # US Air 
    F_22 = "F-22"
    F_35A = "F-35A"
    F_15C = "F-15C"
    F_15E = "F-15E"
    F_16C = "F-16C"
    B_52 = "B-52"
    EC_130 = "EC-130"
    ATTACK_UAS_US = "Attack UAS (US)"
    RECON_UAS_US = "Recon UAS (US)"
    AEW_US = "AEW (US)"

    # US Naval
    DDG_81 = "DDG 81"
    DDG_115 = "DDG 115"

    # US Ground 
    ADA_US = "ADA (US)"

    # PRC Air
    J_20B = "J-20B"
    JH_7 = "JH-7"
    J_10 = "J-10"
    J_15 = "J-15"
    J_16 = "J-16"
    H_6K = "H-6K"
    ATTACK_UAS_PRC = "Attack UAS (PRC)"
    RECON_UAS_PRC = "Recon UAS (PRC)"
    AEW_PRC = "AEW (PRC)"

    # PRC Naval
    DALIAN_105 = "Dalian #105"
    NANNING_162 = "Nanning #162"
    FLOTILLA = "Missile Boat Flotilla"

    # PRC Ground 
    MID_RANGE_ADA_PRC = "Mid Range ADA (PRC)"
    LONG_RANGE_ADA_PRC = "Long Range ADA (PRC)"


class TokenOrigin(Enum):
    SQUADRON_CARD = "SQUADRON_CARD" # Winchester means returns to base
    ENABLER_CARD = "ENABLER_CARD" # Winchester means removed from board, no scoring

class TokenScoreType(Enum):
    SHIP = "SHIP"
    BOMBER = "BOMBER"
    ADA = "ADA"
    AEW = "AEW"
    UAS = "UAS"
    FIGHTER = "FIGHTER"
    EC_130 = "EC_130"

class CardType(Enum):
    SQUADRON = "SQUADRON"
    ENABLER = "ENABLER"
    MISSION = "MISSION"
    POSTURE = "POSTURE"

class PostureType(Enum):
    STANDARD = "STANDARD" # US, PRC
    
    # US Postures
    STANDOFF_US = "STANDOFF_US"
    SURGE = "SURGE"
    ACE = "ACE"
    HEDGEHOG = "HEDGEHOG"

    # PRC Postures 
    STANDOFF_PRC = "STANDOFF_PRC"
    ADA = "ADA"
    JOINT_OPERATIONS = "JOINT_OPERATIONS"
    PLAAF = "PLAAF"

class MissionType(Enum):
    INTERDICTION = "INTERDICTION" # US, PRC
    ECONOMY_OF_FORCE = "ECONOMY_OF_FORCE" # US, PRC
    ATTRITION = "ATTRITION" # US, PRC

    # US Missions 
    N_K_DOMINANCE = "N_K_DOMINANCE"
    RULE_OF_LAW = "ENFORCE_RULE_OF_LAW"

    # PRC Missions 
    THREE_DOMINANCES = "THREE_DOMINANCES"
    COUNTER_INTERVENTION = "COUNTER_INTERVENTION"

class IntelTrack(Enum):
    NORMAL = "NORMAL"
    ADVANTAGE = "ADVANTAGE"

class EnablerClass(Enum):
    # US, PRC
    MARITIME = "MARITIME" # Marines, Navy
    CYBER = "CYBER"
    SPACE = "SPACE"
    AIR_FORCE = "AIR_FORCE"

    # US
    BASES = "BASES" # e.g. resilient bases, HIMARS, infantry, red horse, land based missile defense
    ELECTRONIC_WARFARE = "ELECTRONIC_WARFARE"

    # PRC 
    SOF = "SOF"
    MISSILE = "MISSILE"
    NAVAL_AIR = "NAVAL_AIR"

class EnablerTrigger(Enum):
    ANYTIME = auto()
    OWN_AIRCRAFT_LOST = auto()
    OWN_SQUADRON_LOST_ALL_TOKENS = auto()
    OWN_AIR_TO_SURFACE_DECLARED = auto()
    # One of the owner's Squadron / Enabler Cards was just destroyed or
    # discarded — Rapid Resupply (18) must be played at that moment.
    OWN_CARD_DISCARDED = auto()
    OWN_UAS_ACQUIRE_ROLL = auto()
    OPP_PLAYS_SOF = auto()
    OPP_PLAYS_SPACE_CARD = auto()
    OPP_PLAYS_CYBER_CARD = auto()
    OPP_PLAYS_SUBMARINE_CARD = auto()
    OPP_PLAYS_MOBILITY_MAINT = auto()
    OPP_ATTACKS_BASE = auto()
    OPP_ROLLS_HIT = auto()
    OPP_DECLARES_MISSILE_DEFENSE = auto()


class ActionType(Enum):
    PASS = auto()  # End turn voluntarily
    ACTIVATE_SQUADRON = auto()  # Flip Squadron Card face up, generate tokens
    MOVE = auto()  # Move a single token up to its Move Range
    ACQUIRE = auto()  # Attempt to acquire an enemy token as target
    SHOOT = auto()  # Attempt to shoot an acquired or base target
    PLAY_ENABLER = auto()  # Play an Enabler Card from hand

class Visibility(Enum):
    VISIBLE = "VISIBLE" # Token acquired or card face-up; full info available
    HIDDEN = "HIDDEN" # Token unacquired (face-down); position known, capabilities unknown
    UNKNOWN = "UNKNOWN" # Opponent hand card; count known, identity unknown
    SEALED = "SEALED" # Mission cards only — revealed at game end only

DIE_SIDES = 4 # D4

class RollMode(Enum):
    """
    How a D4 is rolled. Advantage rolls two dice and keeps the higher;
    disadvantage keeps the lower. They cancel one-for-one (no stacking), so a
    single advantage plus a single disadvantage resolves to a NORMAL roll.
    """
    NORMAL = "NORMAL"
    ADVANTAGE = "ADVANTAGE"
    DISADVANTAGE = "DISADVANTAGE"

CYBER_RATE_MIN = 0
CYBER_RATE_START = 1
CYBER_RATE_WIN = 4 # Reaching 4 immediately ends game
MAX_CYBER_RATE = 4
 
DAMAGE_TO_DESTROY_TOKEN = 1 # Single hit destroys a token
DAMAGE_TO_DESTROY_SQUADRON = 2 # Two hits to destroy a Squadron Card
INFANTRY_DAMAGE_BOXES = 2      # Infantry Battalion card: two printed damage boxes
AIRBASE_BONUS_DAMAGE_BOXES = 3 # Extra damage boxes on airbase (VP targets only)
 
MAX_ENABLER_CARDS_DEFAULT   = 5 # From PLAAF posture card example
MAX_SQUADRON_CARDS_DEFAULT  = 6 # From PLAAF posture card example

WINCHESTER_ROLL_STANDARD   = 4  # must roll 4 to avoid Winchester
WINCHESTER_ROLL_UAS_ATTACK = 3  # must roll 3+ to avoid Winchester
 
# ATO Cycles per campaign (keyed by campaign number)
CAMPAIGN_ATO_CYCLES = {
    1: 1,
    2: 2,
    3: 5,
    4: 2,
    5: 2,
}

# Cyber access values 
CYBER_ACCESS_VALUES: dict[int, int] = {
    0: 2, # roll needed to go from rate 0 to 1
    1: 3, # roll needed to go from rate 1 to 2
    2: 4,  # roll needed to go from rate 2 to 3
    3: 4,  # roll needed to go from rate 3 to 4
}