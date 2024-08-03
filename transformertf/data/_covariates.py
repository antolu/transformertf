import typing

TIME_PREFIX = "__time__"
KNOWN_COV_CONT_PREFIX = "__known_continuous__"
PAST_KNOWN_COV_PREFIX = "__past_known_continuous__"
TARGET_PREFIX = "__target__"


def known_cov_col(name: str) -> str:
    return f"{KNOWN_COV_CONT_PREFIX}{name}"


def past_known_cov_col(name: str) -> str:
    return f"{PAST_KNOWN_COV_PREFIX}{name}"


def time_col(name: str) -> str:
    return f"{TIME_PREFIX}{name}"


def target_col(name: str) -> str:
    return f"{TARGET_PREFIX}{name}"


class Covariate(typing.NamedTuple):
    name: str
    col: str
