import re
from typing import *  # pylint: disable=W0401,W0614


def multiple_replace(text_map: Dict[str, str], text: str) -> str:
    if len(text_map) == 0 or len(text) == 0:
        return text
    # Create a regular expression from the dictionary keys
    regex_str = "(%s)" % "|".join(map(re.escape, text_map.keys()))

    regex = re.compile(regex_str)
    # For each match, look-up corresponding value in dictionary
    return regex.sub(
        lambda mo: text_map[mo.string[mo.start() : mo.end()]], text
    )


def truncate(text: str, length: int, ellipsis: Optional[str] = None) -> str:
    if length < 0:
        if ellipsis is None:
            return text[length:]
        return ellipsis + text[length + len(ellipsis) :]
    if ellipsis is None:
        return text[:length]
    return text[: length - len(ellipsis) :] + ellipsis


def rreplace(suffix: str, sub: str, string: str) -> str:
    return string[: -len(suffix)] + sub if string.endswith(suffix) else string


def lreplace(prefix: str, sub: str, string: str) -> str:
    return sub + string[len(prefix) :] if string.startswith(prefix) else string
