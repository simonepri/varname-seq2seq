import re
from typing import *


def multiple_replace(dict: Dict[str, str], text: str) -> str:
    if not len(dict) or not len(text):
        return text
    # Create a regular expression from the dictionary keys
    regex_str = "(%s)" % "|".join(map(re.escape, dict.keys()))

    regex = re.compile(regex_str)
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start() : mo.end()]], text)


def truncate(text: str, length: int, ellipsis: Optional[str] = None) -> str:
    if length < 0:
        if ellipsis is None:
            return text[length:]
        else:
            return ellipsis + text[length + len(ellipsis) :]
    if ellipsis is None:
        return text[:length]
    else:
        return text[: length - len(ellipsis) :] + ellipsis


def rreplace(suffix: str, sub: str, string: str) -> str:
    return string[:-len(suffix)] + sub if string.endswith(suffix) else string


def lreplace(prefix: str, sub: str, string: str) -> str:
    return sub + string[len(prefix):] if string.startswith(prefix) else string
