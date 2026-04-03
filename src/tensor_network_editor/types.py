from __future__ import annotations

from os import PathLike
from typing import TypeAlias

JSONPrimitive: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
MetadataDict: TypeAlias = dict[str, JSONValue]
StrPath: TypeAlias = str | PathLike[str]
