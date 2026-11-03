"""Parameter configuration: map node params onto the chosen cell's parameters, with range checks."""
from __future__ import annotations

import re

from ...core.schemas import Issue, Scalar, TemplateNode
from ...knowledge.pdk import Cell, ParamSpec

_UNIT_SUFFIX = re.compile(r"_(um|nm|mm|ghz|mhz|khz|hz|db|deg|rad)$")


def norm_key(key: str) -> str:
    k = key.lower().strip().replace("-", "_").replace(" ", "_")
    return _UNIT_SUFFIX.sub("", k)


def _coerce(value: Scalar, spec: ParamSpec) -> Scalar:
    default = spec.default
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        raise ValueError("expected a boolean")
    if isinstance(default, (int, float)):
        if isinstance(value, bool):
            raise ValueError("expected a number")
        v = float(value)  # may raise ValueError for non-numeric strings
        return int(v) if isinstance(default, int) and v.is_integer() else v
    return str(value)


class ParamConfigurator:
    def __init__(self, property_keys: set[str]):
        self.property_keys = property_keys

    def configure(self, node: TemplateNode, cell: Cell) -> tuple[dict[str, Scalar], list[Issue]]:
        lookup: dict[str, str] = {}
        for pname, spec in cell.params.items():
            for alias in (pname, *spec.aliases):
                lookup[norm_key(alias)] = pname
        out: dict[str, Scalar] = {}
        issues: list[Issue] = []
        for key, value in node.params.items():
            k = key.lower()
            if k in self.property_keys:  # capability requirement, e.g. modulation_bandwidth_ghz
                have = cell.properties.get(k)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and (
                    not isinstance(have, (int, float)) or have < value
                ):
                    issues.append(Issue(code="PC", node=node.id,
                                        message=f"{cell.name} supports {k}={have}, requested {value}"))
                continue
            pname = lookup.get(norm_key(key))
            if pname is None:
                issues.append(Issue(code="PC", node=node.id,
                                    message=f"parameter '{key}' does not exist on {cell.name} "
                                            f"(has: {', '.join(cell.params) or 'none'})"))
                continue
            spec = cell.params[pname]
            try:
                v = _coerce(value, spec)
            except (TypeError, ValueError) as e:
                issues.append(Issue(code="PC", node=node.id, message=f"{pname}={value!r}: {e}"))
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if (spec.min is not None and v < spec.min) or (spec.max is not None and v > spec.max):
                    issues.append(Issue(code="PC", node=node.id,
                                        message=f"{pname}={v} outside allowed range [{spec.min}, {spec.max}] {spec.unit}"))
                    continue
            out[pname] = v
        return out, issues
