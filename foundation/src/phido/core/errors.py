from __future__ import annotations


class StageError(Exception):
    """A stage could not produce a usable output.

    `code` follows the paper's outcome codes: EE (entity extraction),
    CS (component selection), PC (parameter configuration), SG (schematic generation).
    """

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
