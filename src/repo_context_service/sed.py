"""In-place `sed` edits, applied to file text the way GNU sed applies them.

`sed_edit(words)` reads the arguments of a `sed` command. It returns None when the command does
not edit in place, and otherwise a `SedEdit` of the files it edits and, when every command is a
form this module reproduces, those commands: `s/pattern/replacement/[g]`, `d`, and `a`/`i`/`c`
text, each with an optional line address or range (`3`, `$`, `2,5`). Like sed, `apply` runs
every command in order on each input line in turn, so line addresses always name lines of the
original file.

A pattern or replacement whose meaning differs between GNU sed and Python's regular expressions
(`\\<`, `\\d`, `\\U`, alternation, a global match of the empty string, a character class on
non-ASCII text, ...) is not reproduced, and neither is a substitution on a line an earlier
substitution split: the edit is then changed-but-unknown rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .command_search import MAX_TEXT_CHARS, bre_to_python, ere_to_python
from .shell import Word

_ADDRESS = re.compile(r"(\d+|\$)(?:,(\d+|\$))?")
# the only escapes whose meaning GNU sed and Python (ASCII mode) agree on inside a pattern
_PORTABLE_ESCAPE = set(".*[]^$/\\+?(){}tnsSwWbB<>")
_REGEX_SPECIAL = set(".*[]^$\\+?(){}|")
# a pattern that only anchors: it matches the empty string the same way in both engines
_ANCHORS_ONLY = {"^", "$", "^$"}
_REPLACEMENT_ESCAPE = set("nt&/\\")
# the flags GNU sed accepts after `s///` (`w` takes the rest of the line as a file name)
_S_FLAGS = set("gpiImMe0123456789")
# the last line of the file, as an address
_LAST = -1


@dataclass(frozen=True)
class _Substitute:
    regex: re.Pattern
    replacement: str
    every: bool
    # the same pattern over text with each UTF-8 byte as one character, as the C locale reads
    bytewise: re.Pattern
    # the pattern as written
    source: str

    def apply(self, line: str) -> str | None:
        """The line after the substitution; None when it depends on the locale, which the
        sandbox's is not known: when a character and a byte at a time give different lines."""
        count = 0 if self.every else 1
        # every match (at most one more than the line has characters) may add the replacement
        matches = len(line) + 1 if self.every else 1
        if len(line) + matches * len(self.replacement) > MAX_TEXT_CHARS:
            return None
        out = self.regex.sub(lambda _: self.replacement, line, count=count)
        if not line.isascii():
            if "[:" in self.source:
                return None  # a named class takes in the locale's letters, which only it knows
            view = _byte_view(line)
            replacement = _byte_view(self.replacement)
            bytewise = self.bytewise.sub(lambda _: replacement, view, count=count)
            if bytewise.encode("latin-1").decode("utf-8", "surrogateescape") != out:
                return None
        return out


def _byte_view(text: str) -> str:
    return text.encode("utf-8", "surrogateescape").decode("latin-1")


@dataclass(frozen=True)
class _Text:
    """`a` (append after the line), `i` (insert before it) or `c` (replace it)."""

    verb: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _Command:
    # the addressed lines: first and last (both None for every line, last None for one line)
    first: int | None
    last: int | None
    action: _Substitute | _Text | None  # None deletes the line


class _Rejected(Exception):
    """A script sed refuses (`unknown option to s`, line 0, an unterminated command): it exits 1
    before opening any file."""


@dataclass(frozen=True)
class SedEdit:
    # None when a command is a form this module does not reproduce
    commands: tuple[_Command, ...] | None
    files: tuple[Word, ...]
    # the suffix of the backup copy `-i.bak` keeps, or ""
    backup: str = ""
    # the script is one sed refuses, so nothing is written, not even the backup
    rejected: bool = False

    def apply(self, text: str) -> str | None:
        """`text` after the edit, or None when a line makes it one this does not reproduce."""
        lines = text.split("\n")
        terminated = lines[-1] == ""
        if terminated:
            lines.pop()
        out: list[str] = []
        size = 0
        # whether the last line printed is the file's last line, still without its newline
        open_end = False
        for number, line in enumerate(lines, 1):
            last_line = number == len(lines)
            printed, appended, keep, ended = [], [], True, False
            for command in self.commands:
                if not _addresses(command, number, len(lines)):
                    continue
                action = command.action
                if isinstance(action, _Substitute):
                    line = action.apply(line) if "\n" not in line else None
                    if line is None:
                        return None
                elif action is None:
                    keep = False
                    break
                elif action.verb == "a":
                    # even with no text, appending ends the line before it
                    appended += action.lines
                    ended = True
                elif action.verb == "i":
                    printed += action.lines
                else:
                    # `c` prints its text once, at the end of its range, and ends the cycle
                    if command.last is None or _line(command.last, len(lines)) <= number:
                        printed += action.lines
                    keep = False
                    break
            out += printed + ([line] if keep else []) + appended
            size += sum(map(len, printed)) + len(line) + sum(map(len, appended))
            if size > MAX_TEXT_CHARS:
                return None
            open_end = last_line and keep and not ended and not terminated
        return "\n".join(out) + ("" if open_end or not out else "\n")


def sed_edit(words: tuple[Word, ...]) -> SedEdit | None:
    """How `sed ...` (its words after `sed`) edits files in place, if it does."""
    scripts: list[Word] = []
    operands: list[Word] = []
    in_place, extended, modelled = False, False, True
    backup = ""
    script_from_option = False
    index = 0
    while index < len(words):
        word = words[index]
        token = word.text
        index += 1
        if token == "--":
            operands += words[index:]
            break
        if token.startswith("--"):
            name, equals, value = token.partition("=")
            if name == "--in-place":
                in_place, backup = True, value
            elif name == "--expression":
                if not equals and index < len(words):
                    word, index = words[index], index + 1
                    value = word.text
                scripts.append(Word(value, word.expands))
                script_from_option = True
            elif name == "--regexp-extended":
                extended = True
            else:
                modelled = False
        elif token.startswith("-") and len(token) > 1:
            letters = token[1:]
            while letters:
                letter, letters = letters[0], letters[1:]
                if letter == "i":
                    # GNU reads everything after -i as the backup suffix: `-ie` keeps "e"
                    in_place, backup, letters = True, letters, ""
                elif letter in "Er":
                    extended = True
                elif letter in "ef":
                    if letters:
                        value = Word(letters, word.expands)
                    else:
                        value = words[index] if index < len(words) else Word("")
                        index += 1
                    if letter == "e":
                        scripts.append(value)
                    else:
                        modelled = False
                    script_from_option = True
                    letters = ""
                else:
                    # -n prints only what the script asks for, -s/-z change how input is split
                    modelled = False
        else:
            operands.append(word)
    if not in_place:
        return None
    if not script_from_option and operands:
        scripts, operands = [operands[0]], operands[1:]
    if not operands:
        return None
    commands = None
    if modelled and scripts and not any(script.expands for script in scripts):
        # GNU joins the scripts of several `-e` into one program, a line each
        try:
            commands = parse_program("\n".join(script.text for script in scripts), extended)
        except _Rejected:
            return SedEdit(None, tuple(operands), backup, rejected=True)
    return SedEdit(None if commands is None else tuple(commands), tuple(operands), backup)


def parse_program(program: str, extended: bool = False) -> list[_Command] | None:
    """The commands of a sed program, or None when one is a form this does not reproduce."""
    commands: list[_Command] = []
    index = 0
    while True:
        while index < len(program) and program[index] in " \t\n;":
            index += 1
        if index >= len(program):
            return commands
        first = last = None
        if address := _ADDRESS.match(program, index):
            first, last = _number(address.group(1)), address.group(2)
            last = None if last is None else _number(last)
            if first == 0 or last == 0:
                raise _Rejected("line 0")
            index = address.end()
            while index < len(program) and program[index] in " \t":
                index += 1
        if index >= len(program):
            return None
        verb = program[index]
        index += 1
        if verb == "s":
            parsed = _substitution(program, index, extended)
            if parsed is None:
                return None
            action, index = parsed
        elif verb == "d":
            action = None
        elif verb in "aic":
            placed = _placed_text(program, index, verb)
            if placed is None:
                raise _Rejected(f"{verb} without text")
            action, index = placed
        else:
            return None
        commands.append(_Command(first, last, action))
        if not isinstance(action, _Text):
            # a command ends at `;`, a newline or the end of the program
            while index < len(program) and program[index] in " \t":
                index += 1
            if index < len(program) and program[index] not in ";\n":
                return None


def _substitution(program: str, index: int, extended: bool) -> tuple[_Substitute, int] | None:
    """Parse `s<D>pattern<D>replacement<D>[g]` from just after the `s`."""
    if index >= len(program) or program[index] in "\n\\":
        raise _Rejected("unterminated s")
    delimiter = program[index]
    index += 1
    fields: list[str] = []
    current: list[str] = []
    while len(fields) < 2:
        if index >= len(program):
            raise _Rejected("unterminated s")
        char = program[index]
        if char == "\\" and index + 1 < len(program):
            escaped = program[index + 1]
            if escaped == delimiter:
                if delimiter in _REGEX_SPECIAL and not fields:
                    # whether the escaped delimiter is literal or an operator differs by version
                    return None
                current.append(delimiter)
            else:
                current.append(program[index : index + 2])
            index += 2
            continue
        if char == delimiter:
            fields.append("".join(current))
            current = []
        elif char == "\n" and not fields:
            raise _Rejected("unterminated s")
        else:
            current.append(char)
        index += 1
    flags_start = index
    while index < len(program) and program[index] not in ";\n} \t":
        index += 1
    flags = program[flags_start:index]
    pattern, replacement = fields
    if "w" not in flags and set(flags) - _S_FLAGS:
        raise _Rejected("unknown option to s")
    if flags not in ("", "g") or not _replacement_portable(replacement):
        return None
    regex = _pattern_regex(pattern, extended, flags == "g")
    if regex is None:
        return None
    bytewise = re.compile(_byte_view(regex.pattern), re.ASCII)
    return _Substitute(regex, _unescape(replacement), flags == "g", bytewise, pattern), index


def _placed_text(program: str, index: int, verb: str) -> tuple[_Text, int] | None:
    """The text of `a`, `i` or `c`, from just after the letter: up to the end of the line, where
    a backslash before a line break continues it. Whitespace after the letter is dropped, and so
    is one backslash after that (`a\\`), which keeps the whitespace following it; escapes in the
    text are resolved as in a replacement, and a lone backslash ending the program is dropped.
    After `a\\` with nothing following, there is no text at all. None when neither a backslash
    nor text follows the letter, which sed rejects."""
    while index < len(program) and program[index] in " \t":
        index += 1
    marked = index < len(program) and program[index] == "\\"
    if marked:
        index += 1
        if index < len(program) and program[index] == "\n":
            index += 1
    start = index
    while index < len(program) and program[index] != "\n":
        index += 2 if program[index] == "\\" else 1
    index = min(index, len(program))
    raw = program[start:index]
    if not raw:
        return (_Text(verb, ()), index) if marked else None
    if (len(raw) - len(raw.rstrip("\\"))) % 2:
        raw = raw[:-1]
    return _Text(verb, tuple(_unescape(raw).split("\n"))), index


def _pattern_regex(pattern: str, extended: bool, every_match: bool) -> re.Pattern | None:
    """The pattern as a Python regex, when both engines read it the same way. Alternation is
    not: sed takes the longest alternative that matches, Python the first. Nor, when every match
    is replaced (`every_match`, the `g` flag), is a pattern that can match the empty string: the
    engines disagree on where such a match sits next to a non-empty one."""
    index = 0
    in_brackets = False
    while index < len(pattern):
        char = pattern[index]
        if in_brackets:
            if char == "]":
                in_brackets = False
        elif char == "[":
            in_brackets = True
            if pattern.startswith("[]", index) or pattern.startswith("[^]", index):
                # a `]` right after the opening is a member, not the end
                index += 2 if pattern.startswith("[]", index) else 3
                continue
        elif char == "\\":
            if index + 1 >= len(pattern) or pattern[index + 1] not in _PORTABLE_ESCAPE:
                return None
            index += 1
        elif char == "|" and extended:
            return None
        index += 1
    translated = (ere_to_python if extended else bre_to_python)(pattern, sed=True)
    if translated is None:
        return None
    try:
        regex = re.compile(translated, re.ASCII)
    except re.error:
        return None
    if every_match and regex.match("") and pattern not in _ANCHORS_ONLY:
        return None
    return regex


def _replacement_portable(replacement: str) -> bool:
    """A replacement without `&` or `\\N` (the match and its groups, which this does not
    reproduce) and without escapes whose meaning differs between versions: an escaped letter
    or digit. An escaped punctuation mark stands for itself in every version."""
    index = 0
    while index < len(replacement):
        char = replacement[index]
        if char == "&":
            return False
        if char == "\\":
            following = replacement[index + 1 : index + 2]
            portable = following in _REPLACEMENT_ESCAPE or not (
                following.isalnum() or following == "_"
            )
            if not following or not portable:
                return False
            index += 1
        index += 1
    return True


def _addresses(command: _Command, number: int, count: int) -> bool:
    """Whether a command applies to line `number` of `count`. A range whose end comes before its
    start covers only its first line, as in sed."""
    if command.first is None:
        return True
    first = _line(command.first, count)
    if command.last is None:
        return number == first
    return first <= number <= max(first, _line(command.last, count))


def _line(address: int, count: int) -> int:
    return count if address == _LAST else address


def _number(text: str) -> int:
    return _LAST if text == "$" else int(text)


def _unescape(text: str) -> str:
    """A replacement with its escapes resolved: `\\n` and `\\t`, and any other escaped
    character standing for itself (an escaped line break for a line break)."""
    out, index = [], 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            out.append({"n": "\n", "t": "\t"}.get(text[index + 1], text[index + 1]))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)
