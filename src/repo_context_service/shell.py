"""A parser for the shell commands agents run, precise enough to execute them against a repository.

`parse_command(text)` turns one command (the contents of a bash code block) into a
`ParsedCommand`: stages joined by `&&`, `||` or `;` (a newline counts as `;`), each stage a
pipeline of `SimpleCommand`s with their words, redirects and heredoc. Quoting, escapes, comments,
line continuations and heredoc bodies are resolved here, once, so nothing downstream has to scan
raw command text.

Anything whose meaning cannot be read off the text is refused with `Unsupported(reason)`:
subshells, command substitution, compound commands (`if`, `for`, `while`, ...), background jobs,
here-strings, process substitution, and syntax errors bash itself would reject. Callers treat an
unsupported command as one they cannot run exactly. A carriage return is read as a blank, so a
command pasted with Windows line endings parses like its Unix form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# what the shell expands inside an unquoted heredoc: parameters, substitutions, escapes
_HEREDOC_EXPANSION = re.compile(r"\$[\w{(@*#?!$-]|`|\\[$`\\\n]")

AND = "&&"
OR = "||"
SEQUENCE = ";"

# words that open a compound command when they stand where a command name would
_RESERVED_WORDS = frozenset(
    {
        "if", "then", "elif", "else", "fi", "for", "while", "until", "do", "done", "case",
        "esac", "function", "select", "{", "}", "!", "[[", "]]", "coproc",
    }
)  # fmt: skip
_WORD_END = " \t\r|&;<>()\n"
_GLOB_CHARS = "*?["
# the character after `$` that makes it a parameter expansion rather than a literal dollar sign
_PARAMETER_MARKS = "_{@*#?!$-"
# longest first, so `>>` is not read as `>`
_REDIRECT_OPERATORS = (
    (">>", "append"),
    (">&", "duplicate"),
    ("<&", "duplicate"),
    (">|", "write"),
    (">", "write"),
    ("<", "read"),
)
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\+?=")


@dataclass(frozen=True)
class Word:
    """One shell word after quote removal.

    `expands` marks a word whose value depends on the shell's state - a `$VAR` outside single
    quotes, or a leading `~` - so its text is not the value the command receives. `globs` marks
    an unquoted `*`, `?` or `[`, which the shell replaces with the matching paths. `braces` marks
    an unquoted brace list or sequence (`a.py{,.bak}`, `f{1..3}`), which the shell replaces with
    several words before anything else.
    """

    text: str
    expands: bool = False
    globs: bool = False
    braces: bool = False


@dataclass(frozen=True)
class Redirect:
    """A redirection of file descriptor `fd`.

    `mode` is "write" (`>`), "append" (`>>`), "read" (`<`) or "duplicate" (`2>&1`, where the
    target is the descriptor copied, or "-" when the descriptor is closed).
    """

    fd: int
    mode: str
    target: Word

    @property
    def writes_file(self) -> bool:
        return self.mode in ("write", "append") and self.target.text != "/dev/null"


@dataclass(frozen=True)
class Heredoc:
    """The text a heredoc feeds to its command's standard input.

    `body` is exactly what the command reads: every line up to the delimiter line, each ending in
    a newline. With an unquoted delimiter (`expands`) the shell processes `$`, `$(...)`, backticks
    and backslash-newline inside the body, so the text shown is not literally what is read.
    """

    delimiter: str
    body: str
    expands: bool
    # False when the command ended before the delimiter line: bash then takes the rest as the body
    terminated: bool = True

    @property
    def literal(self) -> str | None:
        """The body as the command reads it, or None when the shell would expand parts of it."""
        if self.expands and _HEREDOC_EXPANSION.search(self.body):
            return None
        return self.body


@dataclass(frozen=True)
class SimpleCommand:
    words: tuple[Word, ...]
    assignments: tuple[str, ...] = ()
    redirects: tuple[Redirect, ...] = ()
    heredoc: Heredoc | None = None
    # the command's source text, without its heredoc body
    text: str = ""

    @property
    def argv(self) -> tuple[str, ...]:
        return tuple(word.text for word in self.words)

    @property
    def name(self) -> str:
        """The program run, without its directory: `/usr/bin/python3 x.py` runs `python3`."""
        return self.words[0].text.rsplit("/", 1)[-1] if self.words else ""

    @property
    def args(self) -> tuple[str, ...]:
        return self.argv[1:]

    @property
    def expands(self) -> bool:
        """Whether a word or redirect target depends on shell state the text does not show."""
        return any(word.expands for word in self.words) or any(
            redirect.target.expands for redirect in self.redirects
        )


@dataclass(frozen=True)
class Stage:
    """One element of the command list: a pipeline, and the separator that joins it to the stage
    before it (empty for the first stage)."""

    separator: str
    pipeline: tuple[SimpleCommand, ...]
    # the stage's source text, as a script would hold it: its heredoc bodies follow it
    text: str

    @property
    def command(self) -> SimpleCommand:
        """The first command of the pipeline, which decides what kind of stage this is."""
        return self.pipeline[0]


@dataclass(frozen=True)
class ParsedCommand:
    stages: tuple[Stage, ...]
    text: str


@dataclass(frozen=True)
class Unsupported:
    reason: str
    text: str


def parse_command(text: str) -> ParsedCommand | Unsupported:
    try:
        return _Parser(text).parse()
    except _Refused as refused:
        return Unsupported(refused.reason, text)


class _Refused(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class _CommandDraft:
    start: int
    end: int
    words: list[Word] = field(default_factory=list)
    assignments: list[str] = field(default_factory=list)
    redirects: list[Redirect] = field(default_factory=list)
    heredoc: Heredoc | None = None
    # where the heredoc body sits in the source: text after the stage, unless a pipe continued it
    heredoc_start: int = 0
    heredoc_end: int = 0
    has_pending_heredoc: bool = False

    def is_empty(self) -> bool:
        return not (self.words or self.assignments or self.redirects or self.has_pending_heredoc)


@dataclass
class _StageDraft:
    separator: str
    commands: list[_CommandDraft]


@dataclass
class _PendingHeredoc:
    owner: _CommandDraft
    delimiter: str
    strip_tabs: bool
    expands: bool


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.position = 0
        self.stages: list[_StageDraft] = []
        self.pipeline: list[_CommandDraft] = []
        self.command: _CommandDraft | None = None
        self.separator = ""
        # heredocs opened on the current line, whose bodies start after its newline
        self.pending: list[_PendingHeredoc] = []

    def parse(self) -> ParsedCommand:
        while self.position < len(self.text):
            char, following = self.text[self.position], self._peek(1)
            if char in " \t\r":
                self.position += 1
            elif char == "\\" and following == "\n":
                self.position += 2
            elif char == "#" and self._at_word_start():
                self._skip_comment()
            elif char == "\n":
                self._end_stage(SEQUENCE, explicit=False)
                self._consume_newline()
            elif char in "<>" or (char == "&" and following == ">"):
                self._redirect(fd=None)
            elif char in "|&;()":
                self._list_operator()
            else:
                self._word()
        self._end_stage("", explicit=False)
        self._read_heredoc_bodies()
        return ParsedCommand(tuple(self._build_stage(draft) for draft in self.stages), self.text)

    def _peek(self, offset: int = 0) -> str:
        index = self.position + offset
        return self.text[index] if index < len(self.text) else ""

    def _at_word_start(self) -> bool:
        return self.position == 0 or self.text[self.position - 1] in _WORD_END

    def _skip_comment(self) -> None:
        end = self.text.find("\n", self.position)
        self.position = len(self.text) if end == -1 else end

    def _skip_spaces(self) -> None:
        while self._peek() in (" ", "\t"):
            self.position += 1

    def _consume_newline(self) -> None:
        self.position += 1
        self._read_heredoc_bodies()

    def _continue_on_next_lines(self) -> None:
        """After `&&`, `||` or `|` the list goes on, however many blank or comment lines follow."""
        while self.position < len(self.text):
            char = self.text[self.position]
            if char in " \t\r":
                self.position += 1
            elif char == "\n":
                self._consume_newline()
            elif char == "#":
                self._skip_comment()
            else:
                return

    def _list_operator(self) -> None:
        char, following = self.text[self.position], self._peek(1)
        if char in "()":
            raise _Refused("subshell")
        if char == ";":
            if following == ";":
                raise _Refused("compound_command")
            self._end_stage(SEQUENCE, explicit=True)
            self.position += 1
        elif char == "&":
            if following != "&":
                raise _Refused("background_job")
            self._end_stage(AND, explicit=True)
            self.position += 2
            self._continue_on_next_lines()
        elif following == "|":
            self._end_stage(OR, explicit=True)
            self.position += 2
            self._continue_on_next_lines()
        else:
            self._end_pipeline_element()
            if following == "&":
                # `a |& b` pipes stderr along with stdout
                self.pipeline[-1].redirects.append(Redirect(2, "duplicate", Word("1")))
                self.position += 1
            self.position += 1
            self._continue_on_next_lines()

    def _word(self) -> None:
        command = self._current()
        start = self.position
        word, quoted = self._read_word()
        if word.text.isdigit() and not quoted and self._peek() in ("<", ">"):
            self._redirect(fd=int(word.text))
            return
        if not command.words and _ASSIGNMENT.match(self.text, start):
            command.assignments.append(word.text)
        else:
            if not command.words and not quoted and word.text in _RESERVED_WORDS:
                raise _Refused("compound_command")
            command.words.append(word)
        command.end = self.position

    def _read_word(self) -> tuple[Word, bool]:
        """The word at the cursor, and whether any part of it was quoted or escaped."""
        parts: list[str] = []
        expands = globs = braces = quoted = False
        while self.position < len(self.text) and self.text[self.position] not in _WORD_END:
            char, following = self.text[self.position], self._peek(1)
            if char == "'":
                end = self.text.find("'", self.position + 1)
                if end == -1:
                    raise _Refused("unterminated_quote")
                parts.append(self.text[self.position + 1 : end])
                self.position, quoted = end + 1, True
            elif char == '"':
                text, dollar = self._read_double_quoted()
                parts.append(text)
                expands, quoted = expands or dollar, True
            elif char == "\\":
                if not following:
                    # a trailing backslash has nothing to escape and stays as it is
                    parts.append(char)
                elif following != "\n":
                    parts.append(following)
                    quoted = True
                self.position += 2
            elif char == "`":
                raise _Refused("command_substitution")
            elif char == "$":
                self._refuse_dollar_forms(following)
                expands = expands or _starts_parameter(following)
                parts.append(char)
                self.position += 1
            else:
                if char == "~" and not parts:
                    expands = True
                if char in _GLOB_CHARS:
                    globs = True
                if char == "{" and self._brace_expansion_follows():
                    braces = True
                parts.append(char)
                self.position += 1
        return Word("".join(parts), expands, globs, braces), quoted

    def _refuse_dollar_forms(self, following: str) -> None:
        if following == "(":
            raise _Refused("command_substitution")
        if following in ("'", '"'):
            raise _Refused("locale_or_ansi_quoting")
        if following == "{" and "}" not in self._rest_of_word():
            # `${x:-a b}` holds blanks or operators the word boundary would cut through
            raise _Refused("parameter_expansion")

    def _brace_expansion_follows(self) -> bool:
        """At a `{`: whether the word goes on to a brace list (`{a,b}`) or sequence (`{1..3}`)."""
        rest = self._rest_of_word()
        closing = rest.find("}")
        return closing != -1 and ("," in rest[:closing] or ".." in rest[:closing])

    def _rest_of_word(self) -> str:
        end = self.position
        while end < len(self.text) and self.text[end] not in _WORD_END:
            end += 1
        return self.text[self.position : end]

    def _read_double_quoted(self) -> tuple[str, bool]:
        """The contents of the double-quoted span at the cursor, and whether it expands `$`."""
        parts: list[str] = []
        expands = False
        index = self.position + 1
        while index < len(self.text):
            char = self.text[index]
            following = self.text[index + 1] if index + 1 < len(self.text) else ""
            if char == '"':
                self.position = index + 1
                return "".join(parts), expands
            if char == "\\" and following and following in '"\\$`\n':
                if following != "\n":
                    parts.append(following)
                index += 2
                continue
            if char == "`" or (char == "$" and following == "("):
                raise _Refused("command_substitution")
            expands = expands or (char == "$" and _starts_parameter(following))
            parts.append(char)
            index += 1
        raise _Refused("unterminated_quote")

    def _redirect(self, fd: int | None) -> None:
        command = self._current()
        if self.text.startswith("<<<", self.position):
            raise _Refused("here_string")
        if self._peek(1) == "(":
            raise _Refused("process_substitution")
        if self.text.startswith("<<", self.position):
            if fd not in (None, 0):
                raise _Refused("heredoc_on_another_descriptor")
            self._open_heredoc(command)
            return
        if self.text.startswith("&>", self.position):
            append = self.text.startswith("&>>", self.position)
            self.position += 3 if append else 2
            target = self._redirect_target()
            command.redirects += [
                Redirect(1, "append" if append else "write", target),
                Redirect(2, "duplicate", Word("1")),
            ]
            command.end = self.position
            return
        operator, mode = next(
            (operator, mode)
            for operator, mode in _REDIRECT_OPERATORS
            if self.text.startswith(operator, self.position)
        )
        self.position += len(operator)
        target = self._redirect_target()
        if mode == "duplicate" and not (target.text.isdigit() or target.text == "-"):
            if operator != ">&" or fd not in (None, 1):
                # `<&file` and `2>&file` are ambiguous redirects: bash refuses to run the command
                raise _Refused("ambiguous_redirect")
            # `>&file` means `&>file`: both streams go to the file
            command.redirects += [Redirect(1, "write", target), Redirect(2, "duplicate", Word("1"))]
        else:
            default_fd = 0 if operator.startswith("<") else 1
            command.redirects.append(Redirect(default_fd if fd is None else fd, mode, target))
        command.end = self.position

    def _redirect_target(self) -> Word:
        self._skip_spaces()
        if self._peek() == "#":
            raise _Refused("redirect_without_target")
        target, _ = self._read_word()
        if not target.text:
            raise _Refused("redirect_without_target")
        return target

    def _open_heredoc(self, command: _CommandDraft) -> None:
        if command.has_pending_heredoc or command.heredoc is not None:
            raise _Refused("several_heredocs_for_one_command")
        strip_tabs = self._peek(2) == "-"
        self.position += 3 if strip_tabs else 2
        self._skip_spaces()
        delimiter, quoted = self._read_word()
        if not delimiter.text:
            raise _Refused("heredoc_without_delimiter")
        command.end = self.position
        command.has_pending_heredoc = True
        self.pending.append(
            _PendingHeredoc(command, delimiter.text, strip_tabs, expands=not quoted)
        )

    def _read_heredoc_bodies(self) -> None:
        """Read the body of every heredoc opened on the line that just ended, in order."""
        for pending in self.pending:
            start = self.position
            lines: list[str] = []
            terminated = False
            while self.position < len(self.text):
                newline = self.text.find("\n", self.position)
                line_end = len(self.text) if newline == -1 else newline
                line = self.text[self.position : line_end]
                self.position = line_end if newline == -1 else line_end + 1
                if pending.strip_tabs:
                    line = line.lstrip("\t")
                if line.rstrip("\r") == pending.delimiter:
                    terminated = True
                    break
                lines.append(line)
            owner = pending.owner
            body = "".join(f"{line}\n" for line in lines)
            owner.heredoc = Heredoc(pending.delimiter, body, pending.expands, terminated)
            owner.heredoc_start, owner.heredoc_end = start, self.position
            owner.has_pending_heredoc = False
        self.pending = []

    def _current(self) -> _CommandDraft:
        if self.command is None:
            self.command = _CommandDraft(start=self.position, end=self.position)
        return self.command

    def _end_pipeline_element(self) -> None:
        if self.command is None or self.command.is_empty():
            raise _Refused("empty_pipeline_element")
        self.pipeline.append(self.command)
        self.command = None

    def _end_stage(self, next_separator: str, explicit: bool) -> None:
        """Close the stage being read. `explicit` is True for a written `;`, `&&` or `||`, which
        bash rejects when there is no command before it."""
        if self.command is not None and not self.command.is_empty():
            self.pipeline.append(self.command)
        elif self.pipeline:
            raise _Refused("empty_pipeline_element")
        self.command = None
        if self.pipeline:
            self.stages.append(_StageDraft(self.separator, self.pipeline))
            self.pipeline = []
            self.separator = next_separator
        elif explicit or self.separator in (AND, OR):
            raise _Refused("list_operator_without_command")

    def _build_stage(self, draft: _StageDraft) -> Stage:
        commands = tuple(self._build_command(command) for command in draft.commands)
        end = draft.commands[-1].end
        text = self.text[draft.commands[0].start : end].strip()
        for command in draft.commands:
            # a body that the pipe continued past already sits inside the stage's own text
            if command.heredoc is not None and command.heredoc_start >= end:
                text += "\n" + command.heredoc.body
                if command.heredoc.terminated:
                    text += command.heredoc.delimiter
        return Stage(draft.separator, commands, text.rstrip("\n"))

    def _build_command(self, draft: _CommandDraft) -> SimpleCommand:
        return SimpleCommand(
            words=tuple(draft.words),
            assignments=tuple(draft.assignments),
            redirects=tuple(draft.redirects),
            heredoc=draft.heredoc,
            text=self.text[draft.start : draft.end].strip(),
        )


def _starts_parameter(following: str) -> bool:
    """Whether a `$` followed by this character begins a parameter expansion."""
    return bool(following) and (following.isalnum() or following in _PARAMETER_MARKS)
