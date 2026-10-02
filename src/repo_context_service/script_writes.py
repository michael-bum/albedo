"""What files a Python or Node program writes, removes, copies or moves, read from its source.

`script_writes(code, language)` finds the file-changing calls of a program - `open(path, "w")`,
`Path(...).write_text(...)`, `os.remove`, `os.makedirs`, `shutil.copy`/`copytree`/`move`,
`os.rename`, Node's `fs.writeFileSync` and the like - and the paths each names. A Python path is
followed through string literals, f-strings, `+`, `Path(...) / ...`, `os.path.join` and names
bound to such a value earlier in the same scope (a function's parameters are not followed). A path
under a temporary directory the program makes lies outside the checkout and is left out.

Each change says whether it surely happens when the program runs to success: a call made
directly by a module-level statement (or in the body of a module-level `with`, of
`if __name__ == "__main__":`, or of a loop over a literal list) does; one in a function, a branch
or a `try` may not. A file-changing call naming a path this cannot follow, and a subprocess
running a file-changing program, make the program's writes untraceable. Python code that does not
parse never runs, so it writes nothing.
"""

from __future__ import annotations

import ast
import posixpath
import re
from dataclasses import dataclass, field

# a path under a directory the program makes with tempfile; no real path starts with it
_TEMPORARY = "\0temporary"
_MAX_VALUES = 32
_WRITE_MODE = set("wax+")
_FILE_CHANGING_PROGRAM = re.compile(
    r"(?<![\w-])(?:sed|rm|mv|cp|perl|patch|truncate|tee|touch|mkdir"
    r"|git\W+(?:apply|checkout|reset|stash|restore|rm|mv|clean))(?![\w-])"
)
_NODE_WRITERS = re.compile(
    r"\.(?P<name>writeFile|appendFile|copyFile|rename|unlink|rm|rmdir|mkdir|writeFileSync|"
    r"appendFileSync|copyFileSync|renameSync|unlinkSync|rmSync|rmdirSync|mkdirSync|"
    r"createWriteStream)\(\s*(?P<args>[^)]*)\)"
)
_NODE_STRING = re.compile(r"""^\s*(?:'([^'\\]*)'|"([^"\\]*)"|`([^`$\\]*)`)\s*$""")


@dataclass
class ScriptWrites:
    # (path, whether the change surely happens); (source, destination, surely, whether onto an
    # existing directory it lands inside it)
    written: list[tuple[str, bool]] = field(default_factory=list)
    removed: list[tuple[str, bool]] = field(default_factory=list)
    made_dirs: list[tuple[str, bool]] = field(default_factory=list)
    copied: list[tuple[str, str, bool, bool]] = field(default_factory=list)
    moved: list[tuple[str, str, bool, bool]] = field(default_factory=list)
    untraceable: bool = False


def script_writes(code: str, language: str) -> ScriptWrites:
    writes = ScriptWrites()
    if language == "node":
        _node_writes(code, writes)
        return writes
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return writes
    _Walker(code, writes).body(tree.body, {}, top=True)
    return writes


_REMOVERS = {"os.remove", "os.unlink", "os.rmdir", "os.removedirs", "shutil.rmtree"}
_DIR_MAKERS = {"os.makedirs", "os.mkdir"}
_COPIERS = {"shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree"}
_MOVERS = {"shutil.move", "os.rename", "os.replace", "os.renames"}
_OPENERS = {"open", "io.open", "codecs.open", "gzip.open", "bz2.open", "lzma.open"}
_SHELL_CALLS = {"subprocess.run", "subprocess.call", "subprocess.check_call",
                "subprocess.check_output", "subprocess.Popen", "os.system", "os.popen"}  # fmt: skip
_METHODS = {"write_text": "write", "write_bytes": "write", "touch": "write", "unlink": "remove",
            "rmdir": "remove", "mkdir": "mkdir", "rename": "move", "replace": "move",
            "to_csv": "write", "to_json": "write", "to_parquet": "write", "to_excel": "write",
            "to_pickle": "write", "savefig": "write"}  # fmt: skip
_TEMP_CALLS = {"tempfile.mkdtemp", "tempfile.gettempdir", "tempfile.TemporaryDirectory",
               "tempfile.NamedTemporaryFile", "tempfile.mkstemp"}  # fmt: skip
_SAME_PATH = {"str", "os.path.abspath", "os.path.realpath", "os.path.normpath", "os.fspath",
              "Path", "pathlib.Path", "PurePath", "PosixPath", "os.path.expanduser"}  # fmt: skip


class _Walker:
    def __init__(self, code: str, writes: ScriptWrites):
        self.code = code
        self.writes = writes

    def body(self, statements: list[ast.stmt], env: dict, top: bool) -> None:
        for statement in statements:
            self.statement(statement, env, top)

    def statement(self, node: ast.stmt, env: dict, top: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            local = dict(env)
            for argument in [*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs]:
                local[argument.arg] = None
            for extra in (node.args.vararg, node.args.kwarg):
                if extra is not None:
                    local[extra.arg] = None
            self.body(node.body, local, top=False)
            env[node.name] = None
            return
        if isinstance(node, ast.ClassDef):
            self.body(node.body, dict(env), top=False)
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is not None:
                self.calls(node.value, env, top)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = self.value(node.value, env) if node.value is not None else None
            for target in targets:
                self.bind(target, value, env)
            return
        if isinstance(node, ast.AugAssign):
            self.calls(node.value, env, top)
            self.bind(node.target, None, env)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self.calls(item.context_expr, env, top)
                if item.optional_vars is not None:
                    self.bind(item.optional_vars, self.value(item.context_expr, env), env)
            self.body(node.body, env, top)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            self.calls(node.iter, env, top)
            items = self.items(node.iter, env)
            if items and len(items) <= _MAX_VALUES and isinstance(node.target, ast.Name):
                for item in items:
                    env[node.target.id] = [item]
                    self.body(node.body, env, top)
            else:
                self.bind(node.target, None, env)
                self.body(node.body, env, False)
            self.body(node.orelse, env, False)
            return
        if isinstance(node, ast.If):
            self.calls(node.test, env, top)
            main = (
                isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            )
            self.body(node.body, env, top and main)
            self.body(node.orelse, env, False)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self.statement(child, env, False)
            elif isinstance(child, ast.expr):
                self.calls(child, env, top and isinstance(node, (ast.Expr, ast.Return)))
            else:
                # an `except` handler or a `match` case: its statements may not run
                for inner in ast.iter_child_nodes(child):
                    if isinstance(inner, ast.stmt):
                        self.statement(inner, env, False)
                    elif isinstance(inner, ast.expr):
                        self.calls(inner, env, False)

    def bind(self, target: ast.expr, value, env: dict) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.bind(element, None, env)

    def calls(self, node: ast.expr, env: dict, top: bool) -> None:
        """Every file-changing call in an expression; only the outermost one surely runs when
        the expression does, the rest are arguments evaluated first, and do too."""
        for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
            self.call(call, env, top and not isinstance(node, (ast.Lambda, ast.IfExp, ast.BoolOp)))

    def call(self, node: ast.Call, env: dict, top: bool) -> None:
        name = _dotted(node.func)
        args = node.args
        writes = self.writes
        if name in _OPENERS:
            mode = _keyword(node, "mode") or (args[1] if len(args) > 1 else None)
            if args and self.opens_for_writing(mode, env):
                self.record(writes.written, args[0], env, top)
        elif name in _REMOVERS and args:
            self.record(writes.removed, args[0], env, top)
        elif name in _DIR_MAKERS and args:
            self.record(writes.made_dirs, args[0], env, top)
        elif name in _COPIERS | _MOVERS and len(args) > 1:
            pairs = writes.moved if name in _MOVERS else writes.copied
            sources, targets = self.paths(args[0], env), self.paths(args[1], env)
            if sources is None or targets is None:
                writes.untraceable = True
                return
            # a copy or move onto an existing directory lands inside it, except a tree copy
            into = name in ("shutil.copy", "shutil.copy2", "shutil.move")
            for source in sources:
                for target in targets:
                    if not target.startswith(_TEMPORARY):
                        pairs.append((source, target, top, into))
                    elif name in _MOVERS and not source.startswith(_TEMPORARY):
                        writes.removed.append((source, top))
        elif name in _SHELL_CALLS or name == "fileinput.input":
            text = ast.get_source_segment(self.code, node) or ""
            if name == "fileinput.input":
                if "inplace" in text and args:
                    self.record(writes.written, args[0], env, top)
            elif _FILE_CHANGING_PROGRAM.search(text) or ">" in text:
                writes.untraceable = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr in _METHODS:
            kind = _METHODS[node.func.attr]
            receiver = node.func.value
            if kind == "write" and node.func.attr.startswith(("to_", "save")):
                if args:
                    self.record(writes.written, args[0], env, top)
                return
            if self.paths(receiver, env) is None and not _looks_like_path(receiver):
                return
            if kind == "move":
                if args:
                    sources, targets = self.paths(receiver, env), self.paths(args[0], env)
                    if sources is None or targets is None:
                        writes.untraceable = True
                        return
                    writes.moved += [(s, t, top, False) for s in sources for t in targets]
                return
            listing = {"write": writes.written, "remove": writes.removed,
                       "mkdir": writes.made_dirs}[kind]  # fmt: skip
            self.record(listing, receiver, env, top)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "open" and args:
            receiver = node.func.value
            if self.opens_for_writing(args[0], env) and self.paths(receiver, env) is not None:
                self.record(writes.written, receiver, env, top)

    def record(self, into: list, node: ast.expr, env: dict, top: bool) -> None:
        paths = self.paths(node, env)
        if paths is None:
            self.writes.untraceable = True
            return
        into += [(path, top) for path in paths if not path.startswith(_TEMPORARY)]

    def opens_for_writing(self, mode: ast.expr | None, env: dict) -> bool:
        if mode is None:
            return False
        values = self.value(mode, env)
        return values is None or any(set(v) & _WRITE_MODE for v in values)

    def paths(self, node: ast.expr, env: dict) -> list[str] | None:
        values = self.value(node, env)
        if values is None:
            return None
        return [posixpath.normpath(v) if not v.startswith(_TEMPORARY) else v for v in values]

    def items(self, node: ast.expr, env: dict) -> list[str] | None:
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [self.value(element, env) for element in node.elts]
            return None if None in values else [v for each in values for v in each]
        return None

    def value(self, node: ast.expr, env: dict) -> list[str] | None:
        """Every text an expression can stand for, or None when it is not one this follows."""
        if isinstance(node, ast.Constant):
            return [node.value] if isinstance(node.value, str) else None
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.JoinedStr):
            pieces = []
            for part in node.values:
                if isinstance(part, ast.FormattedValue):
                    pieces.append(None if part.format_spec else self.value(part.value, env))
                else:
                    pieces.append(self.value(part, env))
            return _combine(pieces, "")
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
            left, right = self.value(node.left, env), self.value(node.right, env)
            return _combine([left, right], "/" if isinstance(node.op, ast.Div) else "")
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in _TEMP_CALLS:
                return [_TEMPORARY]
            if name in ("os.getcwd", "Path.cwd", "pathlib.Path.cwd"):
                return ["."]
            if name in _SAME_PATH and len(node.args) == 1:
                return self.value(node.args[0], env)
            if name == "os.path.join" and node.args:
                return _combine([self.value(a, env) for a in node.args], "/")
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("resolve", "absolute"):
                return self.value(node.func.value, env)
            return None
        if isinstance(node, ast.Attribute) and node.attr == "name":
            inner = self.value(node.value, env)
            return inner if inner == [_TEMPORARY] else None
        return None


def _combine(pieces: list[list[str] | None], separator: str) -> list[str] | None:
    if any(piece is None for piece in pieces):
        return None
    out = [""]
    for position, piece in enumerate(pieces):
        glue = separator if position else ""
        out = [(posixpath.join(a, b) if glue == "/" else a + b) for a in out for b in piece]
        if len(out) > _MAX_VALUES:
            return None
    return out


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f"{inner}.{node.attr}" if inner else ""
    return ""


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in node.keywords if k.arg == name), None)


def _looks_like_path(node: ast.expr) -> bool:
    """A receiver built as a path (`Path(...)`, `p / "x"`), whose methods act on the file."""
    if isinstance(node, ast.Call):
        return _dotted(node.func) in ("Path", "pathlib.Path", "PurePath", "PosixPath")
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)


def _node_writes(code: str, writes: ScriptWrites) -> None:
    for match in _NODE_WRITERS.finditer(code):
        name = match.group("name").removesuffix("Sync")
        arguments = [a for a in match.group("args").split(",") if a.strip()]
        count = 2 if name in ("copyFile", "rename") else 1
        paths = []
        for argument in arguments[:count]:
            literal = _NODE_STRING.match(argument)
            groups = literal.groups() if literal else ()
            paths.append(next((group for group in groups if group is not None), None))
        if len(paths) < count or None in paths:
            writes.untraceable = True
            continue
        if name == "copyFile":
            writes.copied.append((paths[0], paths[1], False, False))
        elif name == "rename":
            writes.moved.append((paths[0], paths[1], False, False))
        elif name in ("unlink", "rm", "rmdir"):
            writes.removed.append((paths[0], False))
        elif name == "mkdir":
            writes.made_dirs.append((paths[0], False))
        else:
            writes.written.append((paths[0], False))
