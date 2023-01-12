import ast
import re
import sys
from typing import List, Optional, Union

from unidiff import PatchSet, PatchedFile, Hunk

Node = Union[ast.stmt, ast.expr, ast.keyword]
RecursiveStmts = Union[Node, List['RecursiveStmts']]


class AstParseError(SyntaxError):
  def __init__(self, *args):
    super(AstParseError, self).__init__(*args)


def _make_indent(level: int, indent: Optional[int]):
  if not indent:
    return ''
  return '\n' + (' ' * (level * indent))


# noinspection DuplicatedCode
class StructureSuperImposer:
  def __init__(
      self, source_nodes: RecursiveStmts, target_nodes: RecursiveStmts,
  ):
    self.source_nodes = source_nodes
    self.target_nodes = target_nodes

  def to_bracketed_notation(self, delimiter: Optional[str] = None, indent=0, source_only=False):
    if bool(delimiter) == bool(source_only):
      assert False, f'Invalid combination delimiter: {delimiter}, source_only: {source_only}'

    bracketed_src = StructureSuperImposer.get_bracketed_notation_of(self.source_nodes, indent)
    if source_only:
      return bracketed_src

    bracketed_target = StructureSuperImposer.get_bracketed_notation_of(self.target_nodes, indent)
    return (
        bracketed_src.replace(delimiter, '...') +
        delimiter +
        bracketed_target.replace(delimiter, '...')
    )

  def __repr__(self):
    import pprint

    ast_repr = ast.AST.__repr__
    if StructureSuperImposer.USE_AST_UNPARSE_REPR:
      ast.AST.__repr__ = ast.unparse
    else:  # otherwise, use AST dump REPR
      ast.AST.__repr__ = lambda nodes: ast.dump(nodes, indent=4)
    str_obj = pprint.pformat(self.source_nodes, indent=4)
    if StructureSuperImposer.USE_TARGET_CODE_IN_REPR:
      str_obj += '\n' + pprint.pformat(self.target_nodes, indent=4)
    ast.AST.__repr__ = ast_repr
    return str_obj

  @classmethod
  def get_bracketed_notation_of(
      cls, nodes: RecursiveStmts, indent: Optional[int], level=0
  ) -> str:
    bracketed_notation = ''
    for node in nodes:
      if isinstance(node, list):
        bracketed_notation += cls.get_bracketed_notation_of(
          node, indent, level=level + 1
        )
      elif isinstance(node, ast.AST):
        _fields = filter(lambda field: field != 'ctx', node._fields)
        children = [node.__getattribute__(field) for field in _fields]
        children = list(filter(lambda x: x is not None, children))
        if isinstance(node, ast.Dict):
          assert len(children) == 2
          children = [list(pair) for pair in zip(children[0], children[1])]
        bracketed_notation += _make_indent(level, indent)
        bracketed_notation += '('
        bracketed_notation += type(node).__name__
        bracketed_notation += cls.get_bracketed_notation_of(
          children, indent, level=level + 1,
        )
        bracketed_notation += ')'
        bracketed_notation += type(node).__name__

      else:
        # Otherwise, `node` is a primitive. To best of my knowledge, in this
        # branch `node` only can can be a string containing a field-name.
        if bracketed_notation.endswith(')Name'):
          # for nested object access (e.g. `p1.field`), this `else` branch
          # creates strings like "(Name_p1)Name_field". To overcome this,
          # remove last five characters.
          bracketed_notation = bracketed_notation[:-5]
        bracketed_notation += f'_{node}'

    return bracketed_notation

  @staticmethod
  def get_patched_file_from_patch(patch: Union[PatchSet, str]) -> PatchedFile:
    if type(patch) is str:
      patch = PatchSet(patch)

    if len(patch) == 1:
      patched_file: PatchedFile = patch[0]
    else:
      # Due to a bug, unidiff creates two `PatchedFile`s when there are
      # whitespace inside the filenames. My ASSUMPTIONS are in such cases,
      # there will be two `PatchedFile`s and the later one will contain
      # the full information. The former one will be essentially empty.
      # Here, asserting my assumptions first.
      assert len(patch) == 2, len(patch)
      assert patch[0].added == patch[0].removed == 0
      patched_file: PatchedFile = patch[1]
      assert (patched_file.added + patched_file.removed) > 0
    return patched_file

  @staticmethod
  def from_source_codes(prev_source: str, curr_source: str) -> 'StructureSuperImposer':
    prev_tree = StructureSuperImposer.parse_ast_with_async_await_check(prev_source)
    target_tree = StructureSuperImposer.parse_ast_with_async_await_check(curr_source)

    return StructureSuperImposer(prev_tree.body, target_tree.body)

  @staticmethod
  def from_source_code(source_code: str, start: int, end: int):
    source_tree = StructureSuperImposer.parse_ast_with_async_await_check(source_code)
    intersecting_nodes = [
      StructureSuperImposer._get_intersecting_nodes(child, start, end)
      for i, child in enumerate(source_tree.body)
      if StructureSuperImposer._has_intersection(child, start, end)
    ]

    return StructureSuperImposer(intersecting_nodes, [])

  @staticmethod
  def from_filenames(filename1: str, filename2: str):
    with open(filename1, 'r') as file1:
      with open(filename2, 'r') as file2:
        return StructureSuperImposer.from_source_codes(
          file1.read(),
          file2.read(),
        )

  @staticmethod
  def from_diff(
      prev_source: str,
      diff: str,
      reverse=False,
      log_syntax_error=False,
  ):
    diff_patch = PatchSet(diff)
    patched_file = StructureSuperImposer.get_patched_file_from_patch(diff_patch)

    try:
      prev_tree = StructureSuperImposer.parse_ast_with_async_await_check(
        prev_source, filename=patched_file.source_file,
      )
    except SyntaxError as e:
      if log_syntax_error:
        print('prev source', e.msg)
        print(e.text, file=sys.stderr)

      raise AstParseError from e

    try:
      target = StructureSuperImposer._make_target_source(prev_source, patched_file, reverse)
      target_tree = StructureSuperImposer.parse_ast_with_async_await_check(
        target, filename=patched_file.target_file,
      )
    except SyntaxError as e:
      if log_syntax_error:
        print('target source', e.msg)
        print(e.text, file=sys.stderr)

      raise AstParseError from e

    intersecting_source_nodes = []
    intersecting_target_nodes = []
    hunk: Hunk
    for hunk in patched_file:
      prev_start = hunk.source_start
      prev_end = hunk.source_start + hunk.source_length
      intersecting_source_nodes.extend(
        StructureSuperImposer._get_intersecting_nodes(child, prev_start, prev_end)
        for i, child in enumerate(prev_tree.body)
        if StructureSuperImposer._has_intersection(child, prev_start, prev_end)
      )

      target_start = hunk.target_start
      target_end = hunk.target_start + hunk.target_length
      intersecting_target_nodes.extend(
        StructureSuperImposer._get_intersecting_nodes(child, target_start, target_end)
        for child in target_tree.body
        if StructureSuperImposer._has_intersection(child, target_start, target_end)
      )

    if reverse:
      return StructureSuperImposer(
        intersecting_target_nodes, intersecting_source_nodes,
      )

    return StructureSuperImposer(
      intersecting_source_nodes, intersecting_target_nodes,
    )

  @staticmethod
  def parse_ast_with_async_await_check(source: str, filename: str = '<unknown>'):
    try:
      return ast.parse(source, type_comments=True, filename=filename)
    except SyntaxError as _e:
      if re.search(r'\basync\b', _e.text):
        # async/await are reserved keywords since python 3.7
        return ast.parse(source, feature_version=(3, 6))

      raise

  @staticmethod
  def _is_unit_type(node) -> bool:
    if isinstance(node, ast.Call):
      # if it is a function call, consider everything as context
      return True

    if isinstance(node, ast.arguments):
      # if it is argument list to a function call, take everything
      return True

    if isinstance(node, ast.alias):
      return True

    if isinstance(node, StructureSuperImposer.COMPREHENSION_TYPES):
      # if it is ListComp | SetComp | GeneratorExp | DictComp, take it all
      return True

    if isinstance(node, ast.expr_context):
      # `ast.expr_context` defines the context in which the expr is used.
      # It can be `Load | Store | Del` which do not have line no
      return True

    if isinstance(node, ast.cmpop):
      # `ast.cmpop` can be comparison operators like
      # Eq, NotEq, Lt, LtE, Gt, GtE, Is, IsNot, In, NotIn
      return True

    if isinstance(node, ast.operator):
      # `ast.operator` can be operators like
      # Add, Sub, Mult, MatMult, Div, Mod, Pow, LShift, RShift,
      # BitOr, BitXor, BitAnd, FloorDiv
      return True

    if isinstance(node, ast.boolop):
      # `ast.boolop` can be operators like And, Or
      return True

    if isinstance(node, ast.unaryop):
      # `ast.unaryop` can be operators like Invert, Not, UAdd, USub
      return True

    return False

  @staticmethod
  def _has_intersection(node, start: int, end: int) -> bool:
    """
    Assume |    | is the diff range and ~~~~ is the node's range
    One intersection could be ~~|~~  |
    which means node's end >= diff start AND node's start <= diff end
    Another could be |  ~~|~~
    which means node's start <= diff end AND node's end >= diff start
    """

    if StructureSuperImposer._is_unit_type(node):
      return True

    return (node.end_lineno >= start and node.lineno <= end) or (
        node.lineno <= end and node.end_lineno >= start)

  @staticmethod
  def _get_intersecting_nodes(
      node: Node, start: int, end: int,
  ) -> RecursiveStmts:
    assert type(node) is not ast.Module

    if StructureSuperImposer._is_unit_type(node):
      if not isinstance(node, ast.Call):
        return node

      call: ast.Call = node.__class__(**node.__dict__)
      call.args = [
        StructureSuperImposer._get_intersecting_nodes(arg, start, end)
        for arg in node.args
        if StructureSuperImposer._has_intersection(arg, start, end)
      ]
      call.keywords = [
        StructureSuperImposer._get_intersecting_nodes(kw, start, end)
        for kw in node.keywords
        if StructureSuperImposer._has_intersection(kw, start, end)
      ]
      return call

    if node.lineno >= start and node.end_lineno <= end:
      return node

    if node.end_lineno < start or node.lineno > end:
      raise RuntimeError(
        'get_intersecting_nodes should not be called if there is no intersection'
      )

    # reaching here means either the diff is totally inside the node

    children = ast.iter_child_nodes(node)

    if isinstance(node, StructureSuperImposer.BLOCK_TYPES):
      if node.lineno >= start:
        # If node is a block and block declaration is
        # included in the diff, return a block statement.
        block = node.__class__(**node.__dict__)
        block.body = [
          StructureSuperImposer._get_intersecting_nodes(child, start, end)
          for child in node.body
          if StructureSuperImposer._has_intersection(child, start, end)
        ]
        return block

      # Otherwise, only the body of the block is part
      # of the diff, so treat the child-statements as regular statements.
      children = node.body

    if isinstance(node, StructureSuperImposer.LIST_TYPES):
      if node.lineno >= start:
        # If node is a list/set/tuple and the list declaration is
        # included in the diff, return a list statement.
        _list = node.__class__(**node.__dict__)
        _list.elts = [
          StructureSuperImposer._get_intersecting_nodes(child, start, end)
          for child in node.elts
          if StructureSuperImposer._has_intersection(child, start, end)
        ]
        return _list

      # Otherwise, only the body of the list is part
      # of the diff, so treat the list-elements as regular statements.
      children = node.elts

    if isinstance(node, ast.Dict):
      if node.lineno >= start:
        # If node is a dict and the dict declaration is
        # included in the diff, return a dict statement.
        # In this case, key-value order will be fixed during SBT generation.
        #
        # ***For spread operator (**), corresponding key is None***
        # ***If the key is actually None, it is wrapped in an `ast.Constant` object***
        _dict: ast.Dict = node.__class__(**node.__dict__)
        intersecting_element_idx = [
          i for i in range(len(node.keys))
          if StructureSuperImposer._has_intersection(node.values[i], start, end)
        ]
        _dict.keys = [node.keys[i] for i in intersecting_element_idx]
        _dict.values = [
          StructureSuperImposer._get_intersecting_nodes(node.values[i], start, end)
          for i in intersecting_element_idx
        ]
        return _dict

      # Otherwise, reorder key-value serials
      children = StructureSuperImposer._reorder_dict_children(node)

    return [
      StructureSuperImposer._get_intersecting_nodes(child, start, end)
      for child in children
      if StructureSuperImposer._has_intersection(child, start, end)
    ]

  @staticmethod
  def _make_target_source(_prev_source: str, file_patch: PatchedFile, reverse: bool):

    def trim_whitespace(target_line: str):
      """
      First char of diff line is '+', '-', or ' ' which are extra chars.
      If last char is '\n', it will produce extra newline
      upon join by '\n' latter.
      """
      end_i = -1 if target_line.endswith('\n') else len(target_line)
      return target_line[1: end_i]

    lines = _prev_source.split('\n')
    # python lists are 0-indexed, but line numbers are 1-indexed.
    # Also, if lengths of a `_hunk.source` and a `_hunk.target` are
    # different they will affect the indexing of the latter replacements.
    # `offset` fixes these issues.
    offset = -1
    _hunk: Hunk
    for _hunk in file_patch:
      _start = _hunk.target_start if reverse else _hunk.source_start
      _length = _hunk.target_length if reverse else _hunk.source_length
      _end = _start + _length
      changed_lines = list(map(
        trim_whitespace, _hunk.source if reverse else _hunk.target,
      ))

      lines[_start + offset: _end + offset] = changed_lines

      num_former_lines = _end - _start
      offset += (len(changed_lines) - num_former_lines)

    return '\n'.join(lines)

  @staticmethod
  def _reorder_dict_children(_dict: ast.Dict):
    ordered_children = []
    position_attrs = 'col_offset', 'end_col_offset', 'lineno', 'end_lineno'
    for i, key in enumerate(_dict.keys):
      value = _dict.values[i]
      if key is None:
        _dummy_key = ast.Constant(**{
          'value': '**',
          'kind': None,
          **{attr: value.__getattribute__(attr) for attr in position_attrs},
        })
        ordered_children.append(_dummy_key)
      else:
        ordered_children.append(key)
      ordered_children.append(value)

    return ordered_children


StructureSuperImposer.BLOCK_TYPES = (
  ast.FunctionDef,
  ast.AsyncFunctionDef,
  ast.ClassDef,
  ast.For,
  ast.AsyncFor,
  ast.While,
  ast.If,
  ast.With,
  ast.AsyncWith,
  ast.Try,
  ast.ExceptHandler,
)
StructureSuperImposer.LIST_TYPES = (
  ast.List,
  ast.Tuple,
  ast.Set,
)
StructureSuperImposer.COMPREHENSION_TYPES = (
  ast.ListComp,
  ast.SetComp,
  ast.GeneratorExp,
  ast.DictComp,
)
StructureSuperImposer.USE_AST_UNPARSE_REPR = True
StructureSuperImposer.USE_TARGET_CODE_IN_REPR = True


def assert_version():
  assert (
      sys.version_info.major == 3 and sys.version_info.minor >= 9
  ), 'StructureSuperImposer requires Python 3.9'


if __name__ == '__main__':
  assert_version()

  StructureSuperImposer.USE_AST_UNPARSE_REPR = False
  StructureSuperImposer.USE_TARGET_CODE_IN_REPR = False

  structure_superimposer = StructureSuperImposer.from_diff("""(
    "loss:",
    loss,
    "epsilon:",
    1e-4,
    "time:",
    time,
    "\\n"
  )
  """, """diff --git a/a.java b/a.java
  index 51c00f47b54..3659a0811cd 100644
  --- a/a.java
  +++ b/a.java
  @@ -2,6 +2,6 @@ "loss:",
     loss,
     "epsilon:",
  -	 1e-4,
  +	 1e-6,
     "time:",
     time,
     "\\n"
  """)
  print(structure_superimposer.to_bracketed_notation(delimiter='</s>'))
