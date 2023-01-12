import base64
import json
import math
import os
import random
import re
import sys
from multiprocessing import Process, current_process
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from unidiff import Hunk, PatchSet, UnidiffParseError

from StructureSuperimposer import AstParseError, StructureSuperImposer
from api import ROOT, ServerError, get_rate_limited_response, get_token_list
from log import create_logger_process, logger

pd.options.mode.chained_assignment = None  # default='warn'


def get_most_starred_python_repos():
  star_conditions = ['>10100']
  for i in range(10000, 0, -100):
    star_conditions.append(f'{i}..{i + 99}')

  path_template = '/search/repositories?' \
                  'q=language:python+stars:{}+archived:false' \
                  '&sort=stars&order=desc&per_page=100'

  yield_count = 0
  for star_condition in star_conditions:
    if yield_count >= NUM_REPOS:
      break
    path_template_with_stars = path_template.format(star_condition)
    for page_n in range(10):
      path = f'{path_template_with_stars}&page={page_n + 1}'
      response = get_rate_limited_response(f'{ROOT}{path}')
      assert isinstance(response, dict)
      assert type(response['items']) is list, type(response['items'])

      for item in response['items']:
        repo = {
          key: item[key]
          for key in item.keys()
          if key not in ['owner', 'licence']
        }

        for key in ['owner', 'licence']:
          if key in item.keys():
            for subkey in item[key].keys():
              repo[f'{key}.{subkey}'] = item[key][subkey]

        yield repo
        yield_count += 1


def get_commits_of(repo: str, token: Optional[str] = None):
  from itertools import count
  path_template = f'/repos/{repo}/commits?per_page=100'
  for page_n in count(1):
    path = path_template + f'&page={page_n}'
    response = get_rate_limited_response(f'{ROOT}{path}', token)
    assert isinstance(response, list), type(response)

    # Response format:
    # https://docs.github.com/en/rest/reference/commits#list-commits--code-samples
    for item in response:
      commit = {}
      assert type(item) is dict, f'{type(item)}: {response}'
      for key, value in item.items():
        if isinstance(value, (dict, list)):
          continue
        commit[key] = value
      commit['message'] = item['commit']['message']
      for key in ['author', 'committer', 'tree']:
        for subkey in item['commit'][key].keys():
          commit[f'{key}.{subkey}'] = item['commit'][key][subkey]

      # merging more than two branch is possible
      # ref: https://softwareengineering.stackexchange.com/a/314216/377568
      commit['merge'] = len(item['parents']) > 1
      if len(item['parents']):
        for key, value in item['parents'][0].items():
          commit[f'parent.{key}'] = value
      commit['parents'] = json.dumps(item['parents'])

      yield commit

    if len(response) < 100:
      break


def filter_bugfix_commits(commit_df: pd.DataFrame, soft=False):
  """
  To identify the bug fixing commits, Fischer et al. [48]checked if
  the commit message follows the pattern – (‘fix’ or‘solve’) and
  (‘bug’ or ‘issue’ or ‘problem’ or ‘error’).

  However, this condition is too strict and keeps too few commits.
  This commits will again be filtered out by commit and message size
  leaving too few comments to train on.
  Therefore, the `soft` parameter loosens the condition to contain
  only ‘fix’ or ‘solve’, which returns reasonable number of commits.
  """
  assert 'message' in commit_df.columns

  def filter_message(message):
    soft_match = isinstance(message, str) and (
        'fix' in message or 'solve' in message.lower())

    if not soft_match:
      return False
    if soft:
      return True
    return (soft_match and any(map(
      lambda kw: kw in message.lower(), ['bug', 'issue', 'problem', 'error'],
    )))

  true_indices = commit_df.message.apply(filter_message)

  return commit_df[true_indices]


def get_diffs(commit_df: pd.DataFrame, token=None):
  assert 'url' in commit_df.columns

  for _, row in commit_df[['url', 'author.date', 'committer.date']].iterrows():
    commit_url, author_date, committer_date = row
    # https://docs.github.com/en/rest/reference/commits#get-a-commit--code-samples
    try:
      diff_response = get_rate_limited_response(
        commit_url, token, accept='application/vnd.github.v3.diff',
      )
      obj_response = get_rate_limited_response(commit_url, token)
    except ServerError as se:
      msg = ' | '.join(list(map(str, se.args)))
      msg += f' | Commit URL: {commit_url}'
      print(msg, file=sys.stderr)
      logger.error(msg)
      continue
    assert isinstance(diff_response, str)
    assert isinstance(obj_response, dict)

    # direct ignore the commits that are too large by the number of lines
    if obj_response['stats']['total'] > 100:
      continue

    # if any filename is non-ascii, ignore it
    if not all(file["filename"].isascii() for file in obj_response['files']):
      continue

    # each diff starts by `diff --git`, first split will be empty
    file_wise_diffs = diff_response.split('diff --git ')[1:]

    # test whether all filenames from JSON matches with diff string
    all_filename_matched = True
    for i, file in enumerate(obj_response['files']):
      first_diff_line = file_wise_diffs[i].split("\n")[0]
      if not (
          first_diff_line.endswith(f'b/{file["filename"]}')
          # filenames with whitespace are contained inside ""
          or first_diff_line.endswith(f'"b/{file["filename"]}"')
      ):
        msg = f'Diff filename mismatch: {file["filename"]} | {first_diff_line}'
        info_dict = {**file, **obj_response['commit'], 'patch': None}
        msg += f' | {json.dumps(info_dict)}'
        logger.error(msg)
        print(msg, file=sys.stderr)

        all_filename_matched = False
        break

    if not all_filename_matched:
      continue

    for i, file in enumerate(obj_response['files']):
      # ignore test files
      if 'test' in file["filename"].lower():
        continue

      file_level_diff: Dict[str, Union[str, int]] = {
        'author.date': author_date,
        'committer.date': committer_date
      }
      # store commit identification data
      for key, value in obj_response.items():
        if isinstance(value, (list, dict)):
          continue
        file_level_diff[f'commit_{key}'] = value

      file_level_diff['commit_message'] = obj_response['commit']['message']
      file_level_diff['commit_merge'] = len(obj_response['parents']) > 1

      # append file change information
      file_level_diff.update(file)
      file_level_diff['patch'] = f'diff --git {file_wise_diffs[i]}'
      yield file_level_diff


def _ensure_repository_list():
  if os.path.exists(REPO_FILENAME):
    return
  print('Caching repo list')
  repos = list(get_most_starred_python_repos())
  repo_df = pd.DataFrame(repos)
  print(f'{len(repo_df.full_name.unique())} repos scrapped')
  repo_df.to_csv(REPO_FILENAME, index=False)


fixes_re = re.compile("fix(es)?\\s+#\\d+", re.I)
merge_re0 = re.compile("merge pull request[^\\n]+", re.I)
merge_re1 = re.compile("Merge (remote-tracking )?branch[^\\n]+", re.I)
sign_re0 = re.compile("(Signed-off-by|Reviewed-By|Change-Id):[^\\n]+", re.I)
git_svn_re0 = re.compile("git-svn-id:[^\\n]+", re.I)
bot_re0 = re.compile("(.|\n)*dependabot(.|\n)*", re.I)
ticket_re0 = re.compile("Ticket: [^\\n]+", re.I)


def filter_small_bugfix_commits(
    changed_file_df: pd.DataFrame,
    min_threshold: int,
    msg_max_threshold: int,
    patch_count_threshold: int,
    patch_size_threshold: int,
):
  groups = changed_file_df.groupby('commit_sha')

  for commit_sha in groups.groups:
    files = groups.get_group(commit_sha)
    commit_message: str = next(iter(files.commit_message))

    commit_message = fixes_re.sub("", commit_message)
    commit_message = merge_re0.sub("", commit_message)
    commit_message = merge_re1.sub("", commit_message)
    commit_message = sign_re0.sub("", commit_message)
    commit_message = git_svn_re0.sub("", commit_message)
    commit_message = bot_re0.sub("", commit_message)
    commit_message = bot_re0.sub("", commit_message)
    commit_message = ticket_re0.sub("", commit_message)
    commit_message = commit_message.strip()

    commit_message_words = len(commit_message.split())
    if (
        commit_message_words < min_threshold
        or commit_message_words > msg_max_threshold
    ):
      continue

    modified_python_mask = ((files.status == 'modified')
                            & (files.filename.str.endswith('.py')))
    modified_python_files = files[modified_python_mask]
    if len(modified_python_files) > patch_count_threshold:
      # if number of files crosses patch count threshold,
      # discard them immediately
      continue

    num_patches = 0
    for i, file in modified_python_files.iterrows():
      try:
        patch = PatchSet(file.patch)
      except UnidiffParseError as e:
        logger.error(f'{e} | {file.to_json()}')
        print(f'{e} | {file.to_json()}', file=sys.stderr)
        continue

      patched_file = StructureSuperImposer.get_patched_file_from_patch(patch)

      hunk: Hunk
      for hunk in patched_file:
        hunk_word_count = sum(map(lambda l: len(l.split()), hunk.source))
        if (
            hunk_word_count < min_threshold
            or hunk_word_count > patch_size_threshold
        ):
          # if patch size crosses the threshold,
          # break current loops and make sure parent LOOPS act accordingly
          num_patches = patch_count_threshold + 1
          break

      num_patches += len(patched_file)
      if num_patches > patch_count_threshold:
        break

    if num_patches <= patch_count_threshold:
      for i, file in modified_python_files.iterrows():
        file['commit_message'] = commit_message
        yield file


def _ensure_all_commits_csv(
    all_commit_file_paths: List[Tuple[str, str]],
    token: str,
):
  for repo_name, all_commit_file_path in all_commit_file_paths:
    if os.path.exists(all_commit_file_path):
      raise Exception(f'Exists: {all_commit_file_path}')
    commits = list(get_commits_of(repo_name, token))
    pd.DataFrame(commits).to_csv(all_commit_file_path, index=False)


def _make_all_commits_filename(full_repo_name):
  repo_without_slash = str(full_repo_name).replace("/", ".")
  filename = f'{ALL_COMMIT_ROOT}/{repo_without_slash}-commits.csv'
  return filename


def _spawn_all_commits_processes():
  if not os.path.exists(ALL_COMMIT_ROOT):
    os.mkdir(ALL_COMMIT_ROOT)

  repo_df = pd.read_csv(REPO_FILENAME, usecols=['full_name'])
  all_file_paths: List[Tuple[str, str]] = [
    (repo_name, _make_all_commits_filename(repo_name))
    for repo_name in repo_df.full_name
  ]
  remaining_filepaths: List[Tuple[str, str]] = [
    (repo_name, filepath)
    for repo_name, filepath in all_file_paths
    if not os.path.exists(filepath)
  ]

  if len(remaining_filepaths) == 0:
    print('All commits scrapped')
    return

  tokens = get_token_list()
  page_size = math.ceil(len(remaining_filepaths) / len(tokens))
  processes = [
    Process(
      name=f'all_commits-{i}',
      target=_ensure_all_commits_csv,
      args=(
        remaining_filepaths[i * page_size: (i + 1) * page_size],
        token,
      )
    )
    for i, token in enumerate(tokens)
  ]

  logger_process = create_logger_process(
    ALL_COMMIT_ROOT,
    init_file_count=len(all_file_paths) - len(remaining_filepaths),
    target_file_count=len(all_file_paths),
  )

  print(f'Fetching all commits for {len(remaining_filepaths)} repos')

  for process in processes:
    process.start()
  logger_process.start()

  for process in processes:
    process.join()
  for process in processes:
    assert process.exitcode == 0

  logger_process.terminate()
  logger_process.join()

  print(f'Fetched all commits for {len(remaining_filepaths)} repos')


def _make_bugfix_filename(full_repo_name, bugfix_dir=None):
  if bugfix_dir is None:
    bugfix_dir = BUGFIX_COMMIT_ROOT
  repo_without_slash = str(full_repo_name).replace("/", ".")
  bugfix_filename = f'{bugfix_dir}/{repo_without_slash}-commits.csv'
  return bugfix_filename


def _ensure_bugfix_commits():
  repo_df = pd.read_csv(REPO_FILENAME, usecols=['full_name'])
  if not os.path.exists(BUGFIX_COMMIT_ROOT):
    os.mkdir(BUGFIX_COMMIT_ROOT)
  total_commits = 0
  for i, full_repo_name in enumerate(repo_df.full_name):
    bugfix_filename = _make_bugfix_filename(full_repo_name)
    if os.path.exists(bugfix_filename):
      total_commits += len(pd.read_csv(bugfix_filename))
      continue

    filename = _make_all_commits_filename(full_repo_name)
    try:
      commit_df = pd.read_csv(filename)
    except pd.errors.EmptyDataError:
      pd.DataFrame([]).to_csv(bugfix_filename, index=False)
      continue
    bugfix_commit_df = filter_bugfix_commits(commit_df, soft=True)
    bugfix_commit_df.to_csv(bugfix_filename, index=False)
    total_commits += len(bugfix_commit_df)
  print(f'Total {total_commits:,} bugfix commits found')


def _make_diff_filename(full_repo_name, diff_root=None):
  if diff_root is None:
    diff_root = DIFF_ROOT
  repo_without_slash = str(full_repo_name).replace("/", ".")
  diff_filename = f'{diff_root}/{repo_without_slash}-diff.csv'
  return diff_filename


def _create_diff_csv(
    repo_df: pd.DataFrame, diff_dir: str, bugfix_dir: str, token=None,
):
  if not os.path.exists(diff_dir):
    os.mkdir(diff_dir)

  for i, full_repo_name in repo_df['full_name'].items():
    diff_filename = _make_diff_filename(full_repo_name, diff_dir)
    if os.path.exists(diff_filename):
      continue

    bugfix_filename = _make_bugfix_filename(full_repo_name, bugfix_dir)

    if not os.path.exists(bugfix_filename):
      raise Exception(f'Bugfix file does not exists: {bugfix_filename}')
    commit_df = pd.read_csv(bugfix_filename)
    changed_files = list(get_diffs(commit_df, token))
    pd.DataFrame(changed_files).to_csv(diff_filename, index=False)

  print(f'{current_process().name} ended')


def _make_explainable_commit_filename(dirname, full_repo_name):
  repo_without_slash = str(full_repo_name).replace("/", ".")
  filename = f'{dirname}/{repo_without_slash}-explainable-commits.csv'
  return filename


def _ensure_explainable_commits(patch_count_threshold=5):
  repo_df = pd.read_csv(REPO_FILENAME, usecols=['full_name'])
  dirname = f'{EXPLAINABLE_COMMIT_ROOT}{patch_count_threshold}'
  if not os.path.exists(dirname):
    os.mkdir(dirname)

  def make_filename_local(_full_repo_name: str):
    return _make_explainable_commit_filename(dirname, _full_repo_name)

  explainable_changes_count = 0
  for i, full_repo_name in repo_df['full_name'].items():
    diff_filename = _make_diff_filename(full_repo_name)
    if not os.path.exists(diff_filename):
      raise Exception(f'Diff File Non-existent {i}: {diff_filename}')

    filename = make_filename_local(full_repo_name)
    if os.path.exists(filename):
      try:
        explainable_changes_count += len(pd.read_csv(filename))
      except pd.errors.EmptyDataError:
        pass
      continue

    try:
      diff_df = pd.read_csv(diff_filename)
    except pd.errors.EmptyDataError:
      continue

    """
    Bugfix commit message stats:
    mean  std  min  50%  60%  70%  75%  80%  85%  90%   95%    max
      20   96    1    9   11   14   17   21   29   41    66  10290
    
    Bugfix commit patch stats:
    mean    std  min  50%  60%  70%  75%  80%  85%  90%  95%      max
     349  13498   10   65   80  101  117  138  169  218  318  4512347
    """
    explainable_changes = list(filter_small_bugfix_commits(
      diff_df,
      min_threshold=5,
      msg_max_threshold=30,  # covers > 85%
      patch_count_threshold=patch_count_threshold,
      patch_size_threshold=170,  # covers > 85%
    ))
    if len(explainable_changes) == 0:
      continue
    pd.DataFrame.from_records(explainable_changes).to_csv(filename, index=False)
    explainable_changes_count += len(explainable_changes)

  print(f'Total {explainable_changes_count} explainable changes found.')
  print('Creating train-dev-test split.')

  global_explainable_changes_df = pd.concat(
    pd.read_csv(os.path.join(dirname, fn))
    for fn in os.listdir(dirname)
    if not any(map(lambda prefix: fn.startswith(prefix), ['train-', 'test-', 'valid-']))
  ).sample(frac=1, random_state=4)
  train_df = global_explainable_changes_df[:-20000]
  validation_df = global_explainable_changes_df[-20000:-10000]
  test_df = global_explainable_changes_df[-10000:]

  train_df.to_csv(make_filename_local('train'), index=False)
  validation_df.to_csv(make_filename_local('valid'), index=False)
  test_df.to_csv(make_filename_local('test'), index=False)
  print(f'Train: {len(train_df)}, Validation: {len(validation_df)}, Test: {len(test_df)}')


def remove_added_lines(patch: str):
  buggy_lines = [
    line for line in patch.splitlines()
    if not (line.startswith('+') and not line.startswith('+++'))
  ]

  return '\n'.join(buggy_lines)


def _ensure_diff_pretrain_splits(patch_count_threshold: int):
  print('Making pretrain & finetune splits for explainable commits')
  x_commit_root = f'{EXPLAINABLE_COMMIT_ROOT}{patch_count_threshold}'
  diff_pretrain_root = f'{DIFF_PRETRAIN_ROOT}{patch_count_threshold}'

  if not os.path.exists(diff_pretrain_root):
    os.mkdir(diff_pretrain_root)

  for split_name in ['train', 'test', 'valid']:
    x_commit_filename = _make_explainable_commit_filename(x_commit_root, split_name)
    x_commit_df: pd.DataFrame = pd.read_csv(x_commit_filename)
    pretrain_split: pd.DataFrame = x_commit_df.iloc[:len(x_commit_df) // 2]
    finetune_split: pd.DataFrame = x_commit_df.iloc[len(x_commit_df) // 2:]
    finetune_split['patch'] = finetune_split['patch'].apply(remove_added_lines)
    standard_split = x_commit_df
    standard_split['patch'] = standard_split['patch'].apply(remove_added_lines)
    # assert no patch value is empty
    # assert len(finetune_split) == finetune_split['patch'].astype(bool).sum()

    print(f'{split_name} split.',
          f'Pretrain: {len(pretrain_split)}',
          f'Finetune: {len(finetune_split)}',
          f'Standard: {len(standard_split)}')

    pretrain_split.to_csv(
      os.path.join(diff_pretrain_root, f'{split_name}-diff-pretrain.csv'),
      index=False,
    )
    finetune_split.to_csv(
      os.path.join(diff_pretrain_root, f'{split_name}-diff-finetune.csv'),
      index=False,
    )
    standard_split.to_csv(
      os.path.join(diff_pretrain_root, f'{split_name}-diff-explain.csv'),
      index=False,
    )


def _get_x_commit_filenames_without_content(
    explainable_commit_root: str, content_root: str,
):
  x_commit_filenames = os.listdir(explainable_commit_root)
  existing_content_filenames = os.listdir(content_root)
  handled_x_commit_filenames = [
    filename.replace('changed-file-content', 'explainable-commits')
    for filename in existing_content_filenames
  ]
  x_commit_filenames_without_content = [
    filename for filename in x_commit_filenames
    if filename not in handled_x_commit_filenames
    if not any(map(lambda prefix: filename.startswith(prefix), ['train-', 'test-', 'valid-']))
  ]
  return x_commit_filenames_without_content


def _ensure_file_content_written(
    x_commit_filenames: List[str],
    token: str,
    explainable_commit_root: str,
    content_root: str,
):
  for x_commit_filename in x_commit_filenames:
    x_commit_filepath = os.path.join(explainable_commit_root, x_commit_filename)
    x_commit_df = pd.read_csv(x_commit_filepath)
    response_content = []
    for i, row in x_commit_df.iterrows():
      content_url = row['contents_url']
      try:
        response = get_rate_limited_response(f'{content_url}~1', token)
      except ServerError as se:
        msg = ' | '.join(list(map(str, se.args)))
        msg += f' | Content URL: {content_url}'
        print(msg, file=sys.stderr)
        logger.error(msg)
        continue
      response['fix_commit_sha'] = row['commit_sha']
      response_content.append(response)
    response_df = pd.DataFrame(response_content)
    decoded_contents = []
    for content in response_df.content:
      try:
        decoded_content = base64.b64decode(content).decode(encoding='utf-8')
      except UnicodeDecodeError:
        decoded_content = ''
      decoded_contents.append(decoded_content)
    response_df.content = decoded_contents

    file_content_filename = x_commit_filename.replace(
      'explainable-commits', 'changed-file-content',
    )
    response_df.to_csv(
      os.path.join(content_root, file_content_filename),
      index=False,
    )


def _requires_warehousing(full_repo_name):
  bugfix_filename = _make_bugfix_filename(full_repo_name)
  diff_filename = _make_diff_filename(full_repo_name)
  return os.path.exists(bugfix_filename) and not os.path.exists(diff_filename)


def _spawn_diff_processes():
  if not os.path.exists(DIFF_ROOT):
    os.mkdir(DIFF_ROOT)

  tokens = get_token_list()
  repo_df = pd.read_csv(REPO_FILENAME, usecols=['full_name'])
  target_file_count = len(repo_df)
  repo_df = repo_df[repo_df.full_name.apply(_requires_warehousing)]
  print(f'Warehousing diffs from {len(repo_df)} repositories')
  if len(repo_df) == 0:
    return
  page_size = (len(repo_df) // len(tokens)) + 1

  repo_df_pages = [
    repo_df.iloc[i * page_size: (i + 1) * page_size]
    for i in range(len(tokens))
  ]

  processes = [
    Process(
      name=f'diff-{i}-{i * page_size}:{(i + 1) * page_size}',
      target=_create_diff_csv,
      args=(
        repo_df_pages[i],
        DIFF_ROOT,
        BUGFIX_COMMIT_ROOT,
        token,
      ),
    )
    for i, token in enumerate(tokens)
  ]
  logger_process = create_logger_process(
    dir_name=DIFF_ROOT,
    init_file_count=target_file_count - len(repo_df),
    target_file_count=target_file_count,
    log_every=5 * 60,  # 5 minute
  )

  for process in processes:
    process.start()
  logger_process.start()

  for process in processes:
    process.join()
  for process in processes:
    assert process.exitcode == 0

  logger_process.terminate()
  logger_process.join()


def _ensure_file_content(patch_count_threshold: int):
  tokens = get_token_list()
  explainable_commit_root = f'{EXPLAINABLE_COMMIT_ROOT}{patch_count_threshold}'
  content_root = f'{FILE_CONTENT_ROOT}{patch_count_threshold}'

  if not os.path.exists(content_root):
    os.mkdir(content_root)

  x_commit_filenames_without_content = _get_x_commit_filenames_without_content(
    explainable_commit_root, content_root,
  )

  if len(x_commit_filenames_without_content) == 0:
    print('All file contents ensured')
    return

  print(f'{len(x_commit_filenames_without_content)} repos requires content')

  page_size = math.ceil(len(x_commit_filenames_without_content) / len(tokens))
  filename_chunks = [
    x_commit_filenames_without_content[i:i + page_size]
    for i in range(0, len(x_commit_filenames_without_content), page_size)
  ]

  processes = [
    Process(
      name=f'file_content-{i}_{i * page_size}-{(i + 1) * page_size}',
      target=_ensure_file_content_written,
      args=(filename_chunk, tokens[i], explainable_commit_root, content_root),
    )
    for i, filename_chunk in enumerate(filename_chunks)
  ]
  logger_process = create_logger_process(
    dir_name=content_root,
    init_file_count=len(os.listdir(content_root)),
    target_file_count=len(os.listdir(explainable_commit_root)),
  )

  for process in processes:
    process.start()
  logger_process.start()
  for process in processes:
    process.join()
  for process in processes:
    assert process.exitcode == 0

  logger_process.terminate()
  logger_process.join()

  print('All file contents ensured')


def _ensure_sbt(patch_count_threshold: int):
  fail_count = 0

  def generate_sbt(row: pd.Series):
    nonlocal fail_count
    try:
      sbt = StructureSuperImposer.from_diff(row.content, row.patch).to_bracketed_notation(SBT_DELIMITER)
      sbt = sbt.replace('\n', '<nl>')
      # replace non-printable chars
      sbt = ''.join([ch if ch.isprintable() else f'\\\\x{ord(ch):0x}' for ch in sbt])
      assert len(sbt.split(SBT_DELIMITER)) == 2
      return sbt
    except AstParseError:
      fail_count += 1
      return ''
    except UnidiffParseError:
      fail_count += 1
      return ''

  file_content_root = f'{FILE_CONTENT_ROOT}{patch_count_threshold}'
  explainable_commit_root = f'{EXPLAINABLE_COMMIT_ROOT}{patch_count_threshold}'
  sbt_root = f'{SBT_ROOT}{patch_count_threshold}'
  if not os.path.exists(sbt_root):
    os.mkdir(sbt_root)

  x_commit_filenames = (
    fn for fn in os.listdir(explainable_commit_root)
    if not any(map(lambda prefix: fn.startswith(prefix), ['train-', 'test-', 'valid-']))
  )

  entries_count = 0
  for i, x_commit_filename in enumerate(x_commit_filenames):
    # content files are named as {owner}.{repo}-changed-file-content.csv
    # trimming suffix, the repository name is found
    sbt_filename = x_commit_filename.replace('explainable-commits', 'sbt')
    sbt_filepath = os.path.join(sbt_root, sbt_filename)
    x_commit_filepath = os.path.join(explainable_commit_root, x_commit_filename)
    file_content_filename = x_commit_filename.replace(
      'explainable-commits', 'changed-file-content',
    )
    file_content_filepath = os.path.join(file_content_root, file_content_filename)
    if os.path.exists(sbt_filepath):
      print(f'\r{i}. File exists: {sbt_filepath}', end='')
      existing_sbt_entry_count = len(pd.read_csv(sbt_filepath))
      file_content_entry_count = len(pd.read_csv(file_content_filepath))
      entries_count += existing_sbt_entry_count
      fail_count += (file_content_entry_count - existing_sbt_entry_count)
      continue

    x_commit_df = pd.read_csv(x_commit_filepath, usecols=[
      'commit_sha',
      'commit_node_id',
      'commit_url',
      'commit_message',
      'sha',
      'filename',
      'contents_url',
      'patch',
      'author.date',
      'committer.date',
    ])
    assert x_commit_df.patch.astype(bool).all()

    file_content_cols = ['fix_commit_sha', 'path', 'content']
    file_content_df = pd.read_csv(
      file_content_filepath,
      usecols=file_content_cols,
    ).drop_duplicates()
    file_content_df.content[file_content_df.content.isna()] = ''

    joined_df = pd.merge(
      x_commit_df, file_content_df,
      left_on=['commit_sha', 'filename'],
      right_on=['fix_commit_sha', 'path'],
      how='inner',
    )

    if len(file_content_df) != len(joined_df):
      assert False, f'Content len {len(file_content_df)} != Joined len {len(joined_df)}'

    joined_df['sbt'] = joined_df.apply(generate_sbt, axis=1).astype(str)
    joined_df = joined_df[joined_df['sbt'] != '']

    try:
      joined_df.to_csv(sbt_filepath, index=False, encoding='utf-8')
      print(f'\r{i}. Written: {sbt_filepath}', end='')
      entries_count += len(joined_df)
    except UnicodeEncodeError:
      fail_count += len(joined_df)
      continue

  print(f'\nWritten {entries_count:,} SBTs')
  print(f'Failed {fail_count:,} SBTs')
  print('Creating train-dev-test split.')

  create_sbt_splits(sbt_root)


def get_global_sbt_df_with_count(root: str):
  count_of: dict[str, int] = {}

  def read_file_and_update_stats(filename: str):
    _df = pd.read_csv(
      os.path.join(root, filename),
      parse_dates=['committer.date', 'author.date'],
    )
    # Drop empty values. This is required as some file-contents could not be decoded.
    _df = _df[_df.sbt.astype(bool)]
    count_of[filename] = len(_df)
    _df['repo'] = filename[:-len('-sbt.csv')]
    return _df

  global_sbt_df = pd.concat(
    read_file_and_update_stats(fn)
    for fn in os.listdir(root)
    if not any(map(lambda prefix: fn.startswith(prefix), ['train-', 'test-', 'valid-']))
  )
  return count_of, global_sbt_df


def write_splits(
    global_df: pd.DataFrame,
    root: str,
    description: str,
    *,
    test_size=10000,
    non_null_column=None,
    ensure_unique_projects=False,
    verbose=False,
):
  train_df = global_df[:-2 * test_size]
  validation_df = global_df[-2 * test_size:-test_size]
  test_df = global_df[-test_size:]

  if ensure_unique_projects:
    validation_df = validation_df[validation_df.repo.apply(lambda repo: repo not in test_df.repo)]
    train_df = train_df[train_df.repo.apply(lambda repo: repo not in validation_df.repo)]

  if non_null_column:
    train_df = train_df[train_df[non_null_column].astype(bool)]
    validation_df = validation_df[validation_df[non_null_column].astype(bool)]
    test_df = test_df[test_df[non_null_column].astype(bool)]

  train_df.to_csv(os.path.join(root, f'train-{description}.csv'), index=False)
  validation_df.to_csv(os.path.join(root, f'valid-{description}.csv'), index=False)
  test_df.to_csv(os.path.join(root, f'test-{description}.csv'), index=False)

  print('Written:', description)
  if verbose:
    print(f'Train: {len(train_df)}, Validation: {len(validation_df)}, Test: {len(test_df)}')

  return train_df, test_df, validation_df


def create_sbt_splits(root: str):
  count_of, global_sbt_df = get_global_sbt_df_with_count(root)

  global_sbt_df = global_sbt_df.sample(frac=1, random_state=4)
  write_splits(global_sbt_df, root, 'sbt-random', verbose=True)
  global_sbt_df.sort_values(by='committer.date', inplace=True)
  write_splits(global_sbt_df, root, 'sbt-time')

  test_projects, val_projects = [], []
  test_instance_count, val_instance_count = 0, 0
  train_projects = list(count_of.keys())
  random.seed(4)
  random.shuffle(train_projects)
  while test_instance_count < 10000 or val_instance_count < 10000:
    project_name = train_projects.pop()
    if test_instance_count < 10000:
      test_projects.append(project_name)
      test_instance_count += count_of[project_name]
    else:
      val_projects.append(project_name)
      val_instance_count += count_of[project_name]

  print('Creating project splits')
  for split, projects in {'train': train_projects, 'test': test_projects, 'valid': val_projects}.items():
    split_dfs = []
    for fn in projects:
      df = pd.read_csv(os.path.join(root, fn))
      df['repo'] = fn[:-len('-sbt.csv')]
      split_dfs.append(df)
    split_df = pd.concat(split_dfs)
    split_df.to_csv(os.path.join(root, f'{split}-sbt-project.csv'), index=False)
    print(f'{split.title()}: {len(split_df)} from {len(projects)} projects')


def make_source_sbt(row: pd.Series):
  splits = row.sbt.split(SBT_DELIMITER)
  assert len(splits) == 2
  return splits[0]


def _ensure_large_sbt_pretrain_splits(patch_count_threshold: int):
  print('Making large pretrain & finetune splits for SBT')
  sbt_root = f'{SBT_ROOT}{patch_count_threshold}'
  sbt_pretrain_root = f'{SBT_PRETRAIN_ROOT}{patch_count_threshold}-large'

  if not os.path.exists(sbt_pretrain_root):
    os.mkdir(sbt_pretrain_root)

  global_sbt_df: dict[str, pd.DataFrame] = {}
  count_of, global_sbt_df['pretrain'] = get_global_sbt_df_with_count(sbt_root)
  global_sbt_df['pretrain'] = global_sbt_df['pretrain'].sample(frac=1, random_state=4)
  global_sbt_df['pretrain']['patch'] = global_sbt_df['pretrain']['patch']
  global_sbt_df['finetune'] = global_sbt_df['pretrain'].copy()
  global_sbt_df['finetune']['sbt'] = global_sbt_df['finetune'].apply(make_source_sbt, axis=1)
  global_sbt_df['finetune']['patch'] = global_sbt_df['finetune']['patch'].apply(remove_added_lines)

  print(f"{len(global_sbt_df['pretrain']):,} entries read")

  for suffix in ['random', 'project', 'time']:
    if suffix == 'time':
      pretrain_df = global_sbt_df['pretrain'].sort_values(by='committer.date')
      finetune_df = global_sbt_df['finetune'].sort_values(by='committer.date')
    elif suffix == 'project':
      pretrain_df = global_sbt_df['pretrain'].sort_values(by='repo')
      finetune_df = global_sbt_df['finetune'].sort_values(by='repo')
    else:
      pretrain_df = global_sbt_df['pretrain']
      finetune_df = global_sbt_df['finetune']

    # drop last 20,000 in pretrain; those are for fine tune testing and validation
    # drop second last 20,000 in finetune; those are for pretrain

    pretrain_df = pretrain_df.iloc[:-20000]
    explain_df = finetune_df
    finetune_df = pd.concat([finetune_df.iloc[:-40000], finetune_df.iloc[-20000:]])

    write_splits(
      pretrain_df, sbt_pretrain_root, f'sbt-{suffix}-pretrain',
      test_size=10000, ensure_unique_projects=suffix == 'project', verbose=True,
    )
    write_splits(
      finetune_df, sbt_pretrain_root, f'sbt-{suffix}-finetune',
      test_size=10000, non_null_column='sbt', ensure_unique_projects=suffix == 'project', verbose=True,
    )
    write_splits(
      explain_df, sbt_pretrain_root, f'sbt-{suffix}-explain',
      test_size=10000, non_null_column='sbt', ensure_unique_projects=suffix == 'project', verbose=True,
    )


def _ensure_sbt_pretrain_splits(patch_count_threshold: int, large=False):
  if large:
    return _ensure_large_sbt_pretrain_splits(patch_count_threshold)

  print('Making pretrain & finetune splits for SBT')
  sbt_root = f'{SBT_ROOT}{patch_count_threshold}'
  sbt_pretrain_root = f'{SBT_PRETRAIN_ROOT}{patch_count_threshold}'

  if not os.path.exists(sbt_pretrain_root):
    os.mkdir(sbt_pretrain_root)

  for suffix in ['random', 'time', 'project']:
    for split_name in ['train', 'test', 'valid']:
      sbt_filename = os.path.join(sbt_root, f'{split_name}-sbt-{suffix}.csv')
      if suffix == 'time':
        sbt_df: pd.DataFrame = pd.read_csv(
          sbt_filename, parse_dates=['committer.date']
        )
        sbt_df.sort_values(by='committer.date', inplace=True)
      elif suffix == 'project':
        sbt_df: pd.DataFrame = pd.read_csv(sbt_filename)
        sbt_df.sort_values(by='repo', inplace=True)
      else:
        sbt_df: pd.DataFrame = pd.read_csv(sbt_filename)
      pretrain_split: pd.DataFrame = sbt_df.iloc[:len(sbt_df) // 2]
      finetune_split: pd.DataFrame = sbt_df.iloc[len(sbt_df) // 2:]

      if suffix == 'project':
        # ensure no pretrain repo is in finetune
        # this can happen for at most one repo, as we already sorted by repo
        finetune_split = finetune_split[
          finetune_split['repo'].apply(lambda repo: repo not in pretrain_split['repo'])
        ]

      finetune_split['sbt'] = finetune_split.apply(make_source_sbt, axis=1)
      standard_split = sbt_df
      standard_split['sbt'] = standard_split.apply(make_source_sbt, axis=1)

      filter_log_and_write_splits(
        standard_split, finetune_split, pretrain_split, sbt_pretrain_root, split_name, suffix,
      )


def filter_log_and_write_splits(
    standard_split: pd.DataFrame,
    finetune_split: pd.DataFrame,
    pretrain_split: pd.DataFrame,
    sbt_pretrain_root: str,
    split_name: str,
    suffix: str,
):
  pretrain_split = pretrain_split[pretrain_split['sbt'].astype(bool)]
  finetune_split = finetune_split[finetune_split['sbt'].astype(bool)]
  standard_split = standard_split[standard_split['sbt'].astype(bool)]
  print(f'{split_name} {suffix} split.',
        f'Pretrain: {len(pretrain_split)}',
        f'Finetune: {len(finetune_split)}',
        f'Standard: {len(standard_split)}')
  pretrain_split.to_csv(
    os.path.join(sbt_pretrain_root, f'{split_name}-sbt-{suffix}-pretrain.csv'),
    index=False,
  )
  finetune_split.to_csv(
    os.path.join(sbt_pretrain_root, f'{split_name}-sbt-{suffix}-finetune.csv'),
    index=False,
  )
  standard_split.to_csv(
    os.path.join(sbt_pretrain_root, f'{split_name}-sbt-{suffix}-explain.csv'),
    index=False,
  )


if __name__ == '__main__':
  assert os.getcwd().endswith('scrap-github')

  DATASET_DIR = '../dataset/scrap-github'
  NUM_REPOS = 10000
  REPO_FILENAME = f'{DATASET_DIR}/top_{NUM_REPOS}_repos_python-non-archived.csv'
  if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

  ALL_COMMIT_ROOT = f'{DATASET_DIR}/all-commits'
  BUGFIX_COMMIT_ROOT = f'{DATASET_DIR}/soft-bugfix-commits'
  DIFF_ROOT = f'{DATASET_DIR}/diff-warehouse'
  EXPLAINABLE_COMMIT_ROOT = f'{DATASET_DIR}/explainable-commits'
  DIFF_PRETRAIN_ROOT = f'{DATASET_DIR}/diff-pretrain'
  FILE_CONTENT_ROOT = f'{DATASET_DIR}/changed-file-content'
  SBT_DELIMITER = '</s>'
  SBT_ROOT = f'{DATASET_DIR}/sbt'
  SBT_PRETRAIN_ROOT = f'{DATASET_DIR}/sbt-pretrain'

  _ensure_repository_list()
  _spawn_all_commits_processes()
  _ensure_bugfix_commits()
  _spawn_diff_processes()
  # `_ensure_explainable_commits` only writes non-empty files
  _ensure_explainable_commits(patch_count_threshold=1)
  _ensure_diff_pretrain_splits(patch_count_threshold=1)
  _ensure_file_content(patch_count_threshold=1)
  _ensure_sbt(patch_count_threshold=1)
  _ensure_sbt_pretrain_splits(patch_count_threshold=1, large=True)
