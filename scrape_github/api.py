import configparser
import pathlib
import sys
import time
from datetime import datetime
from multiprocessing import current_process
from typing import Union

import requests
from requests import Response
from requests.adapters import Retry

config = configparser.ConfigParser()
config.read(pathlib.Path(__file__).parent / 'github-token.cfg', encoding='utf-8')
assert config["default"]["githubToken"], config["default"]["githubToken"]
headers = {
  'Authorization': f'Bearer {config["default"]["githubToken"]}'
}

requests.session().get_adapter('https://').max_retries = Retry(
  total=10,
  backoff_factor=0.1,
)

ROOT = 'https://api.github.com'


class ServerError(Exception):
  pass


def get_response_passing_connection_error(
    path: str, _headers: dict,
) -> Response:
  num_retry = 10
  for i in range(num_retry):
    try:
      response_obj = requests.get(path, headers=_headers)
      if response_obj.status_code < 500:
        if response_obj.status_code == 404 and '+' in path:
          # replace '+' with an encoded space
          path = path.replace('+', '%20')
          continue
        else:
          return response_obj

      print(
        response_obj.status_code,
        response_obj.text,
        path,
        file=sys.stderr,
      )
      print(f'{current_process().name} Sleeping for 1 minute...')
      time.sleep(60)
      print(f'{current_process().name} Woke up.')
    except requests.exceptions.RequestException as e:
      # handles requests.exceptions.ConnectionError
      # and requests.exceptions.ChunkedEncodingError
      print(e, file=sys.stderr)
      print(f'{current_process().name} Sleeping for 1 hour...')
      time.sleep(60 * 60)
      print(f'{current_process().name} Woke up.')

  # noinspection PyUnboundLocalVariable
  raise ServerError(
    f'Could not load resource in {num_retry} tries | '
    f'{response_obj.status_code} | '
    f'{response_obj.text} | {path}'
  )


def get_rate_limited_response(
    path: str,
    token=None,
    accept='application/vnd.github.v3+json',
) -> Union[dict, list, str]:
  if token:
    _headers = {
      'Authorization': f'Bearer {token}'
    }
  else:
    _headers = headers.copy()
  _headers['Accept'] = accept

  response_obj = get_response_passing_connection_error(path, _headers)

  if 'x-ratelimit-remaining' not in response_obj.headers:
    assert False, headers
  limit_remaining = int(response_obj.headers['x-ratelimit-remaining'])
  if limit_remaining <= 1:
    now = datetime.now()
    limit_reset = int(response_obj.headers['x-ratelimit-reset'])
    limit_reset = datetime.fromtimestamp(limit_reset)
    delta = (limit_reset - now)
    print(
      f'{current_process().name} Sleeping for {str(delta).split(".")[0]}...',
    )
    time.sleep(delta.total_seconds() + 10)  # sleeping extra ten seconds
    print(f'{current_process().name} Woke up.')
    response_obj = get_response_passing_connection_error(path, _headers)

  if 'json' in response_obj.headers['Content-Type']:
    response = response_obj.json()
  else:
    return response_obj.text

  while True:
    if not isinstance(response, dict) or 'message' not in response:
      break

    if 'Bad credential' in response['message']:
      raise Exception(f'Bad Credential: {token}')

    # test for second rate limit
    if 'secondary rate limit' in response['message']:
      print(current_process().name, 'Sleeping for 5 minutes...', end=' ')
      time.sleep(5 * 60)
      print('Woke up.')
    elif (
        response['message'].startswith('Server Error')
        or response['message'].startswith('This API returns blobs up to 1 MB')
    ):
      raise ServerError(response['message'], response['errors'], path)
    elif response['message'].startswith('Repository access blocked'):
      raise ServerError(response['message'], response['block'], path)
    else:
      msg = f"Unhandled response message: {response['message']} | {path}"
      raise Exception(msg)

    response_obj = get_response_passing_connection_error(path, _headers)

    if response_obj.headers['Content-Type'].endswith('json'):
      response = response_obj.json()
    else:
      return response_obj.text

  return response


def get_token_list():
  _config = configparser.ConfigParser()
  _config.read('github-token.cfg', encoding='utf-8')
  tokens = list(_config["default"].values())
  return tokens
