import logging
import os
import pathlib
import time
from datetime import datetime
from multiprocessing import Process, current_process

curr_dir = pathlib.Path(__file__).parent
if not os.path.exists(curr_dir / 'logs'):
  os.mkdir(curr_dir / 'logs')

logger = logging.getLogger(__name__)
sanitized_process_name = current_process().name.split('-')[0]
logger_name = curr_dir / f'logs/error-{sanitized_process_name}.log'
logger.addHandler(logging.FileHandler(logger_name))


def create_logger(
    dir_name: str,
    init_count: int,
    target_count: int,
    log_every=60,
):
  tick = datetime.now()

  while True:
    time.sleep(log_every)
    file_count = len(os.listdir(dir_name))
    complete_count = (file_count - init_count)
    job_count = (target_count - init_count)
    progress_frac = complete_count / job_count
    tock = datetime.now()
    elapsed = tock - tick
    remaining_count = job_count - complete_count
    remaining_time = (elapsed / (complete_count + 1e-9)) * remaining_count
    print(
      f'\r{file_count} ({file_count - init_count}) / {target_count} '
      f'({progress_frac * 100:.2f}%) | '
      f'{str(elapsed).split(".")[0]} <- {str(remaining_time).split(".")[0]} | '
      f'{dir_name}',
      end='',
    )


def create_logger_process(
    dir_name: str,
    init_file_count: int,
    target_file_count: int,
    log_every=60,
):
  assert os.path.exists(curr_dir / dir_name), f'{dir_name} does not exist'
  assert target_file_count > init_file_count

  process_name = os.path.basename(dir_name) or os.path.basename(os.path.dirname(dir_name))
  return Process(
    name=process_name,
    target=create_logger,
    args=(dir_name, init_file_count, target_file_count, log_every),
  )
