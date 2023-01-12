from multiprocessing import Process, Queue
import os
from pathlib import Path
import shutil

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
import wandb

from information_extraction.config import WANDB_PROJECT, DATA_DIR


NUM_WORKERS = 32
TIMEOUT = 15


class SeleniumProcess(Process):
    def __init__(self, task_queue: Queue, stat_queue: Queue, input_dir: Path, output_dir: Path):
        super().__init__()

        self.task_queue = task_queue
        self.stat_queue = stat_queue
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self) -> None:
        options = webdriver.FirefoxOptions()
        options.headless = True
        options.set_capability('pageLoadStrategy', 'eager')
        options.set_preference('javascript.enabled', False)

        with webdriver.Firefox(options=options) as driver:
            driver.set_page_load_timeout(TIMEOUT)

            while True:
                next_task = self.task_queue.get()

                if next_task is None:
                    break

                try:
                    self.process_file(driver, next_task)
                    self.stat_queue.put('success')
                except Exception as e:
                    self.stat_queue.put(f'Error for {next_task}: {e}')

            return

    def process_file(self, driver: webdriver.Firefox, filename: str):
        output_file = self.output_dir / filename
        if output_file.exists():
            return

        url = f'file://{str((self.input_dir / filename).absolute())}'

        try:
            driver.get(url)
        except:
            pass

        for element in driver.find_elements(By.TAG_NAME, '*'):
            position_information = {**element.location, **element.size}

            for key, value in position_information.items():
                # Add data-x, data-y, etc. to the DOM tree
                driver.execute_script('arguments[0].setAttribute(arguments[1],arguments[2])', element, f'data-{key}',
                                      value)

        html = driver.execute_script('return document.documentElement.outerHTML')

        output_file.parent.mkdir(exist_ok=True, parents=True)

        with open(output_file, 'w') as _file:
            _file.write(html)


def status_bar(stat_queue: Queue, total: int):
    encountered = 0

    with tqdm(desc='File', total=total) as pbar:
        while encountered < total:
            message = stat_queue.get()
            pbar.update()
            encountered += 1

            if message != 'success':
                pbar.write(message)


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name='add-pos-to-swde',
        job_type='preprocess',
    )

    input_dir = DATA_DIR / 'SWDE'
    output_dir = DATA_DIR / 'SWDE-pos'

    dataset_artifact = wandb.use_artifact('swde:latest')
    dataset_artifact.download(str(input_dir))

    task_queue = Queue()
    total = 0

    for parent_dir, _, filenames in os.walk(input_dir):
        parent_path = Path(parent_dir)
        for filename in filenames:
            if filename.endswith('.htm'):
                file_path = (parent_path / filename).relative_to(input_dir)

                if not (output_dir / file_path).exists():
                    task_queue.put(str(file_path))
                    total += 1

    for _ in range(NUM_WORKERS):
        task_queue.put(None)

    stat_queue = Queue()

    processes = [SeleniumProcess(task_queue, stat_queue, input_dir, output_dir) for _ in range(NUM_WORKERS)]
    processes.append(Process(target=status_bar, args=(stat_queue, total)))

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    shutil.copytree(input_dir / 'groundtruth', output_dir / 'groundtruth', dirs_exist_ok=True)

    artifact = wandb.Artifact('swde-pos', type='dataset',
                              description='An augmented version of the SWDE dataset, containing position '
                                          'information in each HTML tag')
    artifact.add_dir(str(output_dir))

    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == '__main__':
    main()
