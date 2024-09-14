import argparse
import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import List, Any, Optional

import requests
import yaml
from openai import OpenAI
from requests import JSONDecodeError

from horde_openai_proxy import (
    horde_to_openai,
    HordeRequest,
    ModelGenerationInput,
)
from horde_openai_proxy.utils import (
    apply_kobold_formatting_from_payload,
)


@dataclass
class Config:
    api_key: str = ""
    priority_usernames: List[str] = field(default_factory=list)
    interval: float = 1.0
    stats_interval: float = 1.0
    name: str = "My Ollama Worker"
    bridge_agent: str = (
        "Horde-OpenAI-Proxy:0:https://github.com/Luke100000/horde-openai-proxy"
    )
    backend: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    backend_api_key: str = ""
    backend_url: str = "http://localhost:11434/v1"
    horde_url: str = "https://stablehorde.net"
    models: List[Any] = field(default_factory=lambda: ["all"])
    nsfw: bool = True
    require_upfront_kudos: bool = True
    extra_slow_worker: bool = False
    max_length: int = 512
    max_context_length: int = 2048
    logging_level: int = 20
    parallelism: int = 4
    queue: int = 2


def load_config(yaml_file: str) -> Config:
    with open(yaml_file, "r") as f:
        config_data = yaml.safe_load(f)
    return Config(**config_data)


class AtomicCounter:
    def __init__(self, initial=0):
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, value: int = 1):
        with self._lock:
            self._value += value
            return self._value

    def decrement(self):
        self.increment(-1)

    def get(self):
        return self._value

    def set(self, value: int):
        with self._lock:
            self._value = value
            return self._value


@dataclass
class Task:
    uuid: str
    model: str
    payload: dict


@dataclass
class Generation:
    generation: str
    time: float
    prefill_tps: float
    generation_tps: float


class OpenAIBridge:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)

        logging.basicConfig(level=self.config.logging_level)
        self.logger = logging.getLogger(__name__)

        self.run = True
        self.start_time = time.time()
        self.last_stats_time = time.time()

        # General stats
        self.processing = AtomicCounter()
        self.failed = AtomicCounter()
        self.submitted = AtomicCounter()
        self.failed_in_a_row = AtomicCounter()
        self.kudos = AtomicCounter()
        self.task_queue = Queue()

        # Utilization stats
        self.idle = 0
        self.utilization = 0
        self.fetches = 0

        # OpenAI client to forward requests to
        self.client = OpenAI(
            api_key=self.config.backend_api_key,
            base_url=self.config.backend_url,
            max_retries=3,
        )

        # Fetcher and worker threads
        self.fetcher = threading.Thread(target=self.fetch_loop)
        self.workers = []
        for _ in range(self.config.parallelism):
            worker = threading.Thread(target=self.request_loop)
            self.workers.append(worker)

    def request(self, payload: dict, model: str) -> str:
        try:
            horde_payload = HordeRequest(
                prompt=payload["prompt"],
                models=[model],
                timeout=300,
                params=ModelGenerationInput(**payload),
            )
        except Exception as e:
            self.logger.error(f"Failed to convert payload: {e}, {payload}")
            raise

        openai = horde_to_openai(horde_payload)
        completion = self.client.chat.completions.create(**openai.model_dump())
        generation = apply_kobold_formatting_from_payload(
            completion.choices[0].message.content, horde_payload.params
        )

        self.logger.info("  Prompt: " + payload["prompt"].replace("\n", " "))
        self.logger.info("  Generation: " + generation.replace("\n", " "))

        return generation

    def request_loop(self):
        while self.run:
            try:
                task = self.task_queue.get(timeout=1.0)
            except Empty:
                continue
            else:
                if task is not None:
                    self.processing.increment()
                    start_time = time.time()
                    try:
                        result = self.request(task.payload, task.model)
                        delta = time.time() - start_time
                        generation = Generation(
                            generation=result,
                            time=delta,
                            prefill_tps=len(task.payload["prompt"]) / delta / 4,
                            generation_tps=len(result) / delta / 4,
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to generate: {e}")
                        generation = None

                    self.submit(task.uuid, generation)
                    self.processing.decrement()

    def fetch(self):
        headers = {"apikey": self.config.api_key}

        gen_dict = {
            "name": self.config.name,
            "priority_usernames": self.config.priority_usernames,
            "nsfw": self.config.nsfw,
            "models": self.config.models,  # TODO: prefer loaded ones
            "bridge_agent": self.config.bridge_agent,
            "threads": self.config.parallelism,
            "require_upfront_kudos": self.config.require_upfront_kudos,
            "amount": 1,
            "extra_slow_worker": self.config.extra_slow_worker,
            "max_length": self.config.max_length,
            "max_context_length": self.config.max_context_length,
            "softprompts": [],
        }

        pop_request = requests.post(
            self.config.horde_url + "/api/v2/generate/text/pop",
            headers=headers,
            json=gen_dict,
            timeout=60,
        )

        if pop_request.ok:
            pop = pop_request.json()
            if pop["ids"]:
                # Enqueue jobs
                payload = pop["payload"]
                self.logger.info(
                    f"Job received with {payload.get('n', 1)}x {len(payload['prompt'])} characters and {payload.get('max_length', self.config.max_length)} output tokens."
                )
                self.task_queue.put(
                    Task(uuid=pop["id"], model=pop["model"], payload=payload)
                )
            else:
                # No jobs available
                reasons = [
                    f"{reason}: {count}"
                    for reason, count in pop["skipped"].items()
                    if count > 0
                ]
                self.logger.debug(f"No jobs available: {', '.join(reasons)}")
                time.sleep(self.config.interval)
        else:
            try:
                message = pop_request.json().get("message")
            except (JSONDecodeError, KeyError):
                message = pop_request.status_code
            self.logger.error(f"Pop request failed: {message}")
            time.sleep(self.config.interval)

    def submit(self, uuid: str, generation: Optional[Generation]):
        headers = {"apikey": self.config.api_key}
        submit_dict = {
            "id": uuid,
            "state": "faulted" if generation is None else "ok",
            "generation": generation.generation if generation is not None else "",
            "seed": -1,
        }
        submit_request = requests.post(
            self.config.horde_url + "/api/v2/generate/text/submit",
            headers=headers,
            json=submit_dict,
        )
        if submit_request.status_code == 404:
            self.logger.warning("The generation got stale.")
        elif not submit_request.ok:
            try:
                message = submit_request.json().get("message")
            except (JSONDecodeError, KeyError):
                message = submit_request.status_code
            self.logger.error(f"Failed to submit generation: {message}")
        else:
            if generation is None:
                self.logger.info("Faulted generation submitted.")
                self.failed.increment()
                self.failed_in_a_row.increment()
                if self.failed_in_a_row.get() > 3:
                    self.logger.error(
                        "Too many failed generations in a row, aborting..."
                    )
                    self.run = False
            else:
                reward = submit_request.json()["reward"]
                self.logger.info(
                    f"Generation submitted for {reward} Kudos, took {generation.time:.2f}s at {generation.prefill_tps:.1f} prefill tps, {generation.generation_tps:.1f} generation tps, {reward / generation.time:.2f} Kudos/s."
                )
                self.submitted.increment()
                self.failed_in_a_row.set(0)

    def fetch_loop(self):
        while self.run:
            self.fetches += 1
            if self.task_queue.empty():
                self.idle += 1
            self.utilization += self.processing.get() / self.config.parallelism

            # Check if we have enough capacity to start a new generation
            amount = (
                self.config.parallelism
                + self.config.queue
                - self.task_queue.qsize()
                - self.processing.get()
            )
            if amount < 0:
                self.logger.debug("No capacity for new generation")
                time.sleep(self.config.interval)
                continue

            try:
                self.fetch()
            except Exception as e:
                self.logger.error(f"Failed to fetch: {e}")
                time.sleep(self.config.interval)

            # Print stats
            if time.time() - self.last_stats_time > self.config.stats_interval:
                self.last_stats_time = time.time()
                delta = time.time() - self.start_time
                self.logger.info(
                    f"Submitted {self.submitted.get()}, failed {self.failed.get()}"
                )
                self.logger.info(
                    f"Earned {self.kudos.get()} Kudos, at {int(self.kudos.get() / delta * 3600)} Kudos/h."
                )
                self.logger.info(
                    f"Idles {int(self.idle / self.fetches * 100)}%, {int(self.utilization / self.fetches * 100)}% utilization."
                )

    def get_models(self) -> tuple[list[str], list[str]]:
        if self.config.backend == "ollama":
            return self.get_ollama_models()
        else:
            return self.config.models, self.config.models

    @staticmethod
    def _parse_ollama_models(response: requests.Response) -> list[str]:
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]

    def get_ollama_models(self) -> tuple[list[str], list[str]]:
        # Get locally available models
        if "all" in self.config.models:
            available_models = self._parse_ollama_models(
                requests.get(f"{self.config.ollama_url}/api/tags")
            )
        else:
            available_models = self.config.models

        # Get loaded models
        loaded_models = self._parse_ollama_models(
            requests.get(f"{self.config.ollama_url}/api/ps")
        )

        return available_models, loaded_models

    def sanity_check(self):
        models, loaded_models = self.get_models()

        self.logger.info("Beginning sanity check...")
        self.logger.info(f"Available models: {models}")
        self.logger.info(f"Loaded models: {loaded_models}")

        if not models:
            self.logger.error("No models available, aborting...")
            return False

        payload = {
            "frmtrmblln": True,
            "frmtrmspch": True,
            "frmttriminc": True,
            "max_context_length": 1024,
            "max_length": 50,
            "rep_pen": 0.0,
            "singleline": True,
            "temperature": 0.0,
            "stop_sequence": ["<>"],
            "prompt": "Say 'Horde' 10 times!",
        }

        # Check if models can generate the prompt
        for model in models:
            try:
                result = self.request(payload, model)
                self.logger.info(f"Sanity check result for {model}: {result}")
                if "Horde" not in result:
                    self.logger.warning(f"Sanity check for {model} failed!")
            except Exception as e:
                self.logger.error(f"Sanity check for {model} failed: {e}")
                return False

        return True

    def start(self) -> None:
        if not self.sanity_check():
            return

        self.fetcher.start()
        for worker in self.workers:
            worker.start()

        self.logger.info("Bridge started")

    def stop(self):
        self.logger.info("Stopping bridge...")
        self.run = False

        self.fetcher.join()
        for worker in self.workers:
            worker.join()

        self.logger.info("Bridge stopped")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c",
        "--config_path",
        action="store",
        required=False,
        type=str,
        default="bridge_config.yaml",
        help="The amount of seconds with which to check if there's new prompts to generate",
    )

    args = arg_parser.parse_args()

    bridge = OpenAIBridge(args.config_path)

    try:
        bridge.start()
    except KeyboardInterrupt:
        bridge.stop()
