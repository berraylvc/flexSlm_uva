from typing import Iterable, Any, overload, Union
from contextlib import contextmanager
import pickle
import io
import os

from torch import nn
from flex_modules.module import Module, LevelDelta, UpDelta, DownDelta

from networks.config import ModelConfig
from networks.flex_model import FlexModel
import utils


def get_model_deltas(model: Module) -> dict[tuple[int, bool], LevelDelta]:
    deltas = dict()

    for level in range(model.max_level() + 1):
        model.set_level_use(level)
        delta_down, delta_up = model.export_level_delta()
        if level > 0:
            deltas[(level, True)] = delta_up.cpu().clone().detach()
        if level < model.max_level():
            deltas[(level, False)] = delta_down.cpu().clone().detach()

    return deltas


class BaseDeltaManager:
    """
    Defines an interface for all delta managers to follow. A delta manager creates a
    regular model and applies deltas to adapt its level like it is a flexible model.
    """

    def __init__(self, max_level, current_level):
        self._current_level = current_level
        self._max_level = max_level

    def max_level(self):
        return self._max_level

    def current_level(self):
        return self._current_level

    def get_level_delta(self, level: int, up: bool) -> LevelDelta:
        raise NotImplementedError()

    def managed_model(self) -> nn.Module:
        raise NotImplementedError()

    def set_managed_model(self, model: nn.Module):
        raise NotImplementedError()

    def move_to(self, target_level: int) -> None:
        """
        Moves the managed model to the desired level, and
        then returns said model.
        """
        current_level = self._current_level
        if target_level > current_level:
            for i in range(current_level + 1, target_level + 1):
                self.get_level_delta(i, True).clone(
                ).detach().apply(self.managed_model())
        elif target_level < current_level:
            for i in range(current_level - 1, target_level - 1, -1):
                self.get_level_delta(i, False).clone(
                ).detach().apply(self.managed_model())
        self._current_level = target_level
        return self.managed_model()


class InMemoryDeltaManager(BaseDeltaManager):
    def __init__(self, flexible_model: Module, starting_level: int = -1) -> None:
        """
        Takes a flexible model and loads all its deltas along with a copy of the model
        as a regular model at the specified level (by
        default the level the flexible model is currently on).
        """
        if starting_level == -1:
            starting_level = flexible_model.current_level()
        super().__init__(flexible_model.max_level(), starting_level)
        self.deltas = get_model_deltas(flexible_model)
        cpy_level = flexible_model.current_level()
        flexible_model.set_level_use(starting_level)
        self.managed = flexible_model.make_base_copy()
        flexible_model.set_level_use(cpy_level)

    def managed_model(self) -> nn.Module:
        return self.managed

    def set_managed_model(self, model: nn.Module):
        self.managed = model

    def get_level_delta(self, level: int, up: bool) -> Iterable[LevelDelta]:
        return self.deltas[(level, up)]


class FileDeltaManager(BaseDeltaManager):
    def __init__(self, file: io.BufferedIOBase, managed_config: ModelConfig):
        """
        Takes in a delta file as genererated by
        `FileDeltaManager.make_delta_file`. It loads in the saved model to a regular
        model as created by `managed_config` (Make sure this config matches the saved model).
        """
        self._file = file
        self._locations = pickle.load(file)
        self._locations.append(-1)
        self._data_start = file.tell()
        self._managed_config = managed_config

        maxlevel, level, sdict = self._deserialize_chunk(0, self._locations[0])
        self._managed_model = managed_config.make_model()
        self._managed_model.load_state_dict(sdict)
        super().__init__(maxlevel, level)

    def managed_model(self):
        return self._managed_model

    def set_managed_model(self, model: nn.Module):
        self._managed_model = model

    def get_level_delta(self, level: int, up: bool) -> Iterable[LevelDelta]:
        idx = self.max_level() - ((-1) ** up) * level - 1
        return self._deserialize_chunk(self._locations[idx], self._locations[idx + 1])

    def _deserialize_chunk(self, start, end):
        return utils.torch_deserialize(self._read_chunk(start, end))

    def _read_chunk(self, start, end):
        pos = self._data_start + start
        if self._file.tell() != pos:
            self._file.seek(self._data_start + start)
        size = end - start
        if size < 0:
            size = -1
        return self._file.read(size)

    @staticmethod
    def make_delta_file(file: io.BufferedIOBase, model: Module, starting_level: int = -1) -> None:
        """
        Writes a delta file to `file`. This file contains the a regular version
        of model at `starting_level`, and all its deltas. This file can later be used as
        input for a `FileDeltaManager` to load a regular model and let it switch between levels.
        """
        locations = []
        deltas = get_model_deltas(model)

        if starting_level == -1:
            starting_level = model.current_level()
        f = io.BytesIO()
        cpy_level = model.current_level()
        model.set_level_use(starting_level)
        reg_model = model.make_base_copy().cpu()
        model.set_level_use(cpy_level)
        f.write(utils.torch_serialize(
            (model.max_level(), starting_level, reg_model.state_dict())))

        for i in range(model.max_level() - 1, -1, -1):
            d = deltas[(i, False)]
            serialized = utils.torch_serialize(d)
            locations.append(f.tell())
            f.write(serialized)

        for i in range(1, model.max_level() + 1):
            d = deltas[(i, True)]
            serialized = utils.torch_serialize(d)
            locations.append(f.tell())
            f.write(serialized)

        pickle.dump(locations, file)
        file.write(f.getvalue())


@contextmanager
def file_delta_manager(path, managed_config):
    """
    Opens a `FileDeltaManager` with a file from path `path`.
    """
    with open(path, 'rb') as f:
        yield FileDeltaManager(f, managed_config)
