from collections import deque
from typing import List, Tuple, Iterable, cast, Dict, Deque
import logging
import random
import math

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("proportional")
class ProportionalIterator(DataIterator):
    """
    Given a binary classification task, where the class ratio is p:1-p and p<=1-p, this iterator forces each batch
    to have ceiling(p) and floor(1-p) examples of the corresponding classes.

    param:num_cycles will decide how many times to loop through the smaller class before ending the epoch (the epoch
    will end automatically after num_cycles is satisfied for the smaller class, and the bigger class was iterated
    through once).

    Shuffle is always turned on. If you wish to disable it, you can do so through the source code, but that is not
    recommended due to how proportional batching works.
    """

    def __init__(self,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None,
                 num_cycles: int = 1) -> None:
        super().__init__(cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self.num_cycles = num_cycles

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        """
        Given a binary classification task, where the class ratio is p:1-p and p<=1-p, this iterator forces each batch
        to have ceiling(p) and floor(1-p) examples of the corresponding classes.

        Shuffle is always turned on. If you wish to disable it, you can do so through the source code, but that is not
        recommended due to how proportional batching works.
        """
        for instance_list in self._memory_sized_lists(instances):
            shuffle = True

            if shuffle:
                random.shuffle(instance_list)

            instance_lists = {}

            for inst in instance_list:
                if inst['label'].label not in instance_lists:
                    instance_lists[inst['label'].label] = []

                assert len(instance_lists) <= 2, f"The Proportional Iterator class currently only supports binary" \
                    f"tasks with two labels. Found more than two labels: {list(instance_lists)}"

                instance_lists[inst['label'].label].append(inst)

            labels = list(instance_lists)
            if min(len(instance_lists[labels[0]]), len(instance_lists[labels[1]])) == len(instance_lists[labels[0]]):
                smaller_class = instance_lists[labels[0]]
                bigger_class = instance_lists[labels[1]]
            else:
                smaller_class = instance_lists[labels[1]]
                bigger_class = instance_lists[labels[0]]

            num_small_class_per_batch = max(1, math.ceil((len(smaller_class)
                                                          / (len(instance_list))) * self._batch_size))

            logger.info(f"Batching proportionally: "
                        f"{num_small_class_per_batch}:{self._batch_size-num_small_class_per_batch}")

            instance_list = []
            smaller_class_backup = list(smaller_class)

            cycle_smaller_times = self.num_cycles

            while len(bigger_class) != 0:
                if num_small_class_per_batch > len(smaller_class):
                    if shuffle:
                        random.shuffle(smaller_class_backup)
                    smaller_class += smaller_class_backup
                    cycle_smaller_times -= 1

                if cycle_smaller_times == 0:
                    break

                instance_list += smaller_class[:num_small_class_per_batch]
                smaller_class = smaller_class[num_small_class_per_batch:]

                instance_list += bigger_class[:(self._batch_size-num_small_class_per_batch)]
                bigger_class = bigger_class[(self._batch_size-num_small_class_per_batch):]

            iterator = iter(instance_list)
            excess: Deque[Instance] = deque()

            for batch_instances in lazy_groups_of(iterator, self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batch = Batch(possibly_smaller_batches)
                    yield batch

            if excess:
                yield Batch(excess)
