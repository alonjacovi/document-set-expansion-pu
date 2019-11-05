from typing import Dict
import json
import logging
import numpy as np

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("pubmed-expansion")
class PubmedExpansionDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        path_and_args = file_path.split('?')
        file_path = path_and_args[0]
        args = {}
        if len(path_and_args) == 2:
            args_str = path_and_args[1].split('&')
            for arg in args_str:
                arg = arg.split('=')
                args[arg[0]] = arg[1]

        logger.info(f'Reading with arguments: {args}')
        if 'size' in args:
            limit = int(args['size'])
            if limit > 0:
                logger.info(f'Limiting dataset size to (the first) {limit} samples')
        else:
            limit = -1

        assert 'evaluation' not in args or (args['evaluation'] in ['true', 'false'])

        count = 0
        count_pos = 0
        count_neg = 0
        dropped_samples_lp = 0
        dropped_samples_u = 0

        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                pubmed_json = json.loads(line)

                assert args['label'] in pubmed_json, f"{args['label']} not found in {pubmed_json}"
                label = pubmed_json[args['label']]

                label_true = 0 if pubmed_json['label_true'] == 'negative/unlabeled' else 1

                pmid = pubmed_json['pmid']
                title = pubmed_json['title']
                abstract = pubmed_json['abstract']

                if 'limit' in args and label == 'positive/labeled' and count_pos >= int(args['limit']):
                    continue

                instance = self.text_to_instance(title, abstract, pmid, label, label_true=label_true,
                                                 evaluation='evaluation' in args and args['evaluation'] == 'true')

                if 'max_length' in args:
                    if instance['abstract'].sequence_length() <= int(args['max_length']):
                        yield instance
                    else:
                        if label == "positive/labeled":
                            dropped_samples_lp += 1
                        else:
                            dropped_samples_u += 1
                else:
                    yield instance

                if label == 'positive/labeled':
                    count_pos += 1
                elif label == 'negative/unlabeled':
                    count_neg += 1
                count += 1
                if count == limit:
                    break

            logger.info(f"Loaded {count} samples. {count_pos} positive/labeled and {count_neg} negative/unlabeled")
            if dropped_samples_lp != 0 or dropped_samples_u != 0:
                logger.info(f"Skipped {dropped_samples_lp} LP samples and {dropped_samples_u} U samples due to length.")

    @overrides
    def text_to_instance(self, title: str, abstract: str, pmid: str,
                         label: str = None,
                         label_true: float = None,
                         evaluation: bool = False) -> Instance:  # type: ignore
        tokenized_title = self._tokenizer.tokenize(title)
        tokenized_abstract = self._tokenizer.tokenize(abstract)
        title_field = TextField(tokenized_title, self._token_indexers)
        abstract_field = TextField(tokenized_abstract, self._token_indexers)

        md = MetadataField({"pmid": pmid, "evaluation": evaluation})
        fields = {'title': title_field, 'abstract': abstract_field, 'md': md}
        if label is not None:
            fields['label'] = LabelField(label)

        if label_true is not None:
            fields['label_true'] = ArrayField(np.array(label_true, dtype="float32"))

        return Instance(fields)
