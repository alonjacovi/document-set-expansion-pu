import logging
from typing import Dict, Optional, List

import numpy
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import Metric, CategoricalAccuracy, F1Measure

from dse.models.losses.nnpu import NonNegativePULoss

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("doc_classifier")
class AcademicDocumentClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 title_encoder: Seq2VecEncoder,
                 abstract_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 pu_loss: bool = True,
                 # loss: loss = None,
                 prior: float = None,
                 should_log_activations: bool = False,
                 metrics: Dict[str, Metric] = None,
                 pu_beta: float = 0,
                 pu_gamma: float = 1,
                 regularizer: Optional[RegularizerApplicator] = None
                 ) -> None:
        super(AcademicDocumentClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.classifier_feedforward = classifier_feedforward
        self.pu_loss = pu_loss
        self.prior = prior

        label_vocab = self.vocab.get_token_to_index_vocabulary("labels")
        assert (label_vocab["negative/unlabeled"] == 1 and label_vocab["positive/labeled"] == 0) or \
               (label_vocab["negative/unlabeled"] == 0 and label_vocab["positive/labeled"] == 1)
        self.positive_class = label_vocab["positive/labeled"]
        # self.loss = loss

        if not self.pu_loss:
            if self.prior is not None:
                if self.positive_class == 0:
                    self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1-self.prior, self.prior]))
                else:
                    self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([self.prior, 1-self.prior]))
            else:
                self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.prior = prior
            self.normalize = (lambda x: torch.sigmoid(x))
            self.sample_loss = (lambda x: torch.sigmoid(-x))
            self.loss = NonNegativePULoss(self.prior, positive_class=self.positive_class,
                                              loss=self.sample_loss, nnpu=True, beta=pu_beta, gamma=pu_gamma)

        self.metrics = metrics or {
                "accuracy": CategoricalAccuracy(),
                "f1": F1Measure(self.positive_class)
                # "auc": Auc(self.positive_class)
        }

        for module in self.modules():
            module.should_log_activations = should_log_activations

        initializer(self)

    def forward(self,
                title: Dict[str, torch.LongTensor],
                abstract: Dict[str, torch.LongTensor],
                md: MetadataField,
                label: torch.LongTensor = None,
                label_true: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:

        embedded_abstract = self.text_field_embedder(abstract)
        embedded_title = self.text_field_embedder(title)

        title_mask = util.get_text_field_mask(title)
        abstract_mask = util.get_text_field_mask(abstract)

        encoded_title = self.title_encoder(embedded_title, title_mask)
        encoded_abstract = self.abstract_encoder(embedded_abstract, abstract_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_title, encoded_abstract], dim=-1))

        if not self.pu_loss:
            if self.positive_class == 1:
                logits = torch.cat(((-logits).view(-1, 1), logits.view(-1, 1)), dim=1)
            else:
                logits = torch.cat((logits.view(-1, 1), (-logits).view(-1, 1)), dim=1)
            class_probabilities = F.softmax(logits, dim=1)
        else:
            positive_pred = self.normalize(logits)
            negative_pred = 1 - positive_pred

            if self.positive_class == 1:
                class_probabilities = torch.cat((negative_pred.view(-1, 1),
                                                 positive_pred.view(-1, 1)), dim=1)
            else:
                class_probabilities = torch.cat((positive_pred.view(-1, 1),
                                                 negative_pred.view(-1, 1)), dim=1)

        output_dict = {"class_probabilities": class_probabilities}

        if label is not None:
            loss = self.loss(logits, label)

            for metric in self.metrics.values():
                metric(class_probabilities, label.long())
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_results = {}

        if "accuracy" in self.metrics:
            metric_results["accuacy"] = self.metrics["accuracy"].get_metric(reset=reset)
        if "f1" in self.metrics:
            f1_measure_metric = self.metrics["f1"].get_metric(reset=reset)
            metric_results["p"] = f1_measure_metric[0]
            metric_results["r"] = f1_measure_metric[1]
            metric_results["f1"] = f1_measure_metric[2]
        if "auc" in self.metrics:
            metric_results["auc"] = self.metrics["auc"].get_metric(reset=reset)

        return metric_results

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict
