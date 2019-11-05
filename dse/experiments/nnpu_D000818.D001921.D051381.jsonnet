local embedding_size = 50;

{
  dataset_reader: {
    type: "pubmed-expansion"
  },
  vocabulary: {
    min_count: {tokens: 3}
  },
  train_data_path: "http://nlp.biu.ac.il/~jacovia/pubmed-dse/L50/D000818.D001921.D051381/train.jsonl?label=label_L100",
  validation_data_path: "http://nlp.biu.ac.il/~jacovia/pubmed-dse/L50/D000818.D001921.D051381/valid.jsonl?label=label_L100",
  test_data_path: "http://nlp.biu.ac.il/~jacovia/pubmed-dse/L50/D000818.D001921.D051381/test.jsonl?label=label_true&evaluation=true",
  evaluate_on_test: true,
  model: {
    type: "doc_classifier",
    pu_loss: true,
    prior: 0.5,
    pu_beta: 0.0,
    pu_gamma: 1.0,
    text_field_embedder: {
      tokens: {
        type: "embedding",
        embedding_dim: embedding_size,
        trainable: true
      }
    },
    title_encoder: {
      type: "cnn",
      embedding_dim: embedding_size,
      num_filters: 50,
      ngram_filter_sizes: [3, 5]
    },
    abstract_encoder: {
      type: "cnn",
      embedding_dim: embedding_size,
      num_filters: 100,
      ngram_filter_sizes: [3, 5]
    },
    classifier_feedforward: {
      input_dim: 300,
      num_layers: 1,
      hidden_dims: [1],
      activations: ["linear"]
    }
  },
  iterator: {
    type: "proportional",
    batch_size: 1000,
    num_cycles: 1,
  },
  trainer: {
    num_epochs: 100,
    patience: 20,
    cuda_device: 0,
    grad_clipping: 5.0,
    validation_metric: "-loss",
    optimizer: {
      type: "adam",
    },
    num_serialized_models_to_keep: 1,
  }
}
