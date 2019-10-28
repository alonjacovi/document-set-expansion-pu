local embedding_size = 50;

{
  dataset_reader: {
    type: "pubmed-expansion"
  },
  vocabulary: {
    min_count: {tokens: 3}
  },
  train_data_path: "dse/pubmed_dse/L50_U1000/D000818.D001921.D051381/train.jsonl?label=label_true",
  validation_data_path: "dse/pubmed_dse/L50_U1000/D000818.D001921.D051381/valid.jsonl?label=label_true",
  test_data_path: "dse/pubmed_dse/L50_U1000/D000818.D001921.D051381/test.jsonl?label=label_true",
  evaluate_on_test: true,
  model: {
    type: "doc_classifier",
    pu_loss: false,
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
    type: "bucket",
    sorting_keys: [["abstract", "num_tokens"], ["title", "num_tokens"]],
    batch_size: 500,
  },
  trainer: {
    num_epochs: 100,
    patience: 20,
    cuda_device: -1,
    grad_clipping: 5.0,
    validation_metric: "+f1",
    optimizer: {
      type: "adam",
    },
    num_serialized_models_to_keep: 1,
  }
}