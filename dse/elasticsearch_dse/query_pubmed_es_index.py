def get_pmid_query(pmid):
    """
    Search the es pubmed index by pmid
    """
    return {'query': {'match': {'pmid': pmid}}}


def get_conjunction_match_query(conjunction):
    """
    Search the es pubmed index for all documents that admit the MeSH conjunction
    """
    return {
        'query': {
                'match': {
                    'mesh_set': {
                        'query': ' '.join(conjunction),
                        'analyzer': 'whitespace',
                        'operator': 'and'
                    }
                }
        }
    }


def get_mlt_query(like_these, minimum_should_match):
    """
    Perform a More Like This elasticsearch query on a given collection of documents
    """
    return {
        "query": {
            "more_like_this": {
                "fields": [
                    "title", "abstract"
                ],
                "like": like_these,
                # "min_term_freq": 0.,
                # "min_doc_freq": 0,
                "minimum_should_match": minimum_should_match
            }
        }
    }


def scroll_search(es, index_name, query, multiget_api=False, scroll_lifetime='10m', scroll_size=10000):
    res = es.search(index=index_name, body=query, scroll=scroll_lifetime, size=scroll_size)

    sid = res['_scroll_id']
    scroll_size = res['hits']['total']

    mg = []
    retrieved = []

    while scroll_size > 0:
        for hit in res['hits']['hits']:
            if hit['_source']['abstract'] is None or hit['_source']['title'] is None \
               or hit['_source']['mesh_set'] is None:
                continue

            if multiget_api:
                multiget_format = {'_index': hit['_index'], '_type': hit['_type'], '_id': hit['_id']}
                mg.append(multiget_format)
            else:
                hit['_source']['score'] = hit['_score']
                hit['_source']['rank'] = len(retrieved)

            retrieved.append(hit['_source'])

        res = es.scroll(scroll_id=sid, scroll='10m')
        scroll_size = len(res['hits']['hits'])

    if multiget_api:
        return retrieved, mg
    else:
        return retrieved


def shuffle_combined(a, b):
    combined = list(zip(a, b))
    shuffle(combined)
    a[:], b[:] = zip(*combined)


def query_pubmed_index(es, index_name, conjunction, LP_size, U_size, minimum_should_match):
    query = get_conjunction_match_query(conjunction)
    P, multiget_api_P = scroll_search(es, index_name, query, multiget_api=True)

    shuffle_combined(P, multiget_api_P)

    LP = P[:LP_size]
    multiget_api_LP = multiget_api_P[:LP_size]

    print("Size of P:", len(P))
    print("Size of LP:", len(LP))

    query = get_mlt_query(multiget_api_LP, minimum_should_match)
    U = scroll_search(es, index_name, query)

    # assert len(U) == U_size, len(U)

    print("Size of U:", len(U))

    return LP, U, len(P)


def combine_datasets(datasets, tags):
    data = []

    for i in range(len(datasets)):
        for entry in datasets[i]:
            entry_copy = entry.copy()
            entry_copy["label"] = tags[i]
            data.append(entry_copy)

    shuffle(data)
    return data


def dump_jsonl(jsonl, path):
    with open(path, "w") as f:
        for instance in jsonl:
            json.dump(instance, f)
            f.write("\n")


def build_datasets(LP, U, conjunction, true_P_size):
    shuffle(LP)
    shuffle(U)

    train_size = int(len(LP)*0.5)
    valid_test_size = int(len(LP)*0.25)

    for sample in LP:
        sample["label_true"] = 'positive/labeled'
        sample[f'label_L{len(LP)}'] = 'positive/labeled'

    train_LP = LP[:train_size]
    valid_LP = LP[train_size:train_size+valid_test_size]
    test_LP = LP[train_size+valid_test_size:]

    print(len(valid_LP), len(test_LP))

    for sample in U:
        sample[f'label_L{len(LP)}'] = 'negative/unlabeled'

    UP = []
    N = []
    conjunction = set(conjunction)

    for sample in U:
        if conjunction.issubset(set(sample["mesh_set"].split(' '))):
            sample["label_true"] = 'positive/labeled'
            UP.append(sample)
        else:
            sample['label_true'] = 'negative/unlabeled'
            N.append(sample)

    print(f"Number of positives in U: {len(UP)}")
    print("Size of P in dataset:", len(LP) + len(UP))
    print(f"Recall: {len(UP)/(true_P_size-len(LP))}")
    print(f"Precision: {len(UP)/len(U)}")

    train_UP_size = int(len(UP)*0.5)
    valid_test_UP_size = int(len(UP)*0.25)
    train_N_size = int(len(N)*0.5)
    valid_test_N_size = int(len(N)*0.25)

    train_UP = UP[:train_UP_size]
    valid_UP = UP[train_UP_size:train_UP_size+valid_test_UP_size]
    test_UP = UP[train_UP_size+valid_test_UP_size:]

    train_N = N[:train_N_size]
    valid_N = N[train_N_size:train_N_size+valid_test_N_size]
    test_N = N[train_N_size+valid_test_N_size:]

    md = {}

    train = train_LP + train_UP + train_N
    valid = valid_LP + valid_UP + valid_N
    test = test_LP + test_UP + test_N

    shuffle(train)
    shuffle(valid)
    shuffle(test)

    md.update({'P_size': true_P_size, 'LP_size': len(LP), 'U_size': len(U), 'train_LP_size': len(train_LP),
               'train_U_size': len(train_UP) + len(train_N),
               'valid_LP_size': len(valid_LP), 'valid_U_size': len(valid_N) + len(valid_UP),
               'test_LP_size': len(test_LP),
               'test_U_size': len(test_UP) + len(test_N), 'train_size': len(train), 'test_size': len(test),
               'valid_size': len(valid), "precision": len(UP)/len(U), "recall": len(UP)/(true_P_size-len(LP))})

    return train, valid, test, md


if __name__ == "__main__":
    import json
    import os
    from random import shuffle
    import argparse

    import elasticsearch

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-path', action='store', help='Path to store the generated PubMed-DSE tasks')
    parser.add_argument('-n', '--index-name', action='store', default='pubmed_index',
                        help='Name of the index to create/overwrite')
    parser.add_argument('--task-sizes-lp', nargs='+', type=int, default=[20, 50, 100],
                        help='A list of all the |LP| (amount of labeled positives) values to use for task generation. '
                             'Each |LP| value generates a task where that amount of positive docs is chosen.')
    parser.add_argument('--task-sizes-u', nargs='+', type=int, default=[100000],
                        help='A list of all the |U| (amount of unlabeled docs) values to use for task generation. '
                             'Each |U| value generates a task where that amount of unlabeled docs is chosen.')
    parser.add_argument('--mesh-map-path', action='store', default='dse/elasticsearch_dse/mesh_id_to_name.json',
                        help='Path to the mapping between MeSH IDs to their names.'
                             'The mapper given in this repository was created on January 2019. If your version of'
                             ' PubMed contains MeSH terms not in this file, you should add them, or create your own.')
    parser.add_argument('--task-topics', nargs='+', type=str,
                        help='A list MeSH conjunctions to create PubMed-DSE tasks from. Each one is period-separated.'
                             'eg., D000328.D008875.D015658')
    parser.add_argument('--minimum-should-match', action='store', default='15%',
                        help='Elasticsearch parameter "Minimum Should Match" for More Like This queries. '
                             'This parameter decides the minimum amount of words that should match between the LP'
                             ' documents and each U document. This parameter essentially filters out documents to'
                             ' help queries run much faster. '
                             'See: https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-mlt-query.html')
    args = parser.parse_args()

    candidates = [sorted(conj.strip().split(".")) for conj in args.task_topics]
    out_path = args.output_path

    es = elasticsearch.Elasticsearch(timeout=60*5)

    with open(args.mesh_map_path, "r") as f:
        mesh_id_to_name = json.load(f)

    for conjunction in candidates:
        for LP_size in args.task_sizes_lp:
            for U_size in args.task_sizes_u:
                print("Getting PN of:", conjunction, [mesh_id_to_name[mesh_term] for mesh_term in conjunction])

                LP, U, true_P_size = query_pubmed_index(es, args.index_name, conjunction,
                                                        LP_size*2, U_size*2, # *2 for validation and test (half each)
                                                        minimum_should_match=args.minimum_should_match)

                topic_out_path = out_path + "/" + f"/L{LP_size}_U{U_size}/" + ".".join(conjunction)

                if not os.path.exists(topic_out_path):
                    os.makedirs(topic_out_path)

                train_path = topic_out_path + "/train.jsonl"
                test_path = topic_out_path + "/test.jsonl"
                valid_path = topic_out_path + "/valid.jsonl"
                metadata_path = topic_out_path + "/metadata.json"

                train, valid, test, md = build_datasets(LP, U, conjunction, true_P_size)

                with open(metadata_path, "w") as f:
                    md['mesh_conjunction'] = conjunction
                    md['mesh_conjunction_str'] = [mesh_id_to_name[mesh_term] for mesh_term in conjunction]
                    print('Task Metadata:', md)
                    json.dump(md, f)

                dump_jsonl(train, train_path)
                dump_jsonl(valid, valid_path)
                dump_jsonl(test, test_path)
