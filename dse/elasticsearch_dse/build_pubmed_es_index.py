def get_articles_data_from_pubmed_xml(path):
    data = []
    with gzip.open(path, 'rb') as f:
        xml_str = f.read()

        e = ET.ElementTree(ET.fromstring(xml_str))
        articles = e.findall('.//PubmedArticle')

        for article in articles:

            citation = article.find('MedlineCitation')
            pmid = citation.find('PMID')
            meshes = citation.find('MeshHeadingList')

            if meshes is None:
                continue

            mesh_set = []
            for mesh in meshes:
                desc = mesh.find('DescriptorName')
                mesh_set.append(desc.attrib['UI'])

            article = citation.find('Article')
            title = article.find('ArticleTitle')
            abstract = article.find('Abstract')

            if abstract is None:
                continue

            abstract = abstract.find('AbstractText')

            entry = {'pmid': pmid.text, 'title': title.text, 'abstract': abstract.text, 'mesh_set': mesh_set}
            data.append(entry)
    return data


def build_es_pubmed_index(es, data_files, index_name):
    print(f"Indexing PubMed with index name {index_name}")

    try:
        es.indices.delete(index=index_name)
        print("Deleted index")
    except elasticsearch.NotFoundError:
        print("Did not delete index")
        pass

    # Setup fresh index and mapping
    es.indices.create(index=index_name,
                      body={
                          "mappings": {
                              "paper": {
                                  "_source": {"enabled": True},
                                  "properties": {
                                      "pmid": {
                                          "type": "keyword"
                                      },
                                      "abstract": {
                                          "type": "text",
                                          "term_vector": "yes",
                                          "analyzer": "english"
                                      },
                                      "title": {
                                          "type": "text",
                                          "boost": 2.0,
                                          "analyzer": "english",
                                          "term_vector": "yes"
                                      },
                                      "mesh_set": {
                                          "type": "text",
                                          "term_vector": "yes",
                                          "analyzer": "whitespace"
                                      }
                                  }
                              }
                          }})

    def store_record(elastic_object, index_name, record):
        try:
            outcome = elastic_object.index(index=index_name, doc_type='paper', body=record)
        except Exception as ex:
            print('Error in indexing data')
            print(str(ex))
            exit()

        if outcome['_shards']['failed'] != 0:
            print(record, outcome, sep="\n")

    for path in data_files:
        print("Indexing...", path)

        data = get_articles_data_from_pubmed_xml(path)

        for entry in data:
            record = {
                'title': entry['title'],
                'abstract': entry['abstract'],
                'pmid': entry['pmid'],
                'mesh_set': ' '.join(entry['mesh_set'])
            }
            store_record(es, index_name, record)

    return es


if __name__ == "__main__":
    import gzip
    import xml.etree.ElementTree as ET
    import argparse
    from os import listdir
    from os.path import isfile, join

    import elasticsearch

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pubmed-path', action='store', help='Path to the pubmed .xml.gz data files')
    parser.add_argument('-n', '--index-name', action='store', default='pubmed_index',
                        help='Name of the index to create/overwrite')
    args = parser.parse_args()

    pubmed_xml_paths = [join(args.pubmed_path, f) for f in listdir(args.pubmed_path)
                        if isfile(join(args.pubmed_path, f)) and f.endswith('.xml.gz')]

    es = elasticsearch.Elasticsearch(timeout=60*5)

    build_es_pubmed_index(es, pubmed_xml_paths, index_name=args.index_name)
