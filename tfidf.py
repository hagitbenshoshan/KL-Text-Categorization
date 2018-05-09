import math
from collections import Counter, defaultdict

# document id -> category names
doc_categories = {}
# category name -> document ids
category_docs = {}
# doc id -> tfs
docs = {}
# category name -> category weight vectors
categories = {}

def get_idf(documents_tfs):
    total_num_docs = len(documents_tfs)

    # Calculate the number of documents that each term appears in
    num_docs_with_term = Counter()
    for doc_id in documents_tfs.keys():
        doc = documents_tfs[doc_id]
        for term in doc.keys():
            num_docs_with_term[term] += 1

    # Calculate the IDF per term:
    idf = {}
    for term in num_docs_with_term.keys():
        idf[term] = math.log(float(total_num_docs)
                             / num_docs_with_term[term])

    return idf


def build_document_vectors(documents_tfs):
    idf = get_idf(documents_tfs)
    print("Done calculating idfs")
    weights = {}
    for doc in documents_tfs.keys():
        weights[doc] = defaultdict(float)
        for term in idf.keys():
            weight = idf[term] * documents_tfs[doc][term]
            if weight != 0.0:
                weights[doc][term] = weight
    return weights


def similarity(doc_weights, cat_weights):
    numerator = 0.0
    for key in doc_weights.keys():
        numerator += doc_weights[key] * cat_weights[key]
    denomonator = math.sqrt(sum(dw**2 for dw in doc_weights.values()) +
                            sum(dw**2 for dw in cat_weights.values()))
    return numerator/denomonator


def main():
    doc_categories_fname = "rcv1/rcv1-v2.topics.qrels"
    doc_categories_f = open(doc_categories_fname)
    for line in doc_categories_f:
        line = line.split()
        doc_id = int(line[1])
        doc_category = line[0]
        if doc_id not in doc_categories.keys():
            doc_categories[doc_id] = [doc_category]
        else:
            doc_categories[doc_id].append(doc_category)
        if doc_category not in category_docs.keys():
            category_docs[doc_category] = [doc_id]
        else:
            category_docs[doc_category].append(doc_id)
    doc_categories_f.close()

    print("Done reading in categories")
    print("Read in {} categories and {} document ids"
          .format(len(category_docs.keys()), len(doc_categories.keys())))

    doc_fnames = ["rcv1/lyrl2004_tokens_train.dat"]
    for doc_fname in doc_fnames:
        doc_f = open(doc_fname)
        doc_id = -1
        doc_wordlist = Counter()
        for line in doc_f:
            if len(line) > 2 and line[:2] == '.I':
                doc_id = int(line[3:])
            elif len(line) > 2 and line[:2] == '.W':
                continue
            elif len(line) > 1:
                words = line.split()
                for word in words:
                    doc_wordlist[word] += 1
            else:
                docs[doc_id] = doc_wordlist
                doc_id = -1
                doc_wordlist = Counter()

    subdocs = {k: v for k, v in list(docs.items())[:100]}

    print("Done reading in docs & calculating tf vectors")
    print("Read in {} document tf vectors".format(len(subdocs)))

    weights = build_document_vectors(subdocs)
    print("Done building weights")



if __name__ == '__main__':
    main()
