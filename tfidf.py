import math
from collections import Counter, defaultdict

# document id -> category names
doc_categories = {}
# category name -> document ids
category_docs = {}
# category name -> category weight vectors
catweights = {}
catterms = set()


def load_categories(doc_categories_fname):
    doc_categories_f = open(doc_categories_fname)
    for line in doc_categories_f:
        line = line.split()
        doc_id = int(line[1])
        doc_category = line[0]
        if doc_id not in doc_categories.keys():
            doc_categories[doc_id] = set()
            doc_categories[doc_id].add(doc_category)
        else:
            doc_categories[doc_id].add(doc_category)
        if doc_category not in category_docs.keys():
            category_docs[doc_category] = set()
            category_docs[doc_category].add(doc_id)
        else:
            category_docs[doc_category].add(doc_id)
    doc_categories_f.close()


def generate_freq_vectors(doc_fnames):
    docs = {}
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
        return docs


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
    if len(num_docs_with_term.keys()) == 0:
        print("No docs with term")

    for term in num_docs_with_term.keys():
        idf[term] = math.log(float(total_num_docs)
                             / num_docs_with_term[term])

    return idf


def build_document_vectors(documents_tfs, testing):
    idf = get_idf(documents_tfs)
    print("Done calculating idfs")
    weights = {}
    numdocs = len(documents_tfs.keys())
    numdone = 0
    print("Generating tfidf weights")
    if len(documents_tfs.keys()) == 0:
        print("No documents processed")
    for doc in documents_tfs.keys():
        weights[doc] = defaultdict(float)
        if len(idf.keys()) == 0:
            print("No terms in idf vector")
        for term in idf.keys():
            if not testing or term in catterms:
                weight = idf[term] * documents_tfs[doc][term]
                if weight != 0.0:
                    weights[doc][term] = weight
        numdone += 1
        if numdone % (numdocs / 10) == 0:
            print("    {}{} done".format(numdone / numdocs * 100, '%'))
    return weights


def build_category_vectors(weights):
    cvs = {}
    numcategories = len(category_docs.keys())
    numdone = 0

    if len(category_docs.keys()) == 0:
        print("No categories found")
    for category in category_docs.keys():
        cvs[category] = defaultdict(float)
        if len(category_docs[category]) == 0:
            print("No documents in category")
        for doc_id in category_docs[category]:
            if doc_id not in weights.keys():
                continue
            if len(weights[doc_id].keys()) == 0:
                print("No terms in document")
            for term in weights[doc_id].keys():
                catterms.add(term)
                cvs[category][term] += weights[doc_id][term]
        numdone += 1
        if numdone % (numcategories / 10) == 0:
            print("    {}{} done".format(numdone / numcategories * 100, '%'))
    return cvs


def similarity(doc_weights, cat_weights):
    numerator = 0.0
    if len(doc_weights.keys()) == 0:
        return 0
    for key in doc_weights.keys():
        numerator += doc_weights[key] * cat_weights[key]
    denominator = math.sqrt(sum(dw**2 for dw in doc_weights.values()) +
                            sum(dw**2 for dw in cat_weights.values()))
    if denominator == 0:
        return 0
    return numerator / denominator


def train():
    print("===========================================================")
    print("Beginning training routine")
    doc_categories_fname = "rcv1/rcv1-v2.topics.qrels"
    load_categories(doc_categories_fname)

    print("Done reading in categories")
    print("Read in {} categories and {} document ids"
          .format(len(category_docs.keys()), len(doc_categories.keys())))

    doc_fnames = ["rcv1/lyrl2004_tokens_train.dat"]
    traindocs = generate_freq_vectors(doc_fnames)

#    subdocs = {k: v for k, v in list(traindocs.items())[:][:9000]}
#    traindocs = subdocs

    print("Done reading in docs & calculating tf vectors")
    print("Read in {} document tf vectors".format(len(traindocs)))

    docweights = build_document_vectors(traindocs, False)
    print("Done building document tfidf vectors")
    global catweights
    print("Building category vectors")
    catweights = build_category_vectors(docweights)
    print("Done building category vectors")
    print("Ending training routine")
    print("===========================================================")


def test():
    print("Beginning testing routine")
    correct = 0
    total = 0
    global catweights
    doc_fnames = ["rcv1/lyrl2004_tokens_test_pt0.dat"]
#    doc_fnames = ["rcv1/lyrl2004_tokens_train.dat"]
    testdocs = generate_freq_vectors(doc_fnames)
#    subdocs = {k: v for k, v in list(testdocs.items())[:][:5000]}
#    testdocs = subdocs
    print("Read in {} document tf vectors".format(len(testdocs)))

    docweights = build_document_vectors(testdocs, True)
    print("Done building document tfidf vectors")
    print("Running categorization test on test corpus")
    if len(docweights.keys()) == 0:
        print("No documents processed")
    for doc_id in docweights.keys():
        maxcategory = ""
        maxsimilarity = 0
        if len(catweights.keys()) == 0:
            print("No categories processed")
        for category in catweights.keys():
            sim = similarity(docweights[doc_id], catweights[category])
            #print("Doc ID: {}, Cat: {}, Sim: {}".format(doc_id, category, sim))
            if (sim > maxsimilarity):
                maxsimilarity = sim
                maxcategory = category

        if maxcategory in doc_categories[doc_id]:
            correct += 1
        total += 1

        if total % (len(testdocs) / 10) == 0:
            print("    {}{} done".format(total / len(testdocs) * 100, '%'))
    print("===========================================================")
    print("{} Correct: {}".format('%', correct / total * 100))

if __name__ == '__main__':
    train()
    test()
