import itertools
import math
from collections import Counter
import numpy as np

doc_ids_to_rows = {}
rows_to_doc_ids = {}
test_doc_ids_to_rows = {}
test_rows_to_doc_ids = {}
cats_to_rows = {}
rows_to_cats = {}
terms_to_cols = {}

# document id -> category names
doc_categories = {}
# category name -> document ids
category_docs = {}

infty = 100000000

class KDLData:
    def __init__(self, vocab, p_term_c_doc, p_term_c_cat, prob_empty):
        self.vocab = vocab
        self.p_term_c_doc = p_term_c_doc
        self.p_term_c_cat = p_term_c_cat
        self.prob_empty = prob_empty


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

    for i, cat in enumerate(category_docs.keys()):
        cats_to_rows[cat] = i
        rows_to_cats[i] = cat


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


# Convert to: return 2D array rows = files, cols = terms
# returns a dictionary {filenames: {(all)terms: counts}}
def get_document_tf(filenames):
    documents = {}
    for i, filename in enumerate(filenames):
        f = open(filename).read()
        documents[filename] = Counter(word.lower() for word in f.split())
    return documents


def calculate_vocab():
    return range(len(terms_to_cols))


# P and Q must be iterators of the same length
# sum(P) == sum(Q) == 1
# no element in either P or Q is 0
def symetric_KDL(P, Q):
    summation = 0
    for p, q in itertools.zip_longest(P, Q):
        try:
            summation += (p - q) * math.log(float(p) / q)
        except Exception as e:
            print("BLAH")
    return summation


def calculate_conditional(items, num_items):
    V = calculate_vocab()
    sum_per_term = Counter()
    for term in V:
        for item in items:
            sum_per_term[term] += item[term]
        if sum_per_term[term] == 0:
            print(sum_per_term[term])

    P = np.zeros((num_items, len(terms_to_cols)))
    epsilon = 1
    num_terms_not_in_item = np.zeros(num_items)

    for item in range(len(items)):
        num_terms_not_in_item[item] = len(V)
        for term, term_count in enumerate(items[item]):
            if term_count == 0.0:
                continue
            P[item][term] = term_count / sum_per_term[term]
            epsilon = min(P[item][term], epsilon)
            num_terms_not_in_item[item] -= 1

        return epsilon/len(V), num_terms_not_in_item, P


# Calculate Probability Distribution for given documents
def calculate_conditionals_back_off(documents, categories):
    epsilon_d, num_not_in_d, P_d = calculate_conditional(documents, len(doc_ids_to_rows))
    epsilon_c, num_not_in_c, P_c = calculate_conditional(categories, len(cats_to_rows))

    epsilon = min(epsilon_c, epsilon_d)

    beta = [1 - num_not_in_d[i] * epsilon for i, doc in enumerate(documents)]
    gamma = [1 - num_not_in_c[i] * epsilon for i, cat in enumerate(categories)]

    def probablity_term_condOn_doc(term, doc, doctfs, vocab):
        tf_vector = doctfs[doc]
        beta_ = 1.0
        tf_vector /= sum(tf_vector)

        t_min = 1000
        for t in tf_vector:
            if t != 0.0:
                t_min = t
        epsilon_ = min(epsilon_c, t_min)

        cnt = 0
        for v in tf_vector:
            #print(v)
            if v == 0:
                cnt+=1
        beta_ = 1 - cnt * epsilon_
        #print("Beta: {}, epsilon: {}".format(beta_, epsilon_))

        if tf_vector[term] != 0:
            return beta_ * tf_vector[term]
        else:
            return epsilon_

    def probablity_term_condOn_cat(term, cat):
        #print(type(term))
        #print(type(cat))
        #print(categories)
        if P_c[cat][term] != 0:
            return beta[cat] * P_c[cat][term]
        else:
            return epsilon

    def prob_empty():
        return epsilon

    return probablity_term_condOn_doc, probablity_term_condOn_cat, prob_empty


def KDL(cat, doc, p_term_c_doc, p_term_c_cat, vocab, doctfs):
    return symetric_KDL([p_term_c_doc(v, doc, doctfs, vocab) for v in vocab],
                        [p_term_c_cat(v, cat) for v in vocab])


def KDL_star(cat, doc, p_term_c_doc, p_term_c_cat, vocab, prob_empty, doctfs):
    # Note that the division causes this function to be asymmetric
    denom = symetric_KDL([p_term_c_cat(v, cat) for v in vocab],
                         [prob_empty() for v in vocab])
    if denom == 0:
        return infty
    return KDL(cat, doc, p_term_c_doc, p_term_c_cat, vocab, doctfs) / denom


#takes dictionary {filenames: {(all)terms: counts}}
def build_document_vectors(document_tfs, testing):
    dids2rows = doc_ids_to_rows
    rows2dids = rows_to_doc_ids
    if testing:
        dids2rows = test_doc_ids_to_rows
        rows2dids = test_rows_to_doc_ids

    for i, doc_id in enumerate(document_tfs.keys()):
        dids2rows[doc_id] = i
        rows2dids[i] = doc_id

    vocab = sum(document_tfs.values(), Counter())

    for j, word in enumerate(vocab):
        terms_to_cols[word] = j

    counts = np.zeros((len(dids2rows), len(terms_to_cols)))
    for d in document_tfs:
        for t in document_tfs[d]:
            counts[dids2rows[d]][terms_to_cols[t]] = document_tfs[d][t]
    return counts


def build_category_vectors(document_tfs):
    counts = np.zeros((len(cats_to_rows), len(terms_to_cols)))

    for cat in category_docs.keys():
        for doc_id in category_docs[cat]:
            if doc_id not in doc_ids_to_rows.keys():
                continue
            for i, termval in enumerate(document_tfs[doc_ids_to_rows[doc_id]]):
                counts[cats_to_rows[cat]][i] += termval
    return counts


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

    subdocs = {k: v for k, v in list(traindocs.items())[:][:5000]}
    traindocs = subdocs

    print("Done reading in docs & calculating tf vectors")
    print("Read in {} document tf vectors".format(len(traindocs)))

    doctfs = build_document_vectors(traindocs, False)
    print("Done building document tf matrix")
    print("Building category tf matrix")
    cattfs = build_category_vectors(doctfs)
    print("Done building category tf matrix")

    print("Calcualting vocabualry")
    vocab = calculate_vocab()
    print("Find probability distribution functions")
    p_term_c_doc, p_term_c_cat, prob_empty = calculate_conditionals_back_off(doctfs, cattfs)

    print("Ending training routine")
    print("===========================================================")
    return KDLData(vocab, p_term_c_doc, p_term_c_cat, prob_empty)


def test(kdldata):
    vocab = kdldata.vocab
    p_term_c_doc = kdldata.p_term_c_doc
    p_term_c_cat = kdldata.p_term_c_cat
    prob_empty = kdldata.prob_empty
    print("Beginning testing routine")
    correct = 0
    total = 0
    global catweights
    doc_fnames = ["rcv1/rcv1/lyrl2004_tokens_train.dat"]
#    doc_fnames = ["rcv1/lyrl2004_tokens_train.dat"]
    testdocs = generate_freq_vectors(doc_fnames)
    subdocs = {k: v for k, v in list(testdocs.items())[:][5000:5100]}
    testdocs = subdocs
    print("Read in {} document tf vectors".format(len(testdocs)))

    doctfs = build_document_vectors(testdocs, True)
    print("Done building document tfidf vectors")
    print("Running categorization test on test corpus")

    # Need id's for categories -- currently the same
    for doc_idx in range(len(test_doc_ids_to_rows)):
        minsim = infty
        mincat = 0
        for cat_idx in range(len(cats_to_rows)):
            # for each document to categorize take the category that minimizes KDL_star
            sim = KDL_star(cat_idx, doc_idx, p_term_c_doc, p_term_c_cat, vocab, prob_empty, doctfs)
            if (sim < minsim):
                minsim = sim
                mincat = cat_idx
        for cat in doc_categories[test_rows_to_doc_ids[doc_idx]]:
            print(rows_to_cats[mincat])
            if cat == rows_to_cats[mincat]:
                correct += 1
                break
        total += 1
        print(correct/total)
        print(total/len(test_doc_ids_to_rows))

    print("% Correct: {}".format(correct / total * 100))


if __name__ == '__main__':
        kdldata = train()
        test(kdldata)
