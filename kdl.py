import itertools
import math
import sys
import collections as coll
import numpy as np

ids_to_rows = {}
terms_to_cols = {}

# Convert to: return 2D array rows = files, cols = terms
# returns a dictionary {filenames: {(all)terms: counts}}
def get_document_tf(filenames):
        documents = {}
        for i, filename in enumerate(filenames):
                f = open(filename).read()
                documents[filename] = coll.Counter(word.lower()
                                                   for word in f.split())
        return documents

def calculate_vocab():
        return range(len(terms_to_cols))


# P and Q must be iterators of the same length
# sum(P) == sum(Q) == 1
# no element in either P or Q is 0
def symetric_KDL(P, Q):
        summation = 0
        for p, q in itertools.zip_longest(P, Q):
                summation += (p-q)*math.log(float(p)/q)
        return summation

def calculate_conditional(documents):
        V = calculate_vocab()
        sum_per_term = coll.Counter()
        for term in V:
                for doc in documents:
                        sum_per_term[term] += doc[term]
                if sum_per_term[term] == 0:
                        print(sum_per_term[term])

        P = np.zeros((len(ids_to_rows), len(terms_to_cols)))
        epsilon = 1
        number_of_terms_not_in_doc = np.zeros((len(ids_to_rows)))

        for doc in range(len(documents)):
                number_of_terms_not_in_doc[doc] = len(V)
                for term, term_count in enumerate(documents[doc]):
                        if term_count == 0.0:
                                continue
                        P[doc][term] = term_count / sum_per_term[term]
                        epsilon = min(P[doc][term], epsilon)
                        number_of_terms_not_in_doc[doc] -= 1

        return epsilon/len(V), number_of_terms_not_in_doc, P

# Calculate Probability Distribution for given documents
def calculate_conditionals_back_off(documents, catagories):
        epsilon_d, num_not_in_d, P_d = calculate_conditional(documents)
        epsilon_c, num_not_in_c, P_c = calculate_conditional(catagories)

        epsilon = min(epsilon_c, epsilon_d)

        gamma = [1 - num_not_in_c[i] * epsilon for i,cat in enumerate(catagories)]
        beta  = [1 - num_not_in_d[i] * epsilon for i,cat in enumerate(catagories)]

        def probablity_term_condOn_doc(term, doc):
                if documents[doc][term] != 0:
                        return beta[doc] * P_d[doc][term]
                else:
                        return epsilon

        def probablity_term_condOn_cat(term, cat):
                #print(type(term))
                #print(type(cat))
                #print(categories)
                if catagories[cat][term] != 0:
                        return beta[cat] * P_c[cat][term]
                else:
                        return epsilon

        def prob_empty(term, cat):
                return epsilon

        return probablity_term_condOn_doc, probablity_term_condOn_cat, prob_empty

def KDL(cat, doc, p_term_c_doc, p_term_c_cat, vocab):
        return symetric_KDL([p_term_c_doc(v, doc) for v in vocab],
                            [p_term_c_cat(v, cat) for v in vocab])

def KDL_star(cat, doc, p_term_c_doc, p_term_c_cat, vocab, prob_empty):
        # Note that the division causes this function to be asymmetric
        denom = symetric_KDL([p_term_c_cat(v, cat) for v in vocab],
                             [prob_empty(v, doc) for v in vocab])
        return KDL(cat, doc, p_term_c_doc, p_term_c_cat, vocab) / denom

#takes dictionary {filenames: {(all)terms: counts}}
def build_document_vector(documents_tfs):
        for i, filename in enumerate(documents_tfs.keys()):
                ids_to_rows[filename] = i

        vocab = sum(documents_tfs.values(), coll.Counter())

        for j, word in enumerate(vocab):
                terms_to_cols[word] = j

        counts = np.zeros((len(ids_to_rows), len(terms_to_cols)))
        for d in documents_tfs:
                for t in documents_tfs[d]:
                        counts[ids_to_rows[d]][terms_to_cols[t]] = documents_tfs[d][t]
        return counts

def main():
        docs = build_document_vector(get_document_tf(sys.argv[1:]))
        # TODO: obviously this can't stay this way
        cats = build_document_vector(get_document_tf(sys.argv[1:]))
        
        # Here is a model for how to calculate the similarity:
        vocab = calculate_vocab()
        p_term_c_doc, p_term_c_cat, prob_empty = calculate_conditionals_back_off(docs, cats)

        # Need id's for categories -- currently the same
        for d, c in itertools.product(range(len(ids_to_rows)), range(len(ids_to_rows))):
                # for each document to categorize take the category that minimizes KDL_star
                print("Document: {}, Category: {}, Similarity: {}".format(d, c, KDL_star(c,d,p_term_c_doc,p_term_c_cat,vocab, prob_empty)))

if __name__ == '__main__':
        main()
