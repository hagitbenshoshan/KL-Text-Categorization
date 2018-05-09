import itertools
import math
import sys
import collections as coll

# returns a dictionary {filenames: {(all)terms: counts}}
def get_document_tf(filenames):
        documents = {}
        for filename in filenames:
                f = open(filename).read()
                documents[filename] = coll.Counter(word.lower()
                                                   for word in f.split())
        return documents

# P and Q must be iterators of the same length
# sum(P) == sum(Q) == 1
# no element in either P or Q is 0
def symetric_KDL(P, Q):
        summation = 0
        for p, q in itertools.zip_longest(P, Q):
                summation += (p-q)*math.log(float(p)/q)
        return summation

# Calculate Probability Distribution for given documents
def calculate_joint_pmf(documents):
        V = sum(documents.values(), coll.Counter())
        sum_per_term = coll.defaultdict(int)
        for term in V:
                for doc in documents:
                        sum_per_term[term] += documents[doc][term]
        P = {}
        epsilon = 1
        number_of_terms_not_in_doc = {}
        for doc in documents:
                P[doc] = {}
                number_of_terms_not_in_doc[doc] = len(V)
                for term in documents[doc]:
                        P[doc][term] = documents[doc][term] / sum_per_term[term]
                        # Epsilon needs to be minimal over both docs and categories 
                        # (for now just minimal over one)
                        epsilon = min(P[doc][term], epsilon)
                        number_of_terms_not_in_doc[doc] -= 1

        beta = {doc : 1 - number_of_terms_not_in_doc[doc] for doc in documents}

        def probablity_term_condOn_doc(term, doc):
                if documents[doc][term] != 0:
                        return beta[doc] * P[doc][term]
                else:
                        return epsilon

        return probablity_term_condOn_doc


def main():
        print(calculate_joint_pmf(get_document_tf(sys.argv)))

if __name__ == '__main__':
        main()