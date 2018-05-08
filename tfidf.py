import numpy as np
import sys
import collections as coll
import math

def get_document_tf(filenames):
        documents = {}
        for filename in filenames:
                f = open(filename).read()
                documents[filename] = coll.Counter(word.lower()
                                                   for word in f.split())
        return documents

def get_idf(documents_tfs):
        total_num_docs = len(documents_tfs)

        # Calculate the number of documents that each term appears in
        num_docs_with_term = coll.Counter()
        for doc in documents_tfs.values():
                for term in doc.keys():
                        num_docs_with_term[term] += 1

        # Calculate the IDF per term:
        idf = {}
        for term in num_docs_with_term:
                idf[term] = math.log(float(total_num_docs) / num_docs_with_term[term])

        return idf


def build_document_vectors(documents_tfs):
        idf = get_idf(documents_tfs)
        weights = {}
        for term in idf.keys():
                weights[term] = {}
                for doc in documents_tfs.keys():
                        weights[term][doc] = idf[term] * documents_tfs[doc][term]
        return weights


def main():
        print(build_document_vectors(get_document_tf(sys.argv)))

if __name__ == '__main__':
        main()
