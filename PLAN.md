TfIdf Implementation
====================

Documents Represented
---------------------
d_j = (w_{1j}, ... w_{|V|j)

Terms in Document V
-------------------
V is the set of terms in at least one of documents - in our case words

Term Frequency tf
-----------------
tf(t_k, d_j) = number of times term k occurs in document j

Inverse Document Frequency
--------------------------
idf(t_k) = \log(total # of documents/# of documents which contain t_k)

Weights for a Document
----------------------
w_{kj} = tf(t_k, d_j) * idf(t_k)

Classification -- TODO figure out how to make c_i??
--------------
Use Cosine similarity between category vector and the document vector

Category Vector
---------------
c = {w_1, w_2, ... w_l}
The category vector is: w_{ki} = tf(t_k, c_i) x idf(c_i)

tf(t_k, c_i)
------------
tf(t_k, c_i) = 



KDL Classifier
==============
