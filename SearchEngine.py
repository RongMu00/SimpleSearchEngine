import indexer2
import time
import json
import math
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.stem import PorterStemmer

#stopwords = nltk.download('stopwords')

def search(query : str):
    start_time = time.time()
    query_list = list(set(RegexpTokenizer("[a-zA-Z0-9]+").tokenize(query.lower())))
    #print(query_list)
    #query_list = list(set(indexer2.tokenizer(query)))
    common_words = set(stopwords.words('english'))
    #print(common_words)
    for q in query_list:
        if q in common_words:
            query_list.remove(q)

    ps = PorterStemmer()
    query_list = [ps.stem(term.lower()) for term in query_list]

    #print(query_list)
    query_dict = dict()
    for term in query_list:
        with open("D:\letter_partial_indexes1/" + term[0] + ".json", "r") as f:
            inverted_index = json.load(f)
            query_dict[term] = inverted_index[term]
        inverted_index.clear()
        f.close()


    result_doc = dict()

    # sorted postings by length
    posting_length = []
    for term, postings in query_dict.items():
        if postings is None:
            posting_length.append((term, 0))
        else:
            posting_length.append((term, len(postings)))
    sorted_query_term = [term for term, postings in sorted(posting_length, key=lambda x: x[1])]
    #sorted_query_term = [term for term, postings in sorted(query_dict.items(), key=lambda x: len(x[1]))]

    for i in range(len(sorted_query_term)):
        with open("D:\letter_partial_indexes1/" + sorted_query_term[i][0] + ".json", "r") as f:
            inverted_index = json.load(f)
            posting = inverted_index[sorted_query_term[i]]
            if posting is not None:
                for id, freq in posting.items():
                    if id not in posting:
                        result_doc[id] = [(sorted_query_term[i], freq)]
                    else:
                        try:
                            result_doc[id].append((sorted_query_term[i], freq))
                        except:
                            continue

    end_time = time.time()
    print("retrieve time: ", end_time - start_time)
    print(result_doc)
    return result_doc

result = search("Information retrieval is the science of searching for information in a document")

    # rank documents
    #rank(result_doc, )

def rank(result_doc, query_list):
    # calculate term idf score
        # merged.txt: idf = log(doc_num/(doc_freq+1))
        # idf_dict[term] = idf
    scores = dict()

    idf_dict = dict()
    for term in query_list:
        idf_dict[term] = indexer2.idf_dict[term]

    tf_idf_query = 0
    for term in query_list:
        tf = 1 + math.log(1, 10)
        idf = idf_dict[term]
        tf_idf_query += tf * idf

    # calculate query vector
        # for each term in query
            # calculate its tf-idf weights
            # sum up those weights


    # use term idf and query vector to calculate cosine similarity
        # for each doc
            # calculate tf-idf weights of each term in the doc
            # sums up
        # doc_vector * query_vector

    for doc_id in result_doc:
        tf_idf_doc = 0
        info = result_doc[doc_id]
        for term, freq in info:
            tf = 1 + math.log(freq, 10)
            idf = idf_dict[term]
            tf_idf_doc += tf*idf
        scores[doc_id] = tf_idf_doc * tf_idf_query
                
    # detect important content and add weight on it
        # html_dict[doc_id] = [important_content]
        # h1 + 4, h2 + 3, h3 + 2, b + 1, title + 5
        # divided by 15

    for doc_id in result_doc:
        if "title" in indexer2.important_content[doc_id]:
            scores[doc_id] += 5
        if "h1" in indexer2.important_content[doc_id]:
            scores[doc_id] += 4
        if "h2" in indexer2.important_content[doc_id]:
            scores[doc_id] += 3
        if "h3" in indexer2.important_content[doc_id]:
            scores[doc_id] += 2
        if "b" in indexer2.important_content[doc_id]:
            scores[doc_id] += 1
        if "strong" in indexer2.important_content[doc_id]:
            scores[doc_id] += 2

        scores[doc_id] = scores[doc_id] / 15

    # sorted doc based on scores
    ranking = [did for did, info in sorted(scores.items(), key=lambda x: -x[1])]

    return ranking












