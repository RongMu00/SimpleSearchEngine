import re
import json
from bs4 import BeautifulSoup
#import lxml
import os
import math
from collections import defaultdict, OrderedDict
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import PorterStemmer
from simhash import Simhash, SimhashIndex

letters = 'abcdefghijklmnopqrstuvwxyz0'
hash_values = dict()   # len(hash_values.keys()) ->doc_cnt
hashes = SimhashIndex([], k=1)
words_to_weight = []
inverted_index = dict()
doc_cnt = 0
term_cnt = 0
idf_dict = dict()
tf_idf_dict = dict()
doc_dict = dict()
# indexes = []
# for i in range(26):
#     indexes.append(defaultdict(dict))
important_content = defaultdict(list)
bookkeeping = dict()

def store_document(initialPath):
    documents = dict()
    doc_id = 0
    for (root, dirs, files) in os.walk(initialPath, topdown=True):   # initialPath == "D:\DEV"
        for file in files:
            documents[doc_id] = os.path.join(root, file)
            doc_id += 1
    return documents

def tokenizer(text):
    words = RegexpTokenizer("[a-zA-Z0-9]+").tokenize(text.lower())
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in words]

    return stemmed_words

def get_tf_dict(tokens : list):
    if len(tokens) != 0:
        term_freq = dict()
        for token in tokens:
            if token in term_freq:
                term_freq[token] += 1
            else:
                term_freq[token] = 1
        return term_freq
#
def get_idf_score(directory, doc_num):
    global idf_dict
    for file in os.listdir(directory):
        p = directory + '/' + file
        index = json.loads(str(p.read()))
        for k, v in index.items():
            df = len(index[k])
            idf = math.log(doc_num/(df+1), 10)
            idf_dict[k] = idf



def get_tfidf_score(directory, doc_num, doc_id):
    global tf_idf_dict
    for file in os.listdir(directory):
        p = directory + '/' + file
        index = json.loads(str(p.read()))
        for k, v in index.items():
            df = len(index[k])
            idf = math.log(doc_num/(df+1), 10)
            tf = inverted_index[k][doc_id]
            tf_idf = tf * idf
            tf_idf_dict[k] = tf_idf


def get_posting(word, doc_id, tokens):
    posting = dict()
    posting[doc_id] = get_tf_dict(tokens)[word]
    return posting

def create_letter_partial_index():
    path = 'D:\letter_partial_indexes1'
    if not os.path.exists(path):
        os.makedirs(path)
    letter_partial_list = []
    for l in letters:
        p = open(path + "/"+str(l)+".json", 'w')
        letter_partial_list.append(p)
        json.dump(dict(), p, indent=3)
        p.close()
    return letter_partial_list


def buildIndex(path):
    print("buildIndex")
    global inverted_index
    global doc_cnt
    partition = 4
    partial_cnt = 1
    doc_id = 0
    doc_dict = store_document(path)
    print(len(doc_dict))
    #create_letter_partial_index()
    #while doc_cnt < len(doc_dict):
    for doc in doc_dict.values():
        print(doc)
        if re.match(r'.*\.json', doc):
            doc_cnt += 1
            # parse document
            try:
                with open(doc, "r") as d:
                    file = json.load(d)
                    soup = BeautifulSoup(file["content"], "lxml")
                    text = soup.get_text()
                    # detect documents nearest duplicate
                    global hash_values
                    hash_val = Simhash(text).value
                    simhashed_word = Simhash(text)
                    if len(hashes.get_near_dups(simhashed_word)) <= 0:
                    #if hash_val not in hash_values:
                        hash_values[doc_id] = hash_val
                        doc_cnt += 1
                        stemmed_words = tokenizer(text)
                        # build inverted index
                        global term_cnt
                        for word in stemmed_words:
                            #p = letters.index(word[0])
                        #     if word not in indexes[p]:
                        #         indexes[p][word] = {doc_id: 1}
                        #     else:
                        #         if doc_id not in indexes[p][word][doc_id]:
                        #             indexes[p][word][doc_id] = 1
                        #         else:
                        #             indexes[p][word][doc_id] += 1
                            if word not in inverted_index:
                                inverted_index[word] = {doc_id: 1}
                            else:
                                if doc_id not in inverted_index[word]:
                                    inverted_index[word][doc_id] = 1
                                else:
                                    inverted_index[word][doc_id] += 1
                            term_cnt += 1
                            #get_idf_score(doc, doc_cnt)
                            #get_tfidf_score(doc, doc_cnt, doc_id)
                        title = soup.find_all(['title'])
                        h1 = soup.find_all(['h1'], text=True)
                        h2 = soup.find_all(['h2'], text=True)
                        h3 = soup.find_all(['h3'], text=True)
                        b = soup.find_all(['b'], text=True)
                        strong = soup.find_all(['strong'], text=True)
                        if title is not None:
                            important_content[doc_id].append("title")
                        if h1 is not None:
                            important_content[doc_id].append("h1")
                        if h2 is not None:
                            important_content[doc_id].append("h2")
                        if h3 is not None:
                            important_content[doc_id].append("h3")
                        if b is not None:
                            important_content[doc_id].append("b")
                        if strong is not None:
                            important_content[doc_id].append("strong")
                        doc_id += 1
            except:
                pass
        else:
            continue

        if(len(inverted_index) > len(doc_dict)/partition):
            #break

            #create partial index
            path = "D:\partial_indexes"
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path + "/p"+str(partial_cnt)+".json", 'w') as p:
                json.dump(inverted_index, p)
            p.close()
            partial_cnt += 1
            term_cnt += len(list(inverted_index.keys()))
            inverted_index.clear()
            # path = 'D:\letter_partial_indexes'
            # for l in letters:
            #     p = open(path + "/" + str(l) + ".json", 'w')
            #     try:
            #         letter_index = json.load(p)
            #     except:
            #         letter_index = defaultdict(dict)
            #     for term in inverted_index:
            #         if term[0] == l:
            #             if term in letter_index:
            #                 for id, freq in inverted_index[term].items():
            #                     if id in letter_index[term]:
            #                         letter_index[term][id] += freq
            #                     else:
            #                         letter_index[term][id] = freq
            #             else:
            #                 letter_index[term] = inverted_index[term]
            #     json.dump(letter_index, p, indent=3)



def merge_single_posting(pi1_t, pi2_t):
    # id1 = pi1_t["doc_id"]
    # id2 = pi2_t["doc_id"]
    if pi1_t == None:
        return pi1_t
    if pi2_t == None:
        return pi2_t
    id1 = list(pi1_t.keys())[0]
    id2 = list(pi2_t.keys())[0]
    if id1 < id2:
        return pi1_t.update(pi2_t)
    if id1 > id2:
        return pi2_t.update(pi1_t)
    if id1 == id2:
        for d1 in pi1_t.keys():
            for d2 in pi2_t.keys():
                # if d2["doc_id"] == d1["doc_id"]:
                #     d1["freq"] += d2["freq"]
                if d2 == d1:
                    pi1_t[d1] += pi2_t[d2]
                if d2 != d1 and d2 not in pi1_t.keys():
                    pi1_t[d2] = pi1_t[d2]
        return pi1_t


def merge_and_write(directory):
    merged_path = open(directory+"/merged.json", 'a+')
    # p1 = open(directory+"/p1.txt", "r")
    # p1index = json.load(p1)
    # p1.seek(0)
    # p1.truncate()
    #merged_index = json.dump(p1index, merged_path, indent=3)
    print(merged_path)
    merged_index = defaultdict(dict)
    #p1.close()
    for file in os.listdir(directory)[1:]:
        print(file)
        p = open(directory+"/"+file, "r+")
        print(p)
        p_js = p.read()
        partial_index = json.loads(str(p_js))
        p.seek(0)
        p.truncate()
        json.dump(partial_index, merged_path, indent=3)
        p.close()
        for term in sorted(partial_index):
            if term in merged_index:
                merged_index[term] = merge_single_posting(partial_index[term], merged_index[term])
            else:
                merged_index[term] = partial_index[term]

    return merged_index

def letter_partial_index():
    merged_index = merge_and_write("D:\partial_indexes_copy3")
    print(len(merged_index))
    sorted_merge = sorted(merged_index.items(), key=lambda x: x[0])
    sorted_merge_dict = dict(sorted_merge)
    print(len(sorted_merge_dict))
    #merge_file = open("D:\partial_indexes_copy3/merged.json", 'w')
    #bk_file = open("D:\partial_indexes_copy3/bookkeeping.json", 'w')
    #json.dump(sorted_merge_dict, merge_file, indent=3)
    path = 'D:\letter_partial_indexes1'
    for i in range(26):
         p = open(path + "/"+str(letters[i])+".json", 'w')
         letter_index = defaultdict(dict)
         for term in sorted_merge_dict:
            #bookkeeping[term] = list(sorted_merge_dict.keys()).index(term)
            if type(term) is int or term is None or sorted_merge_dict[term] is None:
                if term == 'retriev':
                    print(merged_index['retriev'])
                continue
            if term[0] != letters[i]:
                continue
            letter_index[term].update(sorted_merge_dict[term])
         json.dump(letter_index, p, indent=3)
    #json.dump(bookkeeping, bk_file, indent=3)

def letter_partial_index2():
    create_letter_partial_index()
    path = 'D:\letter_partial_indexes1'
    for l in letters:
        letter_index = dict()
        for pi_path in os.listdir("D:\partial_indexes_copy3"):
            pi = {}
            with open("D:\partial_indexes_copy3/" + pi_path, 'r') as f:
                temp_i = eval(f.read())
            if l != "0":
                for w in [key for key in temp_i.keys() if key[0] == l]:
                    pi[w] = temp_i[w]
            else:
                for w in [key for key in temp_i.keys() if key[:1] not in letters]:
                    pi[w] = temp_i[w]
            letter_index.update(pi)
        letter_index_file = open(path + "/"+str(l)+".json", 'w')
        json.dump(letter_index, letter_index_file, indent=3)
        










#create_letter_partial_index()
#buildIndex("D:\Dev")
#letter_partial_index2()


#
# print("doc_cnt: ", doc_cnt)
# print("term_cnt: ", term_cnt)

# path = "D:\stats"
# if not os.path.exists(path):
#     os.makedirs(path)
#     # with open(path + "/idf.json", 'w') as p:
#     #      json.dump(idf_dict, p, indent=3)
#     # p.close()
#     # with open(path + "/tfidf.json", "w") as p1:
#     #     json.dump(tf_idf_dict, p1, indent=3)
#     # p1.close()
#     with open(path + "/hashes.json", "w") as p2:
#         json.dump(hash_values, p2, indent=3)
#     p2.close()
#
# #merged_index = merge_and_write("D:\partial_indexes_copy")
# #merged_path = open("D:\partial_indexes_copy\merged.json", "r")
#
#
#
#















