import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
import networkx as nx
import scipy as scipy
import math
import random

stopWords = set(stopwords.words('english'))

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')





doc=open('input.txt','r')

sent = []
for s in doc:
    sent.append(s)

preprocessed_sentences = []
for s in sent:
    res=""
    for c in s:
       if c not in string.punctuation:
           res+=c
    
    
    if res!="":
       words = word_tokenize(res)
    

    fwords = []
    for word in words:
        if word not in stopWords:
            fwords.append(WordNetLemmatizer().lemmatize(word) )
    

    ps=""
    ind=1
    for w in fwords:
        if ind!=len(fwords):
            ps+=(w+' ')
        else:
            ps+=w
        ind=ind+1
    
    preprocessed_sentences.append(ps)


ind=1
for s in preprocessed_sentences:
    print(f"List[{ind}] = {s}")
    ind=ind+1

fnames=set()
for s in preprocessed_sentences:
    temp=s.split()
    for w in temp:
        fnames.add(w)
print("\nWords (Features):")
print(fnames)
tf_Matrix=[]
for st in preprocessed_sentences:
    w_c={}
    t=st.split()
    for w in t:
        if w in w_c:
            w_c[w]+=1
        else:
            w_c[w]=1
    tfid={}
    for w,c in w_c.items():
         tfid[w]=c
    tf_Matrix.append(tfid)
#print(tf_Matrix)
idf={}
for w in fnames:
    n=0
    for s in preprocessed_sentences:
        if w in s:
            n+=1
    idf[w]=float(math.log(len(preprocessed_sentences)/(n)))

#print(idf)
f_names_sorted=sorted(fnames)
tfidMatrix=[]
temp=[]
for tf_vec in tf_Matrix:
      vec={}
      for w,tf in tf_vec.items():
         if w in idf.keys():
             vec[w]=tf * idf[w]
         else:
             vec[w]=0.0
      temp.append(vec)
#print(temp)
for tfidf_vector in temp:
    row = [tfidf_vector.get(word, 0) for word in f_names_sorted]
    tfidMatrix.append(row)
print(tfidMatrix)
tfidfArray=np.array(tfidMatrix)
print(tfidfArray)


G = nx.Graph()
sent_num = len(preprocessed_sentences)
print(sent_num)
for node in range(sent_num):
    G.add_node(node)

for i in range(sent_num):
    for j in range(i+1,sent_num):
        cosine = np.dot(tfidfArray[i],tfidfArray[j])/(norm(tfidfArray[i], axis=0)*norm(tfidfArray[j]))
        print(f"{i} to {j},weight is : {cosine} ")
        G.add_edge(i, j, weight=cosine)

rankScores = nx.pagerank(G,alpha=0.9)


n = int(input("enter the number of lines you want in page-rank based summary: "))  
rank_highest = sorted(rankScores, key=rankScores.get, reverse=True)[:n]
rank_highest.sort()  
id=1
ff=open('summary_PR.txt','w')
for x in rank_highest:
    ff.write(sent[x])
    id=id+1
####################################### TASK 0 Day2 #####################
selected=[]
selected.append(0)
print(rankScores)

lbda=0.5
sentences=preprocessed_sentences.copy()
sentences.remove(sentences[0])
# for key in rankScores.keys():
#     rank_new[key]=lambda * rankScores[key]-(1-lambda)
# print("enter number of iterations: \n")
# n_iter=int(input())
# if n_iter>len(sentences):
#     print("invalid input")
while len(sentences)!=0:
    rank_new={}
    for i in range(len(sentences)):
          mx=-100
          for j in selected:
              cosine = np.dot(tfidfArray[i],tfidfArray[j])/(norm(tfidfArray[i], axis=0)*norm(tfidfArray[j]))
              mx=max(mx,cosine)
          rank_new[i]=lbda*rankScores[i]-(1-lbda)*mx
    rank_new_highest = sorted(rank_new, key=rank_new.get, reverse=True)[:1]
    #print(rank_new_highest[0])
    selected.append(rank_new_highest[0])
    sentences.remove(sentences[rank_new_highest[0]])
res=set(selected)
f=open('Summary_MMR.txt','w')
for i in res:
    f.write(preprocessed_sentences[i]+'\n')

############################### TASK 1 #########################################
def find_if_close(list1,list2):
    return np.allclose(list1,list2)
k=int(input("enter the number of lines in cluster based summary: "))
##initializing centroids randomly##
centroids=random.sample(tfidMatrix,k)
n=len(centroids)
#centroids.sort()
cluster_ans=[]
max_iter=50
while True:
    clusters=[]
    clusters_id=[]
    for i in range(n):
        clusters.append([])
        clusters_id.append([])
    for v in range(len(tfidfArray)):
        similarities=[]
        for centr in centroids:
            cosine = np.dot(tfidfArray[v],centr)/(norm(tfidfArray[v], axis=0)*norm(centr))
            similarities.append(cosine)
        closest_cluster = np.argmax(similarities)
        clusters[closest_cluster].append(tfidfArray[v])
        clusters_id[closest_cluster].append(v)
    centroids_modified=[]
    for cluster in clusters:
        centroids_modified.append(np.mean(cluster,axis=0))
    #(centroids_modi
    # fied.all()).sort()
    if find_if_close(centroids,centroids_modified):
        break
    centroids=centroids_modified
    cluster_ans=clusters
    max_iter-=1
ind=0
#final_cluster=[]
for cluster in clusters:
   print(cluster)
   print('\n')
print(clusters[0][0])


######## TASK 2 DAY 2 ###########
'''
class node:
    def __init__(self,str1):
        temp=str1.split()
        self.first=temp[0]
        self.second=temp[1]
        self.next=[]
s="i am a boy"
words = s.split()
result = [' '.join(pair) for pair in zip(words, words[1:])]
nodes=[]
for i in result:
     nodes.append(node(i))
for i in range(len(nodes)-1):
    temp=nodes[i].next
    temp.append(nodes[i+1])
start=node("\0 \0")
start.next.append(nodes[0])
for node in nodes:
    print(f'{node.first} and {node.second} and {node.next}')
'''
def constructGraph(str1, str2):
    # Initialize the graph with start and end nodes.
    graph = {
        "start": [],
        "end": []
    }

    # Tokenize the sentences into bigrams.
    bigrams1=[]
    bigrams2=[]
    temp1=str1.split()
    for i in range(len(temp1)-1):
        bigrams1.append(tuple(temp1[i:i+2]))
    if str2:
        temp2=str2.split()
        for i in range(len(temp2)-1):
            bigrams2.append(tuple(temp1[i:i+2]))
    # Add nodes for bigrams and edges to/from start and end nodes.
    for bigram in bigrams1:
        if bigram not in graph:
            graph[bigram] = []
        if bigram == bigrams1[0]:
            graph["start"].append(bigram)
        if bigram == bigrams1[-1]:
            graph[bigram].append("end")

    for bigram in bigrams2:
        if bigram not in graph:
            graph[bigram] = []
        if bigram == bigrams2[0]:
            graph["start"].append(bigram)
        if bigram == bigrams2[-1]:
            graph[bigram].append("end")

    # Add edges between consecutive bigrams.
    for i in range(0,len(bigrams1) - 1):
        graph[bigrams1[i]].append(bigrams1[i + 1])

    for i in range(0,len(bigrams2) - 1):
        graph[bigrams2[i]].append(bigrams2[i + 1])

    return graph
def count_common_bigrams(str1, str2):
    # Tokenize the sentences into bigrams.
    
    temp1=str1.split()
    bigrams1=set()
    for i in range(0,len(temp1) - 1):
        bigrams1.add(tuple(temp1[i:i+2]))
    temp2=str2.split()
    bigrams2=set()
    for i in range(0,len(temp2) - 1):
        bigrams2.add(tuple(temp2[i:i+2]))
    
    # Calculate the intersection of the two sets of bigrams.
    cnt=0
    for bigram in bigrams1:
        for bgram in bigrams2:
            if bigram==bgram:
                cnt+=1
    
    # Return the count of common bigrams.
    return cnt 
ind=0
clusters_summ={}
for k in range(len(clusters)):
        if not clusters[k]:
            continue
        mx=-1000
        mxid=0
        #### choosing closest sentence id
        for p  in range(len(clusters[k])):
            cosine = np.dot(centroids[k],clusters[k][p])/(norm(centroids[k], axis=0)*norm(clusters[k][p]))
            if mx<cosine:
                mx=cosine
                mxid=p
        # Select the first sentence as S1.
        s1 = preprocessed_sentences[clusters_id[k][mxid]]
        
        # Find S2 with at least 3 common bigrams with S1.
        s2 = None
        for j in clusters_id[k]:
            #print(count_common_bigrams(s1, preprocessed_sentences[j]))
            if count_common_bigrams(s1, preprocessed_sentences[j]) >= 3:
                s2 =  preprocessed_sentences[j]
                break

        # Construct the sentence graph.
        if s2:
            if s1!=s2:
                s_g= constructGraph(s1, s2)
            else:
                s_g= constructGraph(s1, None)
        else:
            s_g= constructGraph(s1, None)

        # Generate a sentence using the graph.
        random_path ="" 
        currentNode = "start"
    
        while currentNode != "end":
            
            next_bigram = random.choice(s_g[currentNode])
            
            if next_bigram == "end":
                break
            
            random_path+=(" "+next_bigram[0])
            currentNode = next_bigram

        print(f"Cluster{ind} : {random_path}")
        clusters_summ[ind]=random_path
        ind+=1
################ TASK 3 DAY2 ###############
min_to_ind={}
for i in range(len(clusters_id)):
     mn=90000
     for j in clusters_id[i]:
            mn=min(mn,j)
     min_to_ind[mn]=i

#print(min_to_ind)
#print(clusters_id)
ind=0
temp=list(min_to_ind.keys())
#print(type(temp))
temp.sort()
fff=open('summary_clusterwise.txt','w')
for k in temp:
    x=min_to_ind[k]
    fff.write(f'line {ind}:{clusters_summ[x]}\n')
    ind+=1
