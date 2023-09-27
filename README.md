# text_summarization
Here I have shown various techniques of extractive text summarization in NLP. 
The techniques I have shown are;
    a.Page-Rank score based
    b.Maximum marginal relevance (MMR)based
    c.Cluster based
a. Page-Rank based: Here I have basically calculated page rank for stences and created a graph where edge weights are page ranks.
  Then took a user input which the number of lines in which he/she wants the summary and gave it int"summary_PR.txt"
b.MMR based: Here I select most relevant lines based on MMR and store them in "Summary_MMR.txt"
c.Cluster based: Here I have created clusters using K_means algorithm and for each cluster ,I have created a bigram graph and then using that graph I have created one summary line for each cluster and stored in "summary_clusterwise.txt".
