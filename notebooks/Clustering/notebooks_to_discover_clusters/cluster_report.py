import numpy as np
import pandas as pd
import pickle
import copy
import os
import time
from IPython.display import HTML, clear_output

from sklearn.metrics.pairwise import cosine_similarity

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

clear_output()

def check_path(path):
    os.system("if [ ! -d " + path + " ]; then mkdir -p " + path + "; fi")

def save_file(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)
        
def load_file(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file

def generate_wordcloud_svg(cluster_texts, width=600, height=400):
    text_to_wordcloud = " ".join(cluster_texts)
    wordcloud = WordCloud(background_color="white", width=width, height=height).generate(text_to_wordcloud)
    display(HTML(wordcloud.to_svg(embed_font=True)))
    #return wordcloud.to_svg(embed_font=True)


def generate_report_for_cluster(cluster_id, texts, clusters, vectors, cluster_dict, k=10):
    display(HTML("<h2>Cluster id: "+str(cluster_id)+" </h2>"))
    
    ## general statistics
    count_texts = len(cluster_dict[cluster_id])
    percent_of_texts = round(100*count_texts/len(clusters), 2)
    
    display(HTML("<p style='font-size:16px'>Texts in cluster: <b>"+str(count_texts)+" </b></p>"))
    display(HTML("<p style='font-size:16px'>Occurs in <b>"+ str(percent_of_texts)+"%</b> from all texts </p>"))
    display()
    
    ## wordcloud
    cluster_texts = texts[cluster_dict[cluster_id]]
    generate_wordcloud_svg(cluster_texts)
    
    
    ## texts that represent cluster
    cluster_vectors = vectors[cluster_dict[cluster_id]]
    sorted_indexes = get_sorted_indexes(cluster_vectors)
    
    sorted_texts = cluster_texts[sorted_indexes]
    text_df = pd.DataFrame({" ": sorted_texts}).drop_duplicates()
    text_df.index = range(len(text_df))
    
    first_k_texts = text_df.head(k)
    other_texts = text_df.loc[k:, :]
    
    display(HTML("<h3>The unique texts that represent this cluster:</h3>"))
    display(HTML(table_to_html(first_k_texts)))
    
    full_html = "<details><summary>More texts</summary>"
    full_html += table_to_html(other_texts)
    full_html += "</details>"
    
    display(HTML(full_html))
    
def group_cluster_indexes(clusters):
    cl_df = pd.DataFrame({"index": list(range(len(clusters))), "cluster": clusters})
    cluster_dict = cl_df.groupby(['cluster'])['index'].apply(lambda grp: list(grp)).to_dict()
    return cluster_dict

def get_sorted_indexes(cluster_vectors):
    cos_cluster = cosine_similarity(cluster_vectors, cluster_vectors)
    text_sim_to_text_in_cluster = cos_cluster.sum(axis=0)/cos_cluster.shape[0]
    sorted_indexes = np.argsort(text_sim_to_text_in_cluster)[::-1]
    return sorted_indexes

def table_to_html(text_df):
    raw_html = text_df.to_html()
    raw_html = raw_html.replace('<tr>','<tr style="text-align: left;">')
    text_df = text_df.style.set_properties(**{'text-align': 'left', 'font-size': '12px'})
    df_html = text_df.render()
    return df_html