
# coding: utf-8

# # Rendu de TP
# 
# ### Quentin MAIRE et Lola NOYELLE

# In[4]:


import ssl
import urllib.request as web
import feedparser as fp
import newspaper as np
import json
import spacy
from tqdm import tqdm


# ## Recuperation des fichiers RSS

# In[5]:


# On recupere le fichier RSS

url = "https://investir.lesechos.fr/RSS"
fn = "info-marches-investir-bourse-les-echos.xml"
data = fp.parse("/".join([url,fn]))

# On visualise ses elements

print(data.feed.title)
print(data.feed.published)

# Puis on itere sur ses entrees

for item in data.entries:
    print(item.title, item.published, item.link)


# ### Premiere Question
# 
# #### Lecture de la base de donnees existantes

# In[6]:


with open("france-info-20200301.json",'r', encoding='utf-8') as file:
    donnees = json.load(file)


# In[7]:


url = 'https://www.francetvinfo.fr/monde/europe/video-l-estonie-est-le-premier-pays-europeen-a-s-etre-dote-d-ambulances-connectees-pour-le-bien-des-patients_3785827.html#xtor=RSS-3-[monde/europe]'

print(donnees[url])


# In[8]:


urls = ["https://www.francetvinfo.fr/france.rss","https://www.francetvinfo.fr/europe.rss","https://www.francetvinfo.fr/entreprises.rss"]

liens = donnees.keys()
identifiants = []
for lien in liens:
    index = lien.find('.html')
    identifiant = lien[index-7:index-1]
    identifiants.append(identifiant)
    
# On recupere le fichier RSS
for url in urls :
    data = fp.parse(url)
    fr = url.find('.fr/')
    category = url[fr+4:].replace(".rss","")
    
    for item in data.entries:
        #Boucle permettant le scan de l'article
        link = item.link
        index = link.find('.html')
        numero = link[index-7:index-1]
        
        if numero not in identifiants:
            #On verifie si l'article est deja presentou non dans la base
            article = np.Article(link)
            article.download()
            article.parse()
            #Afin d'avoir une base la plus complète pour la suite, nous avons décidé de garder l'ensemble de ces infos
            donnees[link] = {'title' : article.title,
                             'date' : article.publish_date.isoformat(), 
                             'author' : article.authors, 
                             'category' : category, 
                             'content' : article.text, 
                             'image_link' : article.top_image}
         
            

    


# #### A quoi sert la librairie Newspaper3k ?
# 
# La librairie **newspaper3k** permet d'extraire les informations d'un site web et d'en faire un resume. Grace aux balises et identifiants presents dans la page HTML, elle peut recuperer les informations demandees.
# 
# Soit dit en passant, on utilise newspaper3k car elle est compatible avec python3 contrairement a newspaper

# #### Mis a jour automatique de la base de donnees 

# In[9]:


#Nous allons juste imbriquer notre precedente procedure dans une fonction

def update(path_donnees,urls):
    with open(path_donnees,'r', encoding='utf-8') as file:
        donnees = json.load(file)
    
    liens = donnees.keys()
    identifiants = []
    for lien in liens:
        index = lien.find('.html')
        identifiant = lien[index-7:index-1]
        identifiants.append(identifiant)

    # On recupere le fichier RSS
    for url in urls :
        data = fp.parse(url)
        fr = url.find('.fr/')
        category = url[fr+4:].replace(".rss","")

        for item in data.entries:
            link = item.link
            index = link.find('.html')
            numero = link[index-7:index-1]
            
            if numero not in identifiants:
                article = np.Article(link)
                article.download()
                article.parse()
                donnees[link] = {'title' : article.title,
                                 'date' : article.publish_date.isoformat(), 
                                 'author' : article.authors, 
                                 'category' : category, 
                                 'content' : article.text, 
                                 'image_link' : article.top_image}
    # On rajoute l'ecriture finale            
    with open(path_donnees,'w', encoding='utf-8') as file:
        file.write(json.dumps(donnees))


# In[10]:


urls = ["https://www.francetvinfo.fr/france.rss","https://www.francetvinfo.fr/europe.rss","https://www.francetvinfo.fr/entreprises.rss"]

update("france-info-20200301.json",urls)


# #### Spacy
# 
# Spacy utilise un modèle (ici fr_core_news_sm-2.2.5) pour faire des prédictions. 
# 
# Les étiquettes IOB (Inside-Outside-Beginning) désignent si un mot est respectivement à l'intérieur, à l'extérieur ou au début d'une entité.
# 
# #### Récupération des entités nommées
# 
# On souhaite conserver les entités nommées de personne, d’organisation et de lieu
# 

# In[13]:


path = "C:\\Users\\Armelle KOEHL\\Desktop\\NLP\\" #A modifier/automatiser
nlp = spacy.load(path + "fr_core_news_sm-2.2.5")
with open("france-info-20200301.json",'r', encoding='utf-8') as file:
    donnees = json.load(file)


# In[21]:


get_ipython().run_cell_magic('time', '', 'keys = list(donnees.keys())\ndoc = nlp(donnees[keys[0]]["content"])\nfor entity in doc.ents:\n    if(entity.label_ in ["PER","LOC","ORG"]):\n        print(entity.text, entity.start_char, entity.end_char, entity.label_)')


# Ici on affiche successivement : Le texte, la délimitation en terme d'indice du premier et dernier caractère du texte, et si c'est une personne, organisation ou lieu.
# 
# Après plusieurs itérations de ce process, on constate qu'il faut en moyenne 60 ms pour récupérer toutes les entités nommées citées précédemment. 
# 
# Par conséquent on peut estimer le temps du process sur plusieurs documents comme suit :
# 
# 60 ms x (nombre_de_documents) + (temps de passage entre deux docs)x(nombre_de_documents -1).
# 
# En soit le second terme de l'addition peut se négliger par rapport au premier terme.

# In[23]:


get_ipython().run_cell_magic('time', '', 'entites_corpus = {}\nkeys = list(donnees.keys())\nfor element in keys:\n    doc = nlp(donnees[element]["content"])\n    entites_doc = {}\n    for entity in doc.ents:\n        if(entity.label_ in ["PER","LOC","ORG"]):\n            entites_doc = {"texte" : entity.text, \n                           "start_char" : entity.start_char, \n                           "end_char" : entity.end_char, \n                           "entity" : entity.label_}\n    if entites_doc != {}:\n        entites_corpus[element] = entites_doc\n            ')


# Sur l'ensemble des articles, on a besoin en moyenne de 50s pour obtenir toutes les entités nommées.

# ### Entités les plus fréquentes
# 
# On va s'y prendre en trois phases :
# 
# 1) On "decoupe nos entites" en 4 parties
# 
# 2) On compte les occurences de chaques entites
# 
# 3) On classe les nombres d'occurences pour en sortir les 20 plus citees

# In[39]:


def decoupage(donnees):
    decoupe = {}
    for key in list(donnees.keys()):
        decoupe[key] = []
        doc = nlp(donnees[key]["content"])
        for entity in doc.ents:
            one_entity = {}
            if entity.label_ in ("PER","LOC","ORG"):
                one_entity["labels"] = entity.label_
                one_entity["texte"] = entity.text
                one_entity["debut"] = entity.start_char
                one_entity["fin"] = entity.end_char
            decoupe[key].append(one_entity)
    return decoupe


# In[40]:


def compte_occurence(decoupe):
    occ = {"PER":{},"LOC":{},"ORG":{}}
    for article in list(decoupe.keys()):
        for label in decoupe[article]:
            if label!={}:
                if label["texte"] not in occ[label["labels"]]:
                    occ[label["labels"]][label["texte"]]=1
                else:
                    occ[label["labels"]][label["texte"]]+=1
    return occ


# In[41]:


def classement(occ,nb_voulu=20):
    rank = {}
    for label in list(occ.keys()):
        rank_ter = sorted(list(occ[label].items()), key = lambda x:x[1], reverse=True)[:nb_voulu]
        liste_ter = []
        for a in rank_ter:
            liste_ter.append(a[0])
        rank[label] = liste_ter
    return rank


# In[42]:


decoupe = decoupage(donnees)
occ = compte_occurence(decoupe)
rank = classement(occ)


# In[36]:


print(rank["LOC"],rank["PER"],rank["ORG"])


# On a ici nos trois listes séparées par entités. 
# Ces listes me confirment bien que l'actualisation de la base est correcte car on voit apparaitre des termes relatifs à l'actualités récentes (OMS, Wuhan

# ### Co-Occurence

# In[43]:


def co_occurences(entite_1,entite_2,decoupe):
    occ = {}
    for article in list(decoupe.keys()):
        compte_entite_1 = ["",0] #La seconde valeur servira à connaitre la presence de l'entite ou pas
        compte_entite_2 = ["",0]
        #Pour tout les articles
        for label in decoupe[article]:
            if label!={}:
                
                if label["texte"] == entite_1: 
                    compte_entite_1 = [label["labels"],1] #L'entite est presente et on garde son label (PER,LOC,ORG)
                if label["texte"] == entite_2:
                    compte_entite_2 = [label["labels"],1]
        if compte_entite_1[1] == 1 and compte_entite_2[1] == 1 : #Si les deux sont presents
            entite = compte_entite_1[0]+": "+entite_1+", "+compte_entite_2[0]+": "+entite_2 #On cree une entite unique de la forme :
            #PER: Emmanuel Macron, LOC: Wuhan
            if entite not in occ: #On rajoute l'entite cree si elle n'existe pas
                occ[entite]=1
            else:
                occ[entite]+=1 #On instancie de 1 si elle existe deja
    return occ


# #### Sur nos précédentes listes
# 
# On va appliquer notre fonction précédente à l'ensemble des couples possibles issues de nos 3 listes
# 
# Dans ce cas précis, nous allons traiter couples du genre : (Wuhan,Macron) et (Macron,Wuhan) en supprimant une entite une fois choisie une fois et une fois que tout les couple spossibles sont fait. De plus il serait possible aussi de chercher a préciser ce décompte en rapportant les mots President de la republique et France à Emmanuel Macron.

# In[51]:


def co_occurences_corpus(decoupe,rank):
    compte_co_occurences={}
    Lind = list(rank.keys())
    for key in rank.keys():
        Lind.remove(key) #Permet de ne pas avoir les couples (a,b) et (b,a)
        for cle in Lind:
            for entite_1 in rank[key]: #Parcourt des entités une a une
                for entite_2 in rank[cle]:
                    compte_co_occurences = {**compte_co_occurences,**co_occurences(entite_1,entite_2,decoupe)}
    return compte_co_occurences


# In[52]:


def classement_co_occurences(occ):
    #Permet le classement 
    rank_ter = sorted(list(occ.items()), key = lambda x:x[1], reverse=True)[:20]
    return rank_ter


# In[53]:


co_occ = co_occurences_corpus(decoupe,rank)
classement_co_occurences(co_occ)


# Les résultats obtenus nous semblent pertinents. En effet, le Brexit à récemment été mis en application ce qui explique pourquoi on retrouve beaucoup d'article sur le Brexit et les co_occurences avec le Royaume uni, l'union européenne ou encore Londres. 
# 
# De ce résultat, on peut remarquer deux choses: 
# 
# 1) On pourrait essayer de rassembler les Acronymes des noms auquels il se rapportent (UE pour Union Européenne)
# 
# 2) On pourrait supprimer les couples (Brexit,Brexit) ou encore les couples (a,b), (b,a) qui semblent ne pas etre totalement supprimé
