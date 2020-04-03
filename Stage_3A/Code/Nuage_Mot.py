#Nuage de mots
#make wordcoud

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

test = "Les ancêtres des publications de presse telles qu'on les connaît aujourd'hui datent du début du xviie siècle, avec les premières gazettes qui rendent compte plus ou moins régulièrement de l'actualité dans des articles distincts. En 1631, La Gazette de Théophraste Renaudot publie des nouvelles de l'étranger et de la Cour. Le ton de ses articles étant jugé trop neutre ou trop soumis au pouvoir, d'autres publications font leur apparition, privilégiant les articles de commentaires. La Révolution française, qui consacre « la libre communication de la pensée et des opinions », permet à tout citoyen d'écrire et d'imprimer librement. Les critiques et les prises de position constituent alors l'essentiel des articles de l'époque."


show_wordcloud(test)
