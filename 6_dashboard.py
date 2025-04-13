import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

OUTPUTS_DATA_DIR_PATH = './data/outputs'
FALSE_POSITIVE_ENTITIES = ('único', 'kinda', 'netflix', 'finaly', 'va', 'casa', 'soltado que', 'cada', 'marihuana',
                           'el cuerpo', 'chico', 'noche esporádica', 'llevado', 'el', 'quedo', 'buen', 'hacia',
                           'la cita', 'grindr', 'pagan', 'comfort zone', 'anorexic',
                           'meth', 'thou', 'fort', 'pica', 'ubiquiti', 'weekends', 'phobia', 'tina')


def get_sentiment_category(score):
    if score < 0.0:
        return 'negative'
    elif score < 0.5:
        return 'neutral'
    else:
        return 'positive'


def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    if num > 1000:
        if not num % 1000:
            return f'{num // 1000} K'
        return f'{round(num / 1000, 1)} K'
    return f'{num}'


def make_wordcloud(df, text_col):
    all_text = ' '.join(df[text_col])
    wordcloud = WordCloud(background_color='black').generate(all_text)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig


def make_scatter_map(df, lat_col, lon_col, size_col, template='plotly_dark'):
    df_tmp = df.rename(columns={"num_comments":'# comments', "upvote_ratio":'Upvote ratio', 'location': 'place'}, inplace=False)

    scatter = px.scatter_geo(df_tmp, lat=lat_col, lon=lon_col, size=size_col, template=template, 
                             hover_data=['place', '# comments', 'Upvote ratio'], hover_name='title')

    scatter.update_layout(
        hoverlabel=dict(
            font_size=16
        )
    )

    return scatter


def make_donut(input_response, input_text, input_color):
  if input_color == 'blue':
      chart_color = ['#29b5e8', '#155F7A']
  if input_color == 'green':
      chart_color = ['#27AE60', '#12783D']
  if input_color == 'orange':
      chart_color = ['#F39C12', '#875A12']
  if input_color == 'red':
      chart_color = ['#E74C3C', '#781F16']
    
  source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
  source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
  plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=130, height=130)
    
  text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
  plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=130, height=130)
  return plot_bg + plot + text


if __name__ == "__main__":
    df_coordinates = pd.read_csv(f'{OUTPUTS_DATA_DIR_PATH}/coordinates.csv', sep=';')

    loc2coord = {}
    for index, row in df_coordinates.iterrows():
        loc = row['location']
        lat = row['lat']
        lon = row['lon']
        if loc not in FALSE_POSITIVE_ENTITIES:
            loc2coord[loc] = loc2coord.get(loc, {'lat': lat, 'lon': lon})

    with open(f'{OUTPUTS_DATA_DIR_PATH}/data_classified.json') as f:
        data_classification = json.load(f)

    plot_data = {'title': [],
                 'title_clean': [],
                 'upvote_ratio': [],
                 'score': [],
                 'num_comments': [],
                 'sentiment_score': [],
                 'sentiment_category': [],
                 'risk': [],
                 'risk_proba': [],
                 'location': [],
                 'lat': [],
                 'lon': []
    }

    for post in data_classification:
        text_clean = post['selftext_clean']
        for loc in loc2coord.keys():
            if loc in text_clean:
                plot_data['title'].append(post['title'])
                plot_data['title_clean'].append(post['title_clean'])
                plot_data['upvote_ratio'].append(post['upvote_ratio'])
                plot_data['score'].append(post['score'])
                plot_data['num_comments'].append(post['num_comments'])
                plot_data['sentiment_score'].append(post['sentiment_score'])
                plot_data['sentiment_category'].append(get_sentiment_category(post['sentiment_score']))
                plot_data['risk'].append(post['risk'])
                plot_data['risk_proba'].append(post['risk_proba'])
                plot_data['location'].append(loc.capitalize())
                plot_data['lat'].append(loc2coord[loc]['lat'])
                plot_data['lon'].append(loc2coord[loc]['lon'])
                
    df_plot = pd.DataFrame(plot_data)

    st.set_page_config(
        page_title="Reddit Suicide Crisis and Mental Health",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded")

    alt.themes.enable("dark")

    with st.sidebar:
        st.title('📈 Reddit Suicide Crisis and Mental Health')
        
        sent_cat_list = list(df_plot.sentiment_category.unique())[::-1]
        
        selected_sent_cat = st.selectbox('Select a sentiment category', sent_cat_list, index=len(sent_cat_list)-1)

        df_selected_sent_cat = df_plot[df_plot.sentiment_category == selected_sent_cat]
        
        df_top_places = df_selected_sent_cat.location.value_counts().rename_axis('location').reset_index(name='counts')

        df_selected_sent_cat = pd.merge(df_selected_sent_cat, df_top_places, on='location', how='left')

    col = st.columns((1.5, 4.5, 2), gap='medium')

    with col[1]:
        st.markdown('#### Posts Distribution')
        
        scatter_map = make_scatter_map(df_selected_sent_cat, 'lat', 'lon', size_col=None)
        scatter_map.update_layout(font=dict(size=60))
        st.plotly_chart(scatter_map, use_container_width=True, key='map')

        wordcloud = make_wordcloud(df_selected_sent_cat, 'title_clean')
        st.pyplot(wordcloud, use_container_width=True)


    with col[0]:
        st.markdown('#### Statistics')

        st.metric(label='Posts', value=df_selected_sent_cat.shape[0], delta=None)

        st.metric(label='Comments', value=df_selected_sent_cat.num_comments.sum())
        v1 = round(100* df_selected_sent_cat.shape[0] / df_plot.shape[0])
        v2 = round(100*df_selected_sent_cat.upvote_ratio.mean())

        donut_chart_selected = make_donut(v1, 'Sentiment category', 'blue')
        donut_chart_upvote_ratio = make_donut(v2, 'Average Upvote Ratio', 'orange')

        st.write('Sentiment category')
        st.altair_chart(donut_chart_selected)
        st.write('Average Upvote Ratio')
        st.altair_chart(donut_chart_upvote_ratio)


    with col[2]:
        st.markdown('#### Top Locations')

        st.dataframe(df_top_places,
                    column_order=("location", "counts"),
                    hide_index=True,
                    width=None,
                    column_config={
                        "location": st.column_config.TextColumn(
                            "Location",
                        ),
                        "counts": st.column_config.ProgressColumn(
                            "# Posts",
                            format="%f",
                            min_value=0,
                            max_value=max(df_top_places.counts),
                        )}
                    )
        
        with st.expander('About', expanded=True):
            st.write('''
                - Data: Reddit posts collected via the Reddit API using keywords related to suicide, addictions and mental health issues.
                - :orange[**Sentiment category**]: percentage of posts from selected sentiment category over the whole dataset.
                - :orange[**Average Upvote Ratio**]: Average upvote vs downvote ratio of posts from the selected sentiment category.
                - Dashboard developed by [Galo Castillo-López](<https://sites.google.com/view/galocst/>).
                ''')

