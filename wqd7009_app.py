import streamlit as st
import numpy as np
import pandas as pd

movie_data = pd.read_csv('movie_data.csv')
data = pd.read_csv('data.csv')

#Drop unwanted columns
data2 = data.drop(['user_id','time'],axis=1)
#Use groupby movie-id to calculate mean ratings for each movie_id
data3 = data2.groupby('movie_id').agg({'rating':'mean'}).reset_index()
#Give the mean rating decimal round off to 2 places
data3['rating'] = data3['rating'].round(decimals=2)
#Rename the column from rating to average rating
data3.rename(columns = {"rating": "average_rating"},  
          inplace = True)

#Merge movie_data with rating_data
movie_data = movie_data.merge(data3,how='left', on='movie_id')

#Extract year from movie_data.title column, new column 'year' created
movie_data['year'] = movie_data.title.str.extract('(\(\d\d\d\d\))',expand=False)
#remove parenthesis from year string
movie_data['year'] = movie_data.year.str.extract('(\d\d\d\d)',expand=False)
#Convert year string to integer to enable sorting
movie_data['year'] = movie_data['year'].astype(int)
#Remove '|' from string in genre column
movie_data['genre'] = movie_data["genre"].str.replace(r'|',' ')

#creat list for movie titles
movie_list = movie_data.title.tolist()
#insert new string into list at index 0
movie_list.insert(0,'-')

#create alphabet list
alpha_list = [chr(i) for i in range(ord('A'),ord('Z')+1)]
alpha_list.insert(0,'-')

#Get a list of unique values from year
year_list = movie_data['year'].unique().tolist()
#convert the year in list into integers
year_list_int = list(map(int, year_list))
#Sort the year list in decreasing order
year_list_int.sort(reverse=True)
#convert the year in list to string
year_list_str = list(map(str, year_list_int))
#insert new string into list at index 0
year_list_str.insert(0,'-')
#convert data type of year in year column from integer to string
movie_data['year'] = movie_data['year'].astype(str)


#create list for genre
genre_list = ['-','Action','Adventure','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery', 'Romance','Sci-Fi','Thriller','War','Western']

#load v.npy file
V = np.load('v.npy')

#Function to calculate the cosine similarity (sorting by most similar and returning the top N)
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    #print('Recommendations for {0}: \n'.format(
    #movie_data[movie_data.movie_id == movie_id].title.values[0]))
    recommed_list = []
    for id in top_indexes + 1:
        recommed_list.append(movie_data[movie_data.movie_id == id].title.values[0])
        movie_recommed = pd.DataFrame(recommed_list,columns=['Movie Title'])
        movie_recommed.index = movie_recommed.index + 1
    return movie_recommed  

#function to return movie_id
def get_movie_id(movie_title):
    movie_id = movie_data.movie_id[movie_data['title']==movie_title].values[0]
    return movie_id

#k-principal components to represent movies, movie_id to find recommendations, top_n print n results        
k = 50
top_n = 10
sliced = V.T[:, :k] # representative data

#create title for dashboard
st.title('Movies Recommender System')

#create sidebar with options
status = st.sidebar.radio("View Movies By: ", ('All','Alphabet', 'Year', 'Genre')) 

if (status == 'All'): 
    st.sidebar.success("Movie Selected By " + status) 
    sidebar = st.selectbox('All Movies List', movie_list)
    movie_all = sidebar
    if movie_all =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_all) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_all,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_all)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))
        
elif (status == 'Alphabet'): 
    st.sidebar.success("Movie Selected By " + status) 
    sidebar = st.sidebar.selectbox('Select By Alphabet ',alpha_list)
elif (status == 'Year'): 
    st.sidebar.success("Movie Selected By " + status) 
    sidebar = st.sidebar.selectbox('Select By Year ',year_list_str)
else:                                                
    st.sidebar.success("Movie Selected By " + status) 
    sidebar = st.sidebar.selectbox('Select By Genre ',genre_list)

    
if sidebar == 'A':
    movie_a = movie_data.title[movie_data.title.str.startswith('A')].tolist()
    movie_a.insert(0,'-')
    movie_a = st.selectbox('Select Movie Title ', movie_a)
    if movie_a =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_a) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_a,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_a)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))
        
if sidebar == 'B':
    movie_b = movie_data.title[movie_data.title.str.startswith('B')].tolist()
    movie_b.insert(0,'-')
    movie_b = st.selectbox('Select Movie Title ', movie_b)
    if movie_b =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_b) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_b,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_b)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))

    
if sidebar == 'C':
    movie_c = movie_data.title[movie_data.title.str.startswith('C')].tolist()
    movie_c.insert(0,'-')
    movie_c = st.selectbox('Select Movie Title ', movie_c)
    if movie_c =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_c) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_c,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_b)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'D':
    movie_d = movie_data.title[movie_data.title.str.startswith('D')].tolist()
    movie_d.insert(0,'-')
    movie_d = st.selectbox('Select Movie Title ', movie_d)
    if movie_d =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_d) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_d,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_d)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'E':
    movie_e = movie_data.title[movie_data.title.str.startswith('E')].tolist()
    movie_e.insert(0,'-')
    movie_e = st.selectbox('Select Movie Title ', movie_e)
    if movie_e =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_e) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_e,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_e)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'F':
    movie_f = movie_data.title[movie_data.title.str.startswith('F')].tolist()
    movie_f.insert(0,'-')
    movie_f = st.selectbox('Select Movie Title ', movie_f)
    if movie_f =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_f) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_f,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_f)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'G':
    movie_g = movie_data.title[movie_data.title.str.startswith('G')].tolist()
    movie_g.insert(0,'-')
    movie_g = st.selectbox('Select Movie Title ', movie_g)
    if movie_g =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_g) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_g,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_g)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'H':
    movie_h = movie_data.title[movie_data.title.str.startswith('H')].tolist()
    movie_h.insert(0,'-')
    movie_h = st.selectbox('Select Movie Title ', movie_h)
    if movie_h =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_h) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_h,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_h)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))

        
if sidebar == 'I':
    movie_i = movie_data.title[movie_data.title.str.startswith('I')].tolist()
    movie_i.insert(0,'-')
    movie_i = st.selectbox('Select Movie Title ', movie_i)
    if movie_i =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_i) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_i,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_i)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'J':
    movie_j = movie_data.title[movie_data.title.str.startswith('J')].tolist()
    movie_j.insert(0,'-')
    movie_j = st.selectbox('Select Movie Title ', movie_j)
    if movie_j =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_j) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_j,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_j)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))

        
if sidebar == 'K':
    movie_k = movie_data.title[movie_data.title.str.startswith('K')].tolist()
    movie_k.insert(0,'-')
    movie_k = st.selectbox('Select Movie Title ', movie_k)
    if movie_k =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_k) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_k,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_k)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'L':
    movie_l = movie_data.title[movie_data.title.str.startswith('L')].tolist()
    movie_l.insert(0,'-')
    movie_l = st.selectbox('Select Movie Title ', movie_l)
    if movie_l =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_l) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_l,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_l)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'M':
    movie_m = movie_data.title[movie_data.title.str.startswith('M')].tolist()
    movie_m.insert(0,'-')
    movie_m = st.selectbox('Select Movie Title ', movie_m)
    if movie_m =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_m) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_m,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_m)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'N':
    movie_n = movie_data.title[movie_data.title.str.startswith('N')].tolist()
    movie_n.insert(0,'-')
    movie_n = st.selectbox('Select Movie Title ', movie_n)
    if movie_n =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_n) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_n,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_n)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'O':
    movie_o = movie_data.title[movie_data.title.str.startswith('O')].tolist()
    movie_o.insert(0,'-')
    movie_o = st.selectbox('Select Movie Title ', movie_o)
    if movie_o =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_o) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_o,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_o)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'P':
    movie_p = movie_data.title[movie_data.title.str.startswith('P')].tolist()
    movie_p.insert(0,'-')
    movie_p = st.selectbox('Select Movie Title ', movie_p)
    if movie_p =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_p) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_p,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_p)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))

        
if sidebar == 'Q':
    movie_q = movie_data.title[movie_data.title.str.startswith('Q')].tolist()
    movie_q.insert(0,'-')
    movie_q = st.selectbox('Select Movie Title ', movie_q)
    if movie_q =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_q) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_q,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_q)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'R':
    movie_r = movie_data.title[movie_data.title.str.startswith('R')].tolist()
    movie_r.insert(0,'-')
    movie_r = st.selectbox('Select Movie Title ', movie_r)
    if movie_r =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_r) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_r,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_r)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'S':
    movie_s = movie_data.title[movie_data.title.str.startswith('S')].tolist()
    movie_s.insert(0,'-')
    movie_s = st.selectbox('Select Movie Title ', movie_s)
    if movie_s =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_s) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_s,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_s)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'T':
    movie_t = movie_data.title[movie_data.title.str.startswith('T')].tolist()
    movie_t.insert(0,'-')
    movie_t = st.selectbox('Select Movie Title ', movie_t)
    if movie_t =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_t) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_t,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_t)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'U':
    movie_u = movie_data.title[movie_data.title.str.startswith('U')].tolist()
    movie_u.insert(0,'-')
    movie_u = st.selectbox('Select Movie Title ', movie_u)
    if movie_u =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_u) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_u,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_u)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'V':
    movie_v = movie_data.title[movie_data.title.str.startswith('V')].tolist()
    movie_v.insert(0,'-')
    movie_v = st.selectbox('Select Movie Title ', movie_v)
    if movie_v =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_v) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_v,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_v)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'W':
    movie_w = movie_data.title[movie_data.title.str.startswith('W')].tolist()
    movie_w.insert(0,'-')
    movie_w = st.selectbox('Select Movie Title ', movie_w)
    if movie_w =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_w) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_w,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_w)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'X':
    movie_x = movie_data.title[movie_data.title.str.startswith('X')].tolist()
    movie_x.insert(0,'-')
    movie_x = st.selectbox('Select Movie Title ', movie_x)
    if movie_x =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_x) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_x,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_x)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'Y':
    movie_y = movie_data.title[movie_data.title.str.startswith('Y')].tolist()
    movie_y.insert(0,'-')
    movie_y = st.selectbox('Select Movie Title ', movie_y)
    if movie_y =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_y) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_y,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_y)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == 'Z':
    movie_z = movie_data.title[movie_data.title.str.startswith('Z')].tolist()
    movie_z.insert(0,'-')
    movie_z = st.selectbox('Select Movie Title ', movie_z)
    if movie_z =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success("You Selected: " + movie_z) 
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_z,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_z)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))

if sidebar == '2000':
    movie_2000 = movie_data.title[movie_data['year']=='2000'].tolist()
    movie_2000.insert(0,'-')
    movie_2000 = st.selectbox('Select Movie Title ', movie_2000)
    if movie_2000 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_2000)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_2000,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_2000)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1999':
    movie_1999 = movie_data.title[movie_data['year']=='1999'].tolist()
    movie_1999.insert(0,'-')
    movie_1999 = st.selectbox('Select Movie Title ', movie_1999)
    if movie_1999 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1999)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1999,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1999)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1998':
    movie_1998 = movie_data.title[movie_data['year']=='1998'].tolist()
    movie_1998.insert(0,'-')
    movie_1998 = st.selectbox('Select Movie Title ', movie_1998)
    if movie_1998 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1998)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1998,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1998)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))

        
if sidebar == '1997':
    movie_1997 = movie_data.title[movie_data['year']=='1997'].tolist()
    movie_1997.insert(0,'-')
    movie_1997 = st.selectbox('Select Movie Title ', movie_1997)
    if movie_1997 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1997)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1997,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1997)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1996':
    movie_1996 = movie_data.title[movie_data['year']=='1996'].tolist()
    movie_1996.insert(0,'-')
    movie_1996 = st.selectbox('Select Movie Title ', movie_1996)
    if movie_1996 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1996)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1996,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1996)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1995':
    movie_1995 = movie_data.title[movie_data['year']=='1995'].tolist()
    movie_1995.insert(0,'-')
    movie_1995 = st.selectbox('Select Movie Title ', movie_1995)
    if movie_1995 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1995)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1995,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1995)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1994':
    movie_1994 = movie_data.title[movie_data['year']=='1994'].tolist()
    movie_1994.insert(0,'-')
    movie_1994 = st.selectbox('Select Movie Title ', movie_1994)
    if movie_1994 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1994)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1994,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1994)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1993':
    movie_1993 = movie_data.title[movie_data['year']=='1993'].tolist()
    movie_1993.insert(0,'-')
    movie_1993 = st.selectbox('Select Movie Title ', movie_1993)
    if movie_1993 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1993)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1993,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1993)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1992':
    movie_1992 = movie_data.title[movie_data['year']=='1992'].tolist()
    movie_1992.insert(0,'-')
    movie_1992 = st.selectbox('Select Movie Title ', movie_1992)
    if movie_1992 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1992)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1992,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1992)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1991':
    movie_1991 = movie_data.title[movie_data['year']=='1991'].tolist()
    movie_1991.insert(0,'-')
    movie_1991 = st.selectbox('Select Movie Title ', movie_1991)
    if movie_1991 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1991)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1991,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1991)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1990':
    movie_1990 = movie_data.title[movie_data['year']=='1990'].tolist()
    movie_1990.insert(0,'-')
    movie_1990 = st.selectbox('Select Movie Title ', movie_1990)
    if movie_1990 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1990)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1990,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1990)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1989':
    movie_1989 = movie_data.title[movie_data['year']=='1989'].tolist()
    movie_1989.insert(0,'-')
    movie_1989 = st.selectbox('Select Movie Title ', movie_1989)
    if movie_1989 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1989)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1989,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1989)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1988':
    movie_1988 = movie_data.title[movie_data['year']=='1988'].tolist()
    movie_1988.insert(0,'-')
    movie_1988 = st.selectbox('Select Movie Title ', movie_1988)
    if movie_1988 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1988)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1988,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1988)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1987':
    movie_1987 = movie_data.title[movie_data['year']=='1987'].tolist()
    movie_1987.insert(0,'-')
    movie_1987 = st.selectbox('Select Movie Title ', movie_1987)
    if movie_1987 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1987)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1987,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1987)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1986':
    movie_1986 = movie_data.title[movie_data['year']=='1986'].tolist()
    movie_1986.insert(0,'-')
    movie_1986 = st.selectbox('Select Movie Title ', movie_1986)
    if movie_1986 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1986)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1986,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1986)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1985':
    movie_1985 = movie_data.title[movie_data['year']=='1985'].tolist()
    movie_1985.insert(0,'-')
    movie_1985 = st.selectbox('Select Movie Title ', movie_1985)
    if movie_1985 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1985)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1985,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1985)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1984':
    movie_1984 = movie_data.title[movie_data['year']=='1984'].tolist()
    movie_1984.insert(0,'-')
    movie_1984 = st.selectbox('Select Movie Title ', movie_1984)
    if movie_1984 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1984)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1984,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1984)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1983':
    movie_1983 = movie_data.title[movie_data['year']=='1983'].tolist()
    movie_1983.insert(0,'-')
    movie_1983 = st.selectbox('Select Movie Title ', movie_1983)
    if movie_1983 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1983)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1983,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1983)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1982':
    movie_1982 = movie_data.title[movie_data['year']=='1982'].tolist()
    movie_1982.insert(0,'-')
    movie_1982 = st.selectbox('Select Movie Title ', movie_1982)
    if movie_1982 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1982)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1982,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1982)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1981':
    movie_1981 = movie_data.title[movie_data['year']=='1981'].tolist()
    movie_1981.insert(0,'-')
    movie_1981 = st.selectbox('Select Movie Title ', movie_1981)
    if movie_1981 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1981)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1981,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1981)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1980':
    movie_1980 = movie_data.title[movie_data['year']=='1980'].tolist()
    movie_1980.insert(0,'-')
    movie_1980 = st.selectbox('Select Movie Title ', movie_1980)
    if movie_1980 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1980)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1980,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1980)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1979':
    movie_1979 = movie_data.title[movie_data['year']=='1979'].tolist()
    movie_1979.insert(0,'-')
    movie_1979 = st.selectbox('Select Movie Title ', movie_1979)
    if movie_1979 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1979)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1979,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1979)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1978':
    movie_1978 = movie_data.title[movie_data['year']=='1978'].tolist()
    movie_1978.insert(0,'-')
    movie_1978 = st.selectbox('Select Movie Title ', movie_1978)
    if movie_1978 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1978)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1978,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1978)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1977':
    movie_1977 = movie_data.title[movie_data['year']=='1977'].tolist()
    movie_1977.insert(0,'-')
    movie_1977 = st.selectbox('Select Movie Title ', movie_1977)
    if movie_1977 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1977)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1977,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1977)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1976':
    movie_1976 = movie_data.title[movie_data['year']=='1976'].tolist()
    movie_1976.insert(0,'-')
    movie_1976 = st.selectbox('Select Movie Title ', movie_1976)
    if movie_1976 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1976)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1976,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1976)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1975':
    movie_1975 = movie_data.title[movie_data['year']=='1975'].tolist()
    movie_1975.insert(0,'-')
    movie_1975 = st.selectbox('Select Movie Title ', movie_1975)
    if movie_1975 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1975)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1975,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1975)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1974':
    movie_1974 = movie_data.title[movie_data['year']=='1974'].tolist()
    movie_1974.insert(0,'-')
    movie_1974 = st.selectbox('Select Movie Title ', movie_1974)
    if movie_1974 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1974)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1974,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1974)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1973':
    movie_1973 = movie_data.title[movie_data['year']=='1973'].tolist()
    movie_1973.insert(0,'-')
    movie_1973 = st.selectbox('Select Movie Title ', movie_1973)
    if movie_1973 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1973)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1973,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1973)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1972':
    movie_1972 = movie_data.title[movie_data['year']=='1972'].tolist()
    movie_1972.insert(0,'-')
    movie_1972 = st.selectbox('Select Movie Title ', movie_1972)
    if movie_1972 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1972)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1972,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1972)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1971':
    movie_1971 = movie_data.title[movie_data['year']=='1971'].tolist()
    movie_1971.insert(0,'-')
    movie_1971 = st.selectbox('Select Movie Title ', movie_1971)
    if movie_1971 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1971)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1971,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1971)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1970':
    movie_1970 = movie_data.title[movie_data['year']=='1970'].tolist()
    movie_1970.insert(0,'-')
    movie_1970 = st.selectbox('Select Movie Title ', movie_1970)
    if movie_1970 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1970)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1970,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1970)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1969':
    movie_1969 = movie_data.title[movie_data['year']=='1969'].tolist()
    movie_1969.insert(0,'-')
    movie_1969 = st.selectbox('Select Movie Title ', movie_1969)
    if movie_1969 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1969)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1969,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1969)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1968':
    movie_1968 = movie_data.title[movie_data['year']=='1968'].tolist()
    movie_1968.insert(0,'-')
    movie_1968 = st.selectbox('Select Movie Title ', movie_1968)
    if movie_1968 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1968)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1968,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1968)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1967':
    movie_1967 = movie_data.title[movie_data['year']=='1967'].tolist()
    movie_1967.insert(0,'-')
    movie_1967 = st.selectbox('Select Movie Title ', movie_1967)
    if movie_1967 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1967)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1967,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1967)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1966':
    movie_1966 = movie_data.title[movie_data['year']=='1966'].tolist()
    movie_1966.insert(0,'-')
    movie_1966 = st.selectbox('Select Movie Title ', movie_1966)
    if movie_1966 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1966)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1966,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1966)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1965':
    movie_1965 = movie_data.title[movie_data['year']=='1965'].tolist()
    movie_1965.insert(0,'-')
    movie_1965 = st.selectbox('Select Movie Title ', movie_1965)
    if movie_1965 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1965)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1965,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1965)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1964':
    movie_1964 = movie_data.title[movie_data['year']=='1964'].tolist()
    movie_1964.insert(0,'-')
    movie_1964 = st.selectbox('Select Movie Title ', movie_1964)
    if movie_1964 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1964)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1964,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1964)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1963':
    movie_1963 = movie_data.title[movie_data['year']=='1963'].tolist()
    movie_1963.insert(0,'-')
    movie_1963 = st.selectbox('Select Movie Title ', movie_1963)
    if movie_1963 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1963)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1963,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1963)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1962':
    movie_1962 = movie_data.title[movie_data['year']=='1962'].tolist()
    movie_1962.insert(0,'-')
    movie_1962 = st.selectbox('Select Movie Title ', movie_1962)
    if movie_1962 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1962)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1962,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1962)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1961':
    movie_1961 = movie_data.title[movie_data['year']=='1961'].tolist()
    movie_1961.insert(0,'-')
    movie_1961 = st.selectbox('Select Movie Title ', movie_1961)
    if movie_1961 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1961)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1961,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1961)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1960':
    movie_1960 = movie_data.title[movie_data['year']=='1960'].tolist()
    movie_1960.insert(0,'-')
    movie_1960 = st.selectbox('Select Movie Title ', movie_1960)
    if movie_1960 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1960)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1960,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1960)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1959':
    movie_1959 = movie_data.title[movie_data['year']=='1959'].tolist()
    movie_1959.insert(0,'-')
    movie_1959 = st.selectbox('Select Movie Title ', movie_1959)
    if movie_1959 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1959)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1959,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1959)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1958':
    movie_1958 = movie_data.title[movie_data['year']=='1958'].tolist()
    movie_1958.insert(0,'-')
    movie_1958 = st.selectbox('Select Movie Title ', movie_1958)
    if movie_1958 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1958)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1958,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1958)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1957':
    movie_1957 = movie_data.title[movie_data['year']=='1957'].tolist()
    movie_1957.insert(0,'-')
    movie_1957 = st.selectbox('Select Movie Title ', movie_1957)
    if movie_1957 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1957)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1957,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1957)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1956':
    movie_1956 = movie_data.title[movie_data['year']=='1956'].tolist()
    movie_1956.insert(0,'-')
    movie_1956 = st.selectbox('Select Movie Title ', movie_1956)
    if movie_1956 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1956)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1956,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1956)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1955':
    movie_1955 = movie_data.title[movie_data['year']=='1955'].tolist()
    movie_1955.insert(0,'-')
    movie_1955 = st.selectbox('Select Movie Title ', movie_1955)
    if movie_1955 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1955)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1955,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1955)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1954':
    movie_1954 = movie_data.title[movie_data['year']=='1954'].tolist()
    movie_1954.insert(0,'-')
    movie_1954 = st.selectbox('Select Movie Title ', movie_1954)
    if movie_1954 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1954)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1954,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1954)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1953':
    movie_1953 = movie_data.title[movie_data['year']=='1953'].tolist()
    movie_1953.insert(0,'-')
    movie_1953 = st.selectbox('Select Movie Title ', movie_1953)
    if movie_1953 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1953)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1953,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1953)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1952':
    movie_1952 = movie_data.title[movie_data['year']=='1952'].tolist()
    movie_1952.insert(0,'-')
    movie_1952 = st.selectbox('Select Movie Title ', movie_1952)
    if movie_1952 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1952)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1952,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1952)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1951':
    movie_1951 = movie_data.title[movie_data['year']=='1951'].tolist()
    movie_1951.insert(0,'-')
    movie_1951 = st.selectbox('Select Movie Title ', movie_1951)
    if movie_1951 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1951)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1951,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1951)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1950':
    movie_1950 = movie_data.title[movie_data['year']=='1950'].tolist()
    movie_1950.insert(0,'-')
    movie_1950 = st.selectbox('Select Movie Title ', movie_1950)
    if movie_1950 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1950)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1950,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1950)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1949':
    movie_1949 = movie_data.title[movie_data['year']=='1949'].tolist()
    movie_1949.insert(0,'-')
    movie_1949 = st.selectbox('Select Movie Title ', movie_1949)
    if movie_1949 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1949)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1949,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1949)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1948':
    movie_1948 = movie_data.title[movie_data['year']=='1948'].tolist()
    movie_1948.insert(0,'-')
    movie_1948 = st.selectbox('Select Movie Title ', movie_1948)
    if movie_1948 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1948)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1948,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1948)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1947':
    movie_1947 = movie_data.title[movie_data['year']=='1947'].tolist()
    movie_1947.insert(0,'-')
    movie_1947 = st.selectbox('Select Movie Title ', movie_1947)
    if movie_1947 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1947)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1947,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1947)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1946':
    movie_1946 = movie_data.title[movie_data['year']=='1946'].tolist()
    movie_1946.insert(0,'-')
    movie_1946 = st.selectbox('Select Movie Title ', movie_1946)
    if movie_1946 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1946)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1946,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1946)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1945':
    movie_1945 = movie_data.title[movie_data['year']=='1945'].tolist()
    movie_1945.insert(0,'-')
    movie_1945 = st.selectbox('Select Movie Title ', movie_1945)
    if movie_1945 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1945)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1945,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1945)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1944':
    movie_1944 = movie_data.title[movie_data['year']=='1944'].tolist()
    movie_1944.insert(0,'-')
    movie_1944 = st.selectbox('Select Movie Title ', movie_1944)
    if movie_1944 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1944)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1944,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1944)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1943':
    movie_1943 = movie_data.title[movie_data['year']=='1943'].tolist()
    movie_1943.insert(0,'-')
    movie_1943 = st.selectbox('Select Movie Title ', movie_1943)
    if movie_1943 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1943)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1943,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1943)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1942':
    movie_1942 = movie_data.title[movie_data['year']=='1942'].tolist()
    movie_1942.insert(0,'-')
    movie_1942 = st.selectbox('Select Movie Title ', movie_1942)
    if movie_1942 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1942)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1942,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1942)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1941':
    movie_1941 = movie_data.title[movie_data['year']=='1941'].tolist()
    movie_1941.insert(0,'-')
    movie_1941 = st.selectbox('Select Movie Title ', movie_1941)
    if movie_1941 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1941)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1941,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1941)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1940':
    movie_1940 = movie_data.title[movie_data['year']=='1940'].tolist()
    movie_1940.insert(0,'-')
    movie_1940 = st.selectbox('Select Movie Title ', movie_1940)
    if movie_1940 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1940)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1940,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1940)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1939':
    movie_1939 = movie_data.title[movie_data['year']=='1939'].tolist()
    movie_1939.insert(0,'-')
    movie_1939 = st.selectbox('Select Movie Title ', movie_1939)
    if movie_1939 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1939)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1939,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1939)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1938':
    movie_1938 = movie_data.title[movie_data['year']=='1938'].tolist()
    movie_1938.insert(0,'-')
    movie_1938 = st.selectbox('Select Movie Title ', movie_1938)
    if movie_1938 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1938)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1938,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1938)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1937':
    movie_1937 = movie_data.title[movie_data['year']=='1937'].tolist()
    movie_1937.insert(0,'-')
    movie_1937 = st.selectbox('Select Movie Title ', movie_1937)
    if movie_1937 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1937)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1937,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1937)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1936':
    movie_1936 = movie_data.title[movie_data['year']=='1936'].tolist()
    movie_1936.insert(0,'-')
    movie_1936 = st.selectbox('Select Movie Title ', movie_1936)
    if movie_1936 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1936)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1936,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1936)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1935':
    movie_1935 = movie_data.title[movie_data['year']=='1935'].tolist()
    movie_1935.insert(0,'-')
    movie_1935 = st.selectbox('Select Movie Title ', movie_1935)
    if movie_1935 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1935)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1935,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1935)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1934':
    movie_1934 = movie_data.title[movie_data['year']=='1934'].tolist()
    movie_1934.insert(0,'-')
    movie_1934 = st.selectbox('Select Movie Title ', movie_1934)
    if movie_1934 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1934)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1934,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1934)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1933':
    movie_1933 = movie_data.title[movie_data['year']=='1933'].tolist()
    movie_1933.insert(0,'-')
    movie_1933 = st.selectbox('Select Movie Title ', movie_1933)
    if movie_1933 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1933)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1933,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1933)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1932':
    movie_1932 = movie_data.title[movie_data['year']=='1932'].tolist()
    movie_1932.insert(0,'-')
    movie_1932 = st.selectbox('Select Movie Title ', movie_1932)
    if movie_1932 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1932)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1932,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1932)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1931':
    movie_1931 = movie_data.title[movie_data['year']=='1931'].tolist()
    movie_1931.insert(0,'-')
    movie_1931 = st.selectbox('Select Movie Title ', movie_1931)
    if movie_1931 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1931)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1931,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1931)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1930':
    movie_1930 = movie_data.title[movie_data['year']=='1930'].tolist()
    movie_1930.insert(0,'-')
    movie_1930 = st.selectbox('Select Movie Title ', movie_1930)
    if movie_1930 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1930)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1930,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1930)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1929':
    movie_1929 = movie_data.title[movie_data['year']=='1929'].tolist()
    movie_1929.insert(0,'-')
    movie_1929 = st.selectbox('Select Movie Title ', movie_1929)
    if movie_1929 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1929)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1929,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1929)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1928':
    movie_1928 = movie_data.title[movie_data['year']=='1928'].tolist()
    movie_1928.insert(0,'-')
    movie_1928 = st.selectbox('Select Movie Title ', movie_1928)
    if movie_1928 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1928)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1928,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1928)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1927':
    movie_1927 = movie_data.title[movie_data['year']=='1927'].tolist()
    movie_1927.insert(0,'-')
    movie_1927 = st.selectbox('Select Movie Title ', movie_1927)
    if movie_1927 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1927)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1927,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1927)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1926':
    movie_1926 = movie_data.title[movie_data['year']=='1926'].tolist()
    movie_1926.insert(0,'-')
    movie_1926 = st.selectbox('Select Movie Title ', movie_1926)
    if movie_1926 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1926)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1926,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1926)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1925':
    movie_1925 = movie_data.title[movie_data['year']=='1925'].tolist()
    movie_1925.insert(0,'-')
    movie_1925 = st.selectbox('Select Movie Title ', movie_1925)
    if movie_1925 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1925)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1925,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1925)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1924':
    movie_1924 = movie_data.title[movie_data['year']=='1924'].tolist()
    movie_1924.insert(0,'-')
    movie_1924 = st.selectbox('Select Movie Title ', movie_1924)
    if movie_1924 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1924)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1924,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1924)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1923':
    movie_1923 = movie_data.title[movie_data['year']=='1923'].tolist()
    movie_1923.insert(0,'-')
    movie_1923 = st.selectbox('Select Movie Title ', movie_1923)
    if movie_1923 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1923)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1923,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1923)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1922':
    movie_1922 = movie_data.title[movie_data['year']=='1922'].tolist()
    movie_1922.insert(0,'-')
    movie_1922 = st.selectbox('Select Movie Title ', movie_1922)
    if movie_1922 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1922)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1922,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1922)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1921':
    movie_1921 = movie_data.title[movie_data['year']=='1921'].tolist()
    movie_1921.insert(0,'-')
    movie_1921 = st.selectbox('Select Movie Title ', movie_1921)
    if movie_1921 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1921)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1921,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1921)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1920':
    movie_1920 = movie_data.title[movie_data['year']=='1920'].tolist()
    movie_1920.insert(0,'-')
    movie_1920 = st.selectbox('Select Movie Title ', movie_1920)
    if movie_1920 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1920)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1920,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1920)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar == '1919':
    movie_1919 = movie_data.title[movie_data['year']=='1919'].tolist()
    movie_1919.insert(0,'-')
    movie_1919 = st.selectbox('Select Movie Title ', movie_1919)
    if movie_1919 =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_1919)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_1919,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_1919)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))

            
if sidebar =='Action':
    movie_action = movie_data.title[movie_data['genre'].str.contains('Action')].tolist()
    movie_action.insert(0,'-')
    movie_action = st.selectbox('Select Movie Title ', movie_action)
    if movie_action =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_action)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_action,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_action)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))

        
if sidebar =='Adventure':
    movie_adventure = movie_data.title[movie_data['genre'].str.contains('Adventure')].tolist()
    movie_adventure.insert(0,'-')
    movie_adventure = st.selectbox('Select Movie Title ', movie_adventure)
    if movie_adventure =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_adventure)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_adventure,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_adventure)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))

    
if sidebar =='Children':
    movie_children = movie_data.title[movie_data['genre'].str.contains('Children')].tolist()
    movie_children.insert(0,'-')
    movie_children = st.selectbox('Select Movie Title ', movie_children)
    if movie_children =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_children)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_children,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_children)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Comedy':
    movie_comedy = movie_data.title[movie_data['genre'].str.contains('Comedy')].tolist()
    movie_comedy.insert(0,'-')
    movie_comedy = st.selectbox('Select Movie Title ', movie_comedy)
    if movie_comedy =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_comedy)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_comedy,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_comedy)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))

        
if sidebar =='Crime':
    movie_crime = movie_data.title[movie_data['genre'].str.contains('Crime')].tolist()
    movie_crime.insert(0,'-')
    movie_crime = st.selectbox('Select Movie Title ', movie_crime)
    if movie_crime =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_crime)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_crime,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_crime)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Documentary':
    movie_document = movie_data.title[movie_data['genre'].str.contains('Documentary')].tolist()
    movie_document.insert(0,'-')
    movie_document = st.selectbox('Select Movie Title ', movie_document)
    if movie_document =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_document)
        st.write('Movie Details')     
        st.table(movie_data.loc[movie_data['title']==movie_document,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_document)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Drama':
    movie_drama = movie_data.title[movie_data['genre'].str.contains('Drama')].tolist()
    movie_drama.insert(0,'-')
    movie_drama = st.selectbox('Select Movie Title ', movie_drama)
    if movie_drama =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_drama)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_drama,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_drama)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Fantasy':
    movie_fantasy = movie_data.title[movie_data['genre'].str.contains('Fantasy')].tolist()
    movie_fantasy.insert(0,'-')
    movie_fantasy = st.selectbox('Select Movie Title ', movie_fantasy)
    if movie_fantasy =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_fantasy)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_fantasy,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_fantasy)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Film-Noir':
    movie_filmnoir = movie_data.title[movie_data['genre'].str.contains('Film-Noir')].tolist()
    movie_filmnoir.insert(0,'-')
    movie_filmnoir = st.selectbox('Select Movie Title ', movie_filmnoir)
    if movie_filmnoir =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_filmnoir)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_filmnoir,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_filmnoir)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Horror':
    movie_horror = movie_data.title[movie_data['genre'].str.contains('Horror')].tolist()
    movie_horror.insert(0,'-')
    movie_horror = st.selectbox('Select Movie Title ', movie_horror)
    if movie_horror =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_horror)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_horror,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_horror)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Musical':
    movie_musical = movie_data.title[movie_data['genre'].str.contains('Musical')].tolist()
    movie_musical.insert(0,'-')
    movie_musical = st.selectbox('Select Movie Title ', movie_musical)
    if movie_musical =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_musical)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_musical,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_musical)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Mystery':
    movie_mystery = movie_data.title[movie_data['genre'].str.contains('Mystery')].tolist()
    movie_mystery.insert(0,'-')
    movie_mystery = st.selectbox('Select Movie Title ', movie_mystery)
    if movie_mystery =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_mystery)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_mystery,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_mystery)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Romance':
    movie_romance = movie_data.title[movie_data['genre'].str.contains('Romance')].tolist()
    movie_romance.insert(0,'-')
    movie_romance = st.selectbox('Select Movie Title ', movie_romance)
    if movie_romance =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_romance)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_romance,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_romance)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Sci-Fi':
    movie_scifi = movie_data.title[movie_data['genre'].str.contains('Sci-Fi')].tolist()
    movie_scifi.insert(0,'-')
    movie_scifi = st.selectbox('Select Movie Title ', movie_scifi)
    if movie_scifi =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_scifi)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_scifi,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_scifi)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Thriller':
    movie_thriller = movie_data.title[movie_data['genre'].str.contains('Thriller')].tolist()
    movie_thriller.insert(0,'-')
    movie_thriller = st.selectbox('Select Movie Title ', movie_thriller)
    if movie_thriller =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_thriller)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_thriller,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_thriller)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='War':
    movie_war = movie_data.title[movie_data['genre'].str.contains('War')].tolist()
    movie_war.insert(0,'-')
    movie_war = st.selectbox('Select Movie Title ', movie_war)
    if movie_war =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_war)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_war,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_war)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))


if sidebar =='Western':
    movie_western = movie_data.title[movie_data['genre'].str.contains('Western')].tolist()
    movie_western.insert(0,'-')
    movie_western = st.selectbox('Select Movie Title ', movie_western)
    if movie_western =='-':
        st.success("No Movies Selected")
        st.write('No Movies Recommended. Please Make a Selection')
    else:
        st.success('You selected: '+ movie_western)
        st.write('Movie Details')
        st.table(movie_data.loc[movie_data['title']==movie_western,['year','genre','average_rating']].assign(hack='').set_index('hack'))
        #Printing the top N similar movies
        movie_id = get_movie_id(movie_western)
        st.write('Top 10 Recommended Movies: ')
        indexes = top_cosine_similarity(sliced, movie_id, top_n)
        st.table(print_similar_movies(movie_data, movie_id, indexes))

             
