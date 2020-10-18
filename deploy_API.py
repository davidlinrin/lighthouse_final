from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
api = Api(app)

user_CF_df = pd.read_csv('/mnt/d/lighthouse/Final_data/user_CF_df.csv', header = 0)
user_CF_df.drop('Unnamed: 0', axis =1 ,inplace = True)
CF_memory_pivot = user_CF_df.pivot(index = 'name', columns = 'user', values = 'rating').fillna(0)

KNN_model = pickle.load(open( "KNN_model_game_input.p", "rb" ))
SVD_model = pickle.load(open( "SVD_model.p", "rb" ))

#### Content Based Model, TAKES UP ALOT OF MEMORY AND DISK SPACE. 
# games_index = pd.read_csv('/mnt/d/lighthouse/Final_data/games_content_based.csv', header = 0)
# games_index.index = games_index['steam_appid']
# games_index.drop(['steam_appid'], axis = 1, inplace = True)
# similarities = np.load('similarity_cosine_sim.npy')
# indices = pd.Series(games_index.reset_index().index, index=games_index['name']).drop_duplicates()

# def get_recommendations(title, cosine_sim=similarities, top = 3):
#     # Get the index of the movie that matches the title
#     idx = indices[title]
#     # Get the pairwsie similarity scores of all movies with that movie
#     sim_scores = list(enumerate(similarities[idx]))
#     # Sort the movies based on the similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     # Get the scores of the 10 most similar movies
#     sim_scores = sim_scores[1:top+1]
#     # Get the movie indices
#     game_indices = [i[0] for i in sim_scores]
#     # Return the top 10 most similar movies
#     return games_index['name'].iloc[game_indices]

# class games_alike(Resource):
#     def post(self):
#         json_data = request.get_json()
#         try:
#             game = json_data['game']
#         except KeyError:
#             return('Invalid Game, make sure the game is spelt correctly')
        
#         try:
#             top_n = json_data['top']
#         except KeyError:
#             top_n = 3
        
#         res = get_recommendations(game, top = top_n) 
#         return res

class games_alike(Resource):
    def post(self):
        json_data = request.get_json()
        try:
            game = json_data['game']
            game_index = list(CF_memory_pivot.index).index(game)
        except KeyError:
            return('Invalid Game, make sure the game is spelt correctly')
        
        try:
            top_n = json_data['top'] + 1
        except KeyError:
            top_n = 4
        
        distances, indices = KNN_model.kneighbors(CF_memory_pivot.iloc[game_index,:].values.reshape(1, -1), n_neighbors = top_n)
        result = [CF_memory_pivot.index[indices.flatten()[i]] for i in range(1,len(distances.flatten()))]
        return result 

class user_recommendation(Resource):
    def post(self):
        json_data = request.get_json()
        try:
            ID = json_data['ID']
        except KeyError:
            return('Invalid ID, please try again')
            
        try:
            top_n = json_data['top']
        except KeyError:
            top_n = 3
        
        iid = user_CF_df['appid'].unique()                             # all games
        iid_cur_user = user_CF_df.loc[user_CF_df['user']==ID,'appid']  # games the user owns
        iid_to_pred = np.setdiff1d(iid,iid_cur_user)                   # games the user does not own
        
        testset = [[ID,iid,4] for iid in iid_to_pred]
        
        SVD_predictions = SVD_model.test(testset)
        SVD_pred_rating = np.array([pred.est for pred in SVD_predictions])
        SVD_i_max = SVD_pred_rating.argsort()[-top_n:][::-1]
        
        SVD_game_id = [SVD_predictions[i].iid for i in SVD_i_max]
        
        return [list(user_CF_df.loc[user_CF_df['appid']==i, 'name'].unique())[0] for i in SVD_game_id]

class game_rating(Resource):
    def post(self):
        json_data = request.get_json()
        try:
            ID = json_data['ID']
        except KeyError:
            return('Invalid ID, please try again')
        
        try:
            game = json_data['game_ID']
        except KeyError:
            return('Invalid game_ID, please try again')
            
        rating = SVD_model.predict(ID,game).est + 1
        
        return rating

class ID_to_game(Resource):
    def post(self):
        json_data = request.get_json()
        game_id = json_data['game_ID']
        print(game_id)
        return list(set(user_CF_df[user_CF_df['appid']==game_id]['name']))[0]
    
class game_to_ID(Resource):
    def post(self):
        json_data = request.get_json()
        game_name = json_data['game_name']
        return list(set(user_CF_df[user_CF_df['name']==game_name]['appid']))[0]
    
class game_list(Resource):
    def post(self):
        return list(CF_memory_pivot.index)
    
class user_list(Resource):
    def post(self):
        return list(CF_memory_pivot.columns)

# assign endpoint
# * parameters are optional
api.add_resource(games_alike, '/games_alike')            # returns games similar to the query (game: 'Game Name')
api.add_resource(user_recommendation, '/user_recommend') # returns user recommendation (ID: UserID, top*: number of recommendations)
api.add_resource(game_rating, '/rating')                 # returns rating of the query game based on the user (ID: UserID, game_ID: game ID)
api.add_resource(ID_to_game, '/id_to_game')              # returns name of game
api.add_resource(game_to_ID, '/game_to_id')              # returns ID of game
api.add_resource(game_list, '/games')                    # returns list of games in database
api.add_resource(user_list, '/users')                    # returns list of games in database


if __name__ == '__main__':
    app.run(debug=True)