import pandas as kl
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import random

data=kl.read_csv("chatbot_dataset.csv")
# print(data)

# preprocess the data
nltk.download('punkt')
data['Question']=data['Question'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))

#split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data['Question'],data['Answer'],test_size=0.2,random_state=42)

# Create a model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(x_train,y_train)
print("model training complete")

#Function to get a response 

def get_response(question):
    question=' '.join(nltk.word_tokenize(question.lower()))
    answer=model.predict([question])[0]
    return answer


# Initialize the Dash app
app = dash.Dash(__name__)
"""
# Create the navigation bar
navbar = html.Nav(
    children=[
        dcc.Link('Home', href='/', className='nav-link'),
        dcc.Link('About', href='/about', className='nav-link'),
        dcc.Link('Contact', href='/contact', className='nav-link'),
    ],
    className='navbar'
)

# Define the layout with the navigation bar and a container for page content
app.layout = html.Div([
    navbar,
    html.Div(id='page-content')
])

# Define callback to update the page content based on the URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/about':
        return html.Div([
            html.H2('About Us'),
            html.P('This is the About page.')
        ])
    elif pathname == '/contact':
        return html.Div([
            html.H2('Contact Us'),
            html.P('This is the Contact page.')
        ])
    else:
        return html.Div([
            html.H2('Welcome to the Home Page'),
            html.P('This is the home page.')
        ])
"""
# Define the layout
app.layout = html.Div([
   html.H1("Chatbot", style={'textAlign': 'left','color':'black',}),
    dcc.Textarea(
        id='user-input',
        value="Any Questions...",
        style={'width': '100%', 'height': 100}
    ),
    
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='chatbot-output', style={'padding': '10px'})
], style={
    'backgroundColor': '#f9e79f', 
    'padding': '20px', 
    'borderRadius': '10px', 
    'maxWidth': '500px', 
    'margin': 'auto'
})


# Define callback to update chatbot response
@app.callback(
    Output('chatbot-output', 'children'),
    Input('submit-button', 'n_clicks'),
    [dash.dependencies.State('user-input', 'value')]
)
def update_output(n_clicks, user_input):
    if n_clicks > 0:
        response = get_response(user_input)
        return html.Div([
            html.P(f"You: {user_input}", style={'margin': '10px'}),
            html.P(f"Bot: {response}", style={'margin': '10px', 'backgroundColor': '#f7f9f9 ', 'padding': '10px'})
        ])
    
    return "Ask me something!"

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)

