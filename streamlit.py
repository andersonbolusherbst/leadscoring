import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

st.title('Lead Scoring')
### gif from url
st.markdown("![Alt Text](https://thumbs.gfycat.com/FastMiserlyJellyfish.webp)")

st.header('The problem')
st.write("""An organization that offers a hiring assessment platform
 is looking to reduce it's annual marketing spending""")
st.header('Approach')
st.write("""Build a sophisticated **Machine Learning model that predicts 
the  percentage probability** of marketing leads purchasing their product.This allows the company to be more precise with their marketing by targeting high probability customers.  """)
st.write("**Notes**:")
st.write("""\n 1. The data is from a Kaggle machine learning competition and anyone can download it: https://www.kaggle.com/parv619/hackerearths-reduce-marketing-waste
\n 2. This company has already aggregated a lot of data to reach this point that's why they have a success probability already,if your company has similar data 
we are able to create a similar metric based on your dataset before employing the machine learning algorithm .""")
st.header('Sample of the data')

#reading in the data
@st.cache(allow_output_mutation=True)
def data_reader(path_full,path_clean):
    df = pd.read_csv(path_full)
    clean_df = pd.read_csv(path_clean,index_col=0)
    return df, clean_df 
df, clean_df = data_reader('train.csv','cleaned_data.csv')    

st.dataframe(df.head(5))

st.write(""" This is a snapshot of the clients in the dataset. The goal is to predict is the Success probability. This was provided in the dataset and goal is to be able to accurately
            predict this percentage""")

st.header("Machine learning")
st.write("""After extensive data cleaning and exploratory data analysis, the columns we included in our model were: 
        **deal value, product pitched,lead revenue, fund category, country, state, hiring candidate role, lead source, internal point of contact, date of creation** """)


filename = 'grad_boost.sav'
loaded_model = pickle.load(open(filename, 'rb'))


st.header('What features were most important for our model? ')
feature_importance = pd.read_csv('feature_importance.csv',index_col=0)


plot_imp = feature_importance.sort_values(by = 'Feature Importance',ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(data = plot_imp, x ='Feature Importance', y = plot_imp.index )
ax.set_yticklabels(['India','USA','Internal Rating','Level 3 Meeting','Fund Category 1','Lead Source: Contact Email','Meeting level 1','Deal Value','Lead revenue 500Mil to 1 Bil','Lead Source Email'])
st.pyplot(fig)

st.write("""Feature importances is simply showing the extent to which the model used each feature. These features are very specific to this business, so yours will look different. This information could be useful
             to a business when deciding how to act on the infomration from the model.  """)



st.header('Random Prediction')
st.write("This has randomly selected one of the customers, now let's see how well the model does")


unclean_df = df[df.index.isin(clean_df.index)]

clean_df.reset_index(inplace=True,drop=True)
unclean_df.reset_index(inplace=True,drop=True)
chosen_one = unclean_df.sample(1)

chosen_one_index = chosen_one.index
 

st.dataframe(chosen_one)
actual =chosen_one.Success_probability


c1,c2,c3= st.columns((1,1,1))
with c1:
    st.write('The answer we hope to get:🤞',actual)



cat_cols = ['Industry',
'Pitch',
'Lead_revenue',
'Fund_category',
'Location',
'Geography',
'Hiring_candidate_role',
'Lead_source',
'Level_of_meeting',
'Internal_POC']



# columns transfomer and pipeline
ct = ColumnTransformer(
        [
        ("cat_transformer", OneHotEncoder(sparse=False),cat_cols),
        ("num_transformer", StandardScaler(),['Deal_value'])
        ]
        ,remainder = 'passthrough')

all_prepped = ct.fit_transform(clean_df)
chosen_prepped = all_prepped[chosen_one_index]

prediction = loaded_model.predict(chosen_prepped)
with c2:
    st.write('Prediction:',prediction)
with c3:
    st.write('Error: ',abs(prediction - actual.values))
    st.write('This should be quite small, on average our model only had an error of 0.22  on unseen data')
st.write('**Refresh the page for a new customer**')
st.header('What next ?')
st.write("""In this case the model gives us a success probability and we would now use our model on 
new clients where we have not already calculated the success probability and this should save the company money on marketing 
and also give them some more insights into their product.
""")
