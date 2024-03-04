import streamlit as st
import numpy as np
import pandas as pd
import logic
import keras as k
from streamlit_option_menu import option_menu

# settings
page_title='Inspire AI-Link'
page_emoji=':file_folder:'
layout='centered'

#setup the page configuration
st.set_page_config(page_title=page_title,page_icon=page_emoji,layout=layout)
st.title(page_title +" "+ page_emoji)
st.write("An AI Driven Record Linkage App.")
selected= option_menu(
    menu_title=None,
    options=['Siamese','Sanitize','Search'],
    icons=['lightning','eye','search'],
    orientation='horizontal'
)
if 'button' not in st.session_state:
    st.session_state.button = False

def click_button():
    st.session_state.button = not st.session_state.button
#hiding the streamlit widget

hide="""
    <style>
    #MainMenu {visibility:hidden;}
    header {visibility:hidden;}
    footer {visibility:hidden;}
    </style>
"""

st.markdown(hide,unsafe_allow_html=True)

if selected=='Siamese':
    # Check if the DataFrames are already in the session state
    if 'user_uploaded_data1' not in st.session_state:
        # If not, initialize them as None
        st.session_state.user_uploaded_data1 = None
    if 'user_uploaded_data2' not in st.session_state:
        # If not, initialize them as None
        st.session_state.user_uploaded_data2 = None

    with st.container(border=True):
        col1,col2=st.columns(2)
        with col1:
            st.header("HDSS")
            hdss1=st.file_uploader('Upload HDSS data', type={"csv"})
        with col2:
            st.header('Facility')
            facility1=st.file_uploader('Upload Facility data', type={"csv"})
        if hdss1 is not None:
            or_hdss = pd.read_csv(hdss1)
        else:
            or_hdss=pd.read_csv('data/synthetic_hdss_v3.csv')
            # Store the DataFrame in the session state

        if facility1 is not None:
            or_facility=pd.read_csv(facility1)
        else:
            or_facility=pd.read_csv('data/synthetic_facility_v3.csv')
                # Store the DataFrame in the session stat
                
    st.session_state.user_uploaded_data2 = or_facility
    st.session_state.user_uploaded_data1 = or_hdss

    or_hdss=st.session_state.user_uploaded_data1
    or_facility=st.session_state.user_uploaded_data2
    if 'df1' not in st.session_state:
        st.session_state.df1 = None

    if 'df2' not in st.session_state:
        st.session_state.df1 = None

    or_hdss_cop=or_hdss.copy()
    or_facility_cop=or_facility.copy()
    st.session_state.df1 =or_hdss_cop
    st.session_state.df2 =or_facility_cop
     #prepare the data 
    hdss=logic.prepare_data(st.session_state.df1,'hdss')
    facility=logic.prepare_data(st.session_state.df2,'facility')

#perform local hashing
    rand=logic.generate_random(hdss.shape[1])
    hdss_bin=logic.local_hashing(hdss,rand)
    facility_bin=logic.local_hashing(facility,rand)

    #compute similarity
    
    similar,count=logic.compute_similarity(hdss_bin,facility_bin)
    #get similar pairs 
    similar_df=logic.get_candidate_pairs(similar)
    
    if 'candidate' not in st.session_state:
        # If not, initialize them as None
        st.session_state.candidate = None

    #create candidate links
    st.session_state.candidate = similar_df
    #generate featureshdss
    features=logic.get_hdss_and_facility(similar_df,hdss,facility)
    #load the ML model and predict with the features
    if 'result' not in st.session_state:
        # If not, initialize them as None
        st.session_state.result = None
    with st.container(border=True):
        btn1=st.button('Get head',type='primary',on_click=click_button)
        if btn1:
            f5_hdss=or_hdss.head()
            f5_facility=or_facility.head()
        # Display the heads of both DataFrames side by side in columns
            col1, col2 = st.columns(2)
            with col1:
                st.write("HDSS data:")
                st.write(or_hdss.head())
            with col2:
                st.write("Facility data:")
                st.write(or_facility.head())

    
    with st.container(border=True):
        col1,col2,col3=st.columns(3)
        with col2:
            st.header('Predict')
            compare=st.button("Compare and Predict", type="primary",on_click=click_button)


    
    if 'outmatch' not in st.session_state:
        # If not, initialize them as None
        st.session_state.outmatch = None
    if 'out_tent' not in st.session_state:
        # If not, initialize them as None
        st.session_state.out_tent = None
    if 'out_unmatch' not in st.session_state:
    # If not, initialize them as None
        st.session_state.out_unmatch = None

    with st.container(border=True):
        col1,col2=st.columns(2)
        with col1:
            st.image('images/ai.jpg')
        with col2:
            if compare:
                if st.session_state.result is not None:
                    result= st.session_state.result
                else:
                    result = logic.predict(features)
                    st.session_state.result=result
                
                # if st.session_state.result is not None:
                out_match=sum(result==0)
                out_tent=sum(result==1)
                out_unmatch=sum(result==2)

                st.session_state.out_unmatch=out_unmatch
                st.session_state.out_tent=out_tent
                st.session_state.out_match=out_match

            if st.session_state.out_unmatch is not None:     
                st.metric(label='The Number of perfect Matches'+" " +':dart:',value=st.session_state.out_match)
                st.metric(label='The Number of partial Matches'+" "+':question:',value=st.session_state.out_tent)
                st.metric(label='The Number of UnMatches'+" "+':no_entry_sign:',value=st.session_state.out_unmatch)
            else:
                st.info('The results will appear here')  

    if 'sanit' not in st.session_state:
        st.session_state.sanit=None
    if st.session_state.result is not None:
        sanit=similar_df.iloc[st.session_state.result==1]
        simi=similar_df.iloc[st.session_state.result==0]
        diff= similar_df.iloc[st.session_state.result==2]
        st.session_state.sanit=sanit

    #
    with st.container(border=True):
        st.write('View sample records randomly')
        option = st.selectbox('Choose the category to view',('Match','Partial Match','Distinct'),placeholder="Choose")
        
        btn2=st.button('View',type='primary',on_click=click_button)
        col1,col2=st.columns(2)
        if btn2:
            if option=='Match':
                rand=np.random.randint(simi.shape[0])
                row=np.array(simi.iloc[rand])
                # if pre:
                #     rand=np.random.randint(simi.shape[0])
                #     row=np.array(simi.iloc[rand])
                
                st.write('TheseRecords Match')
                with col1:
                    st.write(or_hdss_cop.iloc[[row[0]]])
                with col2:
                    st.write(or_facility_cop.iloc[[row[1]]])
            elif option=='Partial Match':
                rand=np.random.randint(sanit.shape[0])
                row=np.array(sanit.iloc[rand])
                # if pre:
                #     rand=np.random.randint(simi.shape[0])
                #     row=np.array(sanit.iloc[rand])
                st.write('These Records May Match')
                with col1:
                    st.write(or_hdss_cop.iloc[[row[0]]])
                with col2:
                    st.write(or_facility_cop.iloc[[row[1]]])
            else:
                rand=np.random.randint(diff.shape[0])
                row=np.array(diff.iloc[rand])
                # if pre:
                #     rand=np.random.randint(simi.shape[0])
                #     row=np.array(diff.iloc[rand])
                st.write('These Records Do Not Match')
                with col1:
                    st.write(or_hdss_cop.iloc[[row[0]]])
                with col2:
                    st.write(or_facility_cop.iloc[[row[1]]])
    if 'merged' not in st.session_state:
        st.session_state.merged = None
    col1,col2,col3=st.columns(3)
    with st.container(border=True):
        with col2:
            st.write('Merge the Perfect Matches-')
            merge=st.button('Merge',type='primary',on_click=click_button)
            if merge:
                merged_df=logic.merge_data(simi,st.session_state.df1,st.session_state.df2,add_cols=True,ask='merge')
                st.session_state.merged=merged_df
                st.write(f'Successfully merged the Matching entities. The New dataframe is of size {merged_df.shape[0]}')
        

if selected=='Sanitize':
    st.info("Sanitize the partially matching records.")
    #read the merged df from the previous page

    or_facility_cop=st.session_state.df2
    if st.session_state.merged is not None:
        merged_df=st.session_state.merged
        st.write(merged_df.head())
    else:
        merged_df=None
        st.info('The merged df appears')

    #create two sides one for hdss and the other for facility
    if st.session_state.sanit is not None:
        sanitiz=st.session_state.sanit
        tent=sanitiz.shape[0]
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        prev=st.button('Prev',type='primary')
    with col2:
        next=st.button('Next',type='primary')
    with col4:
        Add=st.button('Merge',type='primary')
    
    #create a state:

    if 'pos' not in st.session_state:
        st.session_state.pos=0
    
    with st.container(border=True):
        st.write(f'You have {tent} files to sanitize')

        if next:
            ind=st.session_state.pos
            ind=ind+1
            st.session_state.pos=ind
        if prev:
            ind=st.session_state.pos
            ind=ind-1
            st.session_state.pos=ind
            # if i<0:
            #     
            #     i=st.session_state.pos
            # else:
            #     st.write('You have reached the beginning')
        if Add:
            #update the merged df
            merged_df=logic.merge_data(sanitiz.iloc[[i]],merged_df,or_facility_cop,add_cols=False,ask='sanitize')
            st.session_state.merged=merged_df
            sanitiz.remove(sanitiz[i])
            st.info('Added successfuly')
        i=st.session_state.pos
        if i<tent and i>0:
            tu=sanitiz.iloc[i].values
        else:
            tu=sanitiz.iloc[0].values
            st.info('Out of limit')
        col1,col2=st.columns(2)
        with col1:
            df1=st.session_state.df1
            if df1 is not None:
                st.write(df1.iloc[[tu[0]]])
        with col2:
            df2=st.session_state.df2
            if df2 is not None:
                st.write(df2.iloc[[tu[1]]])


    col1,col2,col3=st.columns(3)
    with st.container(border=True):
        st.info('The key to the labels')
        st.info('0-> match, 1-> partial match and 2-> not matching')
        with col1:
            if merged_df is not None:
                m_d=st.session_state.merged
                csv=m_d.to_csv(index=False).encode("utf-8")
                st.download_button(
                "Download the Final data",
                csv,
            "master_merged.csv",
            "text/csv"
                )
            else:
                st.info('The merged dataframe will appear hear for download!')
        with col3:
            
            if st.session_state.candidate is not None:
                prs=st.session_state.candidate
                prs['labels']=st.session_state.result
                csv=prs.to_csv(index=False).encode("utf-8")
                st.download_button(
                "Download Pairs",
                csv,
            "pairs.csv",
            "text/csv"
                )
                
            else:
               st.info('The pairs will appear here for download') 

#################################### SEARCH ######################################
if selected=='Search':
   
    st.info('This section allows you to search for a patient using their hdss record number :sunglasses:')
    rec_number=st.number_input(label='Enter the Hdss Record number',format='%i',min_value=0)
    
    #search column
    column_name='INDEX'
    if st.session_state.merged is not None:
        merged_df=st.session_state.merged
        st.write(merged_df[merged_df[column_name]==int(rec_number)])
    else:
        st.info('The results of your search will appear here')


