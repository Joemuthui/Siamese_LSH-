import numpy as np
import pandas as pd
from recordlinkage.preprocessing import clean
from recordlinkage.preprocessing import phonetic
import streamlit as st
import recordlinkage
import datetime as dt
import pickle

st.cache
def read_csv(file):
    return pd.read_csv(file)

#year, month, and day column
def connvert_date(hdss):
    hdss['dob']=pd.to_datetime(hdss['dob'],dayfirst=True)
    hdss['year']=hdss['dob'].dt.year/1970
    hdss['month']=hdss['dob'].dt.month/12
    hdss['day']=hdss['dob'].dt.day/31
    return hdss.drop('dob',axis=1)

def generate_random(n):
    np.random.seed(0)
    rand=np.random.randn(n,150)
    return rand

#create a function that can be used to convert names to vectors
def convert_name_to_vector(name):
    size=15
    code=list('abcdefghijklmnopqrstuvwxyz')
    name=name.lower()
    initials=np.zeros(size)
    i=0
    for a in name:
        if a in code:
            value=code.index(a)+1
            initials[i]=value
            i+=1
    return (initials/26).flatten()

#drop the unnecessary columns
def prepare_data(data,name):
    #convert the dates first
    or_hdss=connvert_date(data)
    or_hdss.replace(np.NaN,'',inplace=True)
    #generate the column names for the vectors
    first=[]
    last=[]
    pet=[]
    for i in range(15):
        first.append(f'f_{i}')
        last.append(f'l_{i}')
        pet.append(f'p_{i}')
    #convert the names to vectors. Initiliazed vector of size 15
    or_hdss[first]=pd.DataFrame(or_hdss['firstname'].apply(convert_name_to_vector).tolist())
    or_hdss[last]=pd.DataFrame(or_hdss['lastname'].apply(convert_name_to_vector).tolist())
    or_hdss[pet]=pd.DataFrame(or_hdss['petname'].apply(convert_name_to_vector).tolist())
    if name=='hdss':
        or_hdss=or_hdss.drop(['recnr','firstname', 'lastname', 'petname', 'hdssid','hdsshhid','nationalid'],axis=1)
    else:
        or_hdss=or_hdss.drop(['recnr', 'firstname', 'lastname', 'petname','nationalid','patientid', 'visitdate',],axis=1)
    return or_hdss


#perform local hashinb
def local_hashing(data,random_matrix):
    prod=np.matmul(data.values,random_matrix)
    binary=np.where(prod>0,1,0)
    return binary

def get_index(vector,facility):
    equate=list(np.sum(facility==np.array(vector),axis=1)>=148)
    indices=np.where(equate)[0]
    return indices
def compute_similarity(hd_factor,fac_facilty):
    similar={}
    count=0
    for i in range(hd_factor.shape[0]):
        vector=hd_factor[i]
        ind=get_index(vector,fac_facilty)
        if len(ind)>0:
            count+=len(ind)
            similar[i]=ind
    return similar,count
def get_candidate_pairs(similar):
    k=[]
    v=[]
    keys=list(similar.keys())
    values=list(similar.values())
    for i in range(len(keys)):
        key=keys[i]
        val=values[i]
        for j in range(len(val)):
            k.append(key)
            v.append(val[j])
    pairs=pd.DataFrame(np.array([k,v]).T)
    return pairs

def get_hdss_and_facility(candidate_links,hdss,facility):
    hd,fc=candidate_links[0],candidate_links[1]
    #take back the values by of year,month,day to original va;ues
    hdss['year']=hdss['year']*1970
    hdss['month']=hdss['month']*12
    hdss['day']=hdss['day']*31

    facility['year']=facility['year']*1970
    facility['month']=facility['month']*12
    facility['day']=facility['day']*31

    hdss=hdss.iloc[hd].values
    facility=facility.iloc[fc].values
    return [hdss,facility]

@st.cache_data
def predict(features):
   # Load the trained classifier from the file
    with open('model/saimese.pkl', 'rb') as file:
        classifier = pickle.load(file)
    return  np.argmax(classifier.predict(features),axis=1)


from streamlit.components.v1 import html

def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)

def merge_data(links,or_hdss,or_facility,add_cols=True,ask='merge'):
    if ask:
        or_hdss['index']=or_hdss.index
        new_hdss=or_hdss.set_index('index')
    else:
        new_hdss=or_hdss
    #Additional columns
    if add_cols is True:
        add_cols=['patientid','visitdate']
        new_hdss[add_cols]=np.NAN  
    else:
        pass

    h=links.values[:,0]
    f=links.values[:,1]
    new_match=pd.DataFrame(or_facility.iloc[f].values,index=h,columns=or_facility.columns)
    df=pd.concat([new_hdss,new_match])
    df['INDEX']=df.index
    return df