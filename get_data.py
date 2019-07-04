
import os

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import string

# function to extract article-level information and parse raw text
def get_post(post_url):
    post = requests.get(post_url)
    soup = BeautifulSoup(post.text,'html.parser')
    
    occupation = ''
    industry = ''
    age = np.nan
    
    # parse article text to find author info: occupation, industry, age
    sections = soup.find_all('div',{'class':'section-text'})
    idx_content_start = 0
    for i, section in enumerate(sections):
        # note where the "diary" actually begins (vs R29 preamble)
        if section.find('h3') and section.text=='Day One':
            idx_content_start = i
        if 'Occupation' in section.text:
            try:
                occupation = re.findall(r'(?<=Occupation:</strong>).*?(?=<br/><strong>)',str(section))[0].strip()
            except:
                try:
                    occupation = re.findall(r'(?<=Occupation: </strong>).*?(?=<br/><strong>)',str(section))[0].strip()
                except: 
                    occupation = occupation # leave occupation missing
            try:
                industry = re.findall(r'(?<=Industry:</strong>).*?(?=<br/><strong>)',str(section))[0].strip()
            except:
                try:
                    industry = re.findall(r'(?<=Industry: </strong>).*?(?=<br/><strong>)',str(section))[0].strip()
                except:
                    industry = industry # leave industry missing
            try:
                age = int(re.findall(r'(?<=Age:</strong>).*?(?=<br/><strong>)',str(section))[0].strip())
            except:
                age = age # leave age missing
        
    # clean "diary" text in preparation for parsing 
    punctuation_table = str.maketrans(dict.fromkeys(string.punctuation + '\u2014')) 
    content_sections = [c.text for c in sections[idx_content_start:-1] if not c.find('h3') and not 'Daily Total:' in c.text]
    content_sections = [re.sub(r'(\$[0-9,]*)([.][0-9]{2})?','',c) for c in content_sections]
    content_sections = [re.sub(r'([0-9]{1,2})([:][0-9]{2})?(\sa.m.)?(\sp.m.)?','',c) for c in content_sections]
    content = ' '.join([c.translate(punctuation_table).strip() for c in content_sections]).lower()
    
    post_dict = {'occupation':occupation,
                 'industry':industry,
                 'age':age,
                 'content':content}
    
    return post_dict

# function to get hrefs to all articles on the nth landing page
def get_page(base_url,n):
    
    page_df = pd.DataFrame()
    page = requests.get(base_url + str(n))
    soup = BeautifulSoup(page.text,'html.parser')
    
    # get all divs which hold links to relevant articles
    cards = soup.find_all('div',{'class':'card'})
    for card in cards:
        card_text = card.find_all('span')[0].text
        
        # if the div holds a link to a (generic-type) money diaries article:
        if 'A Week In' in card_text and "On" in card_text and "$" in card_text:
            
            # get basic article info: location, income type, income amount
            location = card_text.split('In')[1].split('On')[0].strip()
            if location.endswith(','):
                location = location[:-1]
                
            money_str = card_text.split(' In ')[1].split('On')[1].strip()
            is_joint = 'Joint' in money_str
            is_periodic = 'Per' in money_str or '/' in money_str
            if is_periodic:
                period = money_str.split(' ')[-1]
                if '/' in period:
                    period = period.split('/')[1]
            else:
                period = ''
            
            punctuation_table = str.maketrans({'+':''})
            if '/' in money_str:
                amt = float(money_str.split('$')[1].split('/')[0].replace(',','').translate(punctuation_table))
            else:
                amt = float(money_str.split('$')[1].split(' ')[0].replace(',','').translate(punctuation_table))
            
            entry_dict = {'location':location,
                     'amt':amt,
                     'is_joint':is_joint,
                     'is_periodic':is_periodic,
                     'period': period}
            
            # send the article off for parsing
            anchor = card.find('a').get('href')
            post_url = 'https://www.refinery29.com' + anchor
            post_dict = get_post(post_url)
            entry_dict.update(post_dict)
            
            page_df = page_df.append(pd.DataFrame(entry_dict,index=[0]))
    
    return page_df

# load general inquirer dictionary csv
# (see http://www.wjh.harvard.edu/~inquirer/homecat.htm)
gi_dict = pd.read_csv('data/inquirerbasic.csv')

# select relevant word categories for this analysis 
gi_dict = gi_dict[['Entry','Pstv','Ngtv','Strong','Weak','Active','Passive',
                   'Pleasur','Pain','Feel','Arousal','Virtue','Vice']]

# drop words that don't fall into one of the 12 categories above
gi_dict = gi_dict.loc[gi_dict.apply(lambda x: x.count(), axis=1) > 1]

# convert dataframe into boolean
gi_dict = gi_dict.fillna(value='')
for col in ['Pstv','Ngtv','Strong','Weak','Active','Passive',
            'Pleasur','Pain','Feel','Arousal','Virtue','Vice']:
    gi_dict[col] = gi_dict[col].apply(lambda x: len(x)>0)
    
# collapse word with multiple "verisons" into one row
gi_dict['Entry'] = gi_dict['Entry'].str.split('#').str[0]
gi_dict = gi_dict.groupby('Entry').sum()
gi_dict[gi_dict>1] = 1
gi_dict = gi_dict.reset_index()
gi_dict['Entry'] = gi_dict['Entry'].str.lower()

# save cleaned dictionary
gi_dict.to_csv('data/inquirercleaned.csv',index=False)

# get cleaned r29 money diaries article data
base_url = 'https://www.refinery29.com/en-us/money-diary?page=' 
pg_num_list = list(range(1,16))
for n in pg_num_list:
    print(n)
    page_data = get_page(base_url,n)
    try:
        article_data = article_data.append(page_data)
    except NameError:
        article_data = page_data
        
article_data = article_data.reset_index(drop=True)

# clean out entries with a "period" (based on scraping) that doesn't make sense
valid_periods = ['Hour','Day','Week','Month','Year']
idx_invalid_periods = (article_data['is_periodic']==1) & (~article_data['period'].isin(valid_periods))
article_data = article_data.loc[idx_invalid_periods==0]

# compute yearly income amounts
article_data.loc[article_data['is_periodic']==0,'yearly_amt'] = article_data['amt']
pd_multiplier_dict = dict(zip(valid_periods,[8*252,252,52,12,1]))
for key, value in pd_multiplier_dict.items():
    article_data.loc[article_data['period']==key,'yearly_amt'] = article_data['amt']*value

article_data.to_csv('data/article_data.csv',index=False)    