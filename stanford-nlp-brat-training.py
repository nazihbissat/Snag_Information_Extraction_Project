
# coding: utf-8

# In[1]:


import snowflake
import getpass
import pprint
from snowflake.connector import DictCursor
from preprocessing import *
from postings_ner import *
import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, join
from subprocess import Popen, PIPE
import argparse

CONNECTION_PARAMS = {
    "user":None,
    "password":None,
    "account":'snagajob',
    "authenticator":"https://snagajob.okta.com/",
    "database":"PROD_SAJ_SHARE",
    "warehouse":"PROD_WH",
    "schema":"CUSTOMER"
}


# In[ ]:


# CONNECTION_PARAMS["user"] = input("your snowflake user: ")


# In[ ]:


# CONNECTION_PARAMS["password"] = getpass.getpass("your snowflake password: ")


# In[7]:


# TO USE READY SET OOF POSTINGS - USING FIXED SET FOR NOW
def load_postings():
    with open('1000-postings.pkl', 'rb') as pkl_file:
        results = pickle.load(pkl_file)
    return results


# In[ ]:


### TAKES A DATE (YYYY-MM-DD)
def get_snowflake_data(date_from):
    query_p1 = """select JOBDESCRIPTION from CUSTOMER.DIMJOBPOSTING_VIEW where CREATEDATE >= """
    query_p2 = '::date limit 100;'
    with snowflake.connector.connect(** CONNECTION_PARAMS) as ctx:
        with ctx.cursor(DictCursor) as cs:
            with cs.execute(query) as results:
                results = [r for r in results]
            
    with mp.Pool(mp.cpu_count(), worker_init) as pool:
        try:
            results = pool.map(scrub_posting, results)
        except KeyboardInterrupt:
            pool.terminate()


# In[ ]:


### TO DUMP SET OF POSTINGS FOR FUTURE USE
def dump_postings():
    with open('1000-postings.pkl', 'wb') as output:
        pickle.dump(results, output)


# In[10]:


stfd_entity_types = {'ORGANIZATION': 'Organization', 'LOCATION': 'Location', 'PERSON': 'Person', 'MONEY': 'Money', 
                     'TIME': 'Time', 'DURATION': 'Duration', 'NUMBER': 'Number', 'PERCENT': 'Percent', 
                     'MISC': 'Miscellaneous', 'DATE': 'Date', 'ORDINAL': 'Ordinal'}


# In[4]:


# CREATING POSTING TEXT FILES AND RESPECTIVE ENTITY ANNOTATION FILES FOR BRAT TEXT ANNOTATION TOOL (StanfordNLP)

def insertPeriod(position, mystring):
    longi = len(mystring)
    mystring   =  mystring[:position] + '.' + mystring[position:] 
    return mystring

def postings_to_brat(results, data_dir):
    # StanfordNLP NER
    startup_corenlp_server()
    
    posting_index = 1
    for r in results:
        posting_fname = 'posting' + str(posting_index)
        posting_text = r['JD_SCRUBBED'].strip()
        posting_text = re.sub(r'(\n-)', '\n', posting_text)
        posting_text = re.sub(r'(\n  -)', '\n', posting_text)
        posting_text = re.sub(r'(\n  -)', '\n', posting_text)
        posting_text = re.sub(r'(\n  )', '\n', posting_text)
        
        indices = [x.start() for x in re.finditer(r'\n', posting_text)]

        counter = 0
        for i in indices:
            if posting_text[i+counter-1] != '.':
                if posting_text[i+counter-1] != '\n':
                    posting_text = insertPeriod(i+counter, posting_text)
                    counter += 1
    
#         with open('/Users/nazih.bissat/Desktop/brat-v1.3_Crunchy_Frog/data/Training_NER/StanfordNLP/' + posting_fname + '.txt', 'w') as text_file:
#             text_file.write(posting_text)
#             text_file.close()
        with open(data_dir + '/'  + posting_fname + '.txt', 'w') as text_file:
            text_file.write(posting_text)
            text_file.close()
    
        entity_index = 1
        entity_ann_file_text = ''
        posting_details = annotate_posting(posting_text)
        for s in posting_details['sentences']:
            for e in s['entitymentions']:
                entity_ann_file_text += 'T' + str(entity_index) + '\t' + stfd_entity_types[e['ner']] + ' '                                         + str(e['characterOffsetBegin']) + ' ' + str(e['characterOffsetEnd']) + '\t'                                         + e['text'] + '\n'
                entity_index += 1
        
#         ann_file = open('/Users/nazih.bissat/Desktop/brat-v1.3_Crunchy_Frog/data/Training_NER/StanfordNLP/' + posting_fname + '.ann', 'w')
        ann_file = open(data_dir + posting_fname + '.ann', 'w')
        ann_file.write(entity_ann_file_text)
        ann_file.close()
        
        posting_index += 1


# In[11]:


# results = load_postings()
# postings_to_brat('/Users/nazih.bissat/Desktop/brat-v1.3_Crunchy_Frog/data/Training_NER/StanfordNLP/')


# In[6]:


# A python script to turn annotated data in standoff format (brat annotation tool) to the formats expected by Stanford NER and Relation Extractor models
# - NER format based on: http://nlp.stanford.edu/software/crf-faq.html#a

def compile_training_data(data_dir):
    DEFAULT_OTHER_ANNO = 'O'
# #     IF RUNNING ON MY MACHINE, USE THE PATH BELOW
#     DATA_DIRECTORY = '/Users/nazih.bissat/Desktop/brat-v1.3_Crunchy_Frog/data/Training_NER/StanfordNLP'
    DATA_DIRECTORY = data_dir
    OUTPUT_DIRECTORY = 'stanford-nlp-train'
    
    NER_TRAINING_DATA_OUTPUT_PATH = join(OUTPUT_DIRECTORY, 'stanford-nlp-training-data.tsv')
    
    if os.path.exists(OUTPUT_DIRECTORY):
        if os.path.exists(NER_TRAINING_DATA_OUTPUT_PATH):
            os.remove(NER_TRAINING_DATA_OUTPUT_PATH)
    else:
        os.makedirs(OUTPUT_DIRECTORY)
    
    sentence_count = 0
#     startup_corenlp_server()

    # looping through .ann files in the data directory
    ann_data_files = [f for f in listdir(DATA_DIRECTORY) if isfile(join(DATA_DIRECTORY, f)) and f.split('.')[1] == 'ann']

    for file in ann_data_files:
        entities = []
    
        # process .ann file - place entities and relations into 2 seperate lists of tuples
        with open(join(DATA_DIRECTORY, file), 'r') as document_anno_file:
            lines = document_anno_file.readlines()
            for line in lines:
                standoff_line = line.split()
                entity = {}
                entity['standoff_id'] = int(standoff_line[0][1:])
                entity['entity_type'] = standoff_line[1].capitalize()
                entity['offset_start'] = int(standoff_line[2])
                entity['offset_end'] = int(standoff_line[3])
                entity['word'] = standoff_line[4]
                entities.append(entity)
    
        # read the .ann's matching .txt file and tokenize its text using stanford corenlp
        with open(join(DATA_DIRECTORY, file.replace('.ann', '.txt')), 'r') as document_text_file:
            document_text = document_text_file.read()
    
        output = annotate_posting(document_text)
    
        # write text and annotations into NER
        with open(NER_TRAINING_DATA_OUTPUT_PATH, 'a') as ner_training_data:
            for sentence in output['sentences']:
                entities_in_sentence = {}
                sentence_re_rows = []
    
                for token in sentence['tokens']:
                    offset_start = int(token['characterOffsetBegin'])
                    offset_end = int(token['characterOffsetEnd'])
                    
                    re_row = {}
                    entity_found = False
                    ner_anno = DEFAULT_OTHER_ANNO

                    # searching for token in annotated entities
                    for entity in entities:
                        if offset_start >= entity['offset_start'] and offset_end <= entity['offset_end']:
                            ner_anno = entity['entity_type']
                        
                        # multi-token entities for RE need to be handled differently than NER
                        if offset_start == entity['offset_start'] and offset_end <= entity['offset_end']:
                            entities_in_sentence[entity['standoff_id']] = len(sentence_re_rows)
                            re_row['entity_type'] = entity['entity_type']
                            re_row['pos_tag'] = token['pos']
                            re_row['word'] = token['word']
                        
                            sentence_re_rows.append(re_row)
                            entity_found = True
                            break
                        elif offset_start > entity['offset_start'] and offset_end <= entity['offset_end'] and len(
                                sentence_re_rows) > 0:
                            sentence_re_rows[-1]['pos_tag'] += '/{}'.format(token['pos'])
                            sentence_re_rows[-1]['word'] += '/{}'.format(token['word'])
                            entity_found = True
                            break
                        
                    if not entity_found:
                        re_row['entity_type'] = DEFAULT_OTHER_ANNO
                        re_row['pos_tag'] = token['pos']
                        re_row['word'] = token['word']
                        
                        sentence_re_rows.append(re_row)

                    # writing tagged tokens to NER training data
                    ner_training_data.write('{}\t{}\n'.format(token['word'], ner_anno))

                sentence_count += 1

            ner_training_data.write('\n')

        print('Processed file pair: {} and {}'.format(file, file.replace('.ann', '.txt')))


# In[7]:


##### COMMAND TO TRAIN STANFORDNLP NER: java -cp "stanford-ner.jar:lib/*" -mx4g edu.stanford.nlp.ie.crf.CRFClassifier -prop train/prop.txt

def train_ner_model(stfd_ner_dir):
    ## CHANGE THIS
#     commands = '''
#     cd /Users/nazih.bissat/Desktop/match.fracking/stanford-ner-tagger;
#     java -cp "stanford-ner.jar:lib/*" -mx4g edu.stanford.nlp.ie.crf.CRFClassifier -prop train_stanford_nlp/prop.txt
#     '''
    commands = 'cd '
    commands += stfd_ner_dir + ';'
    commands += '''
    java -cp "stanford-ner.jar:lib/*" -mx4g edu.stanford.nlp.ie.crf.CRFClassifier -prop train_stanford_nlp/prop.txt
    '''
    
    process = Popen('/bin/bash', stdin=PIPE, stdout=PIPE)
    out, err = process.communicate(commands.encode('utf-8'))


# In[19]:


## THIS TAKES A TEXT FILE, RUNS A SPECIFIED NER MODEL ON IT, AND OUTPUTS TO A SPECIFIED OUTPUT DIRECTORY

def posting_ner(stfd_ner_dir, data_dir, f):
#     CHANGE THIS
#     commands = '''
#     cd /Users/nazih.bissat/Desktop/match.fracking/stanford-ner-tagger;
#     java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier postings-ner-model.ser.gz -outputFormat inlineXML -textFile '''
    commands = 'cd '
    commands += stfd_ner_dir + ';'
    commands += '''
    java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier postings-ner-model.ser.gz -outputFormat inlineXML -textFile '''
# 
    commands += data_dir + '/' + f + '.txt > ' + f + '-ner.txt'

    process = Popen('/bin/bash', stdin=PIPE, stdout=PIPE)
    out, err = process.communicate(commands.encode('utf-8'))


# In[20]:


# posting_ner('/Users/nazih.bissat/Desktop/match.fracking/stanford-ner-tagger/', '/Users/nazih.bissat/Desktop/brat-v1.3_Crunchy_Frog/data/Training_NER/StanfordNLP', 'posting1')


# In[12]:


# CREATING POSTING TEXT FILES AND RESPECTIVE ENTITY ANNOTATION FILES FOR BRAT TEXT ANNOTATION TOOL (StanfordNLP)

def reannotate_postings(data_dir, stfd_ner_dir):
#     DATA_DIRECTORY = '/Users/nazih.bissat/Desktop/brat-v1.3_Crunchy_Frog/data/Training_NER/StanfordNLP'
    DATA_DIRECTORY = data_dir
    text_files = [f for f in listdir(DATA_DIRECTORY) if isfile(join(DATA_DIRECTORY, f)) and f.split('.')[1] == 'txt']
    
    for f in text_files:
        file_path = join(DATA_DIRECTORY, f)
        posting_fname = f.split('.')[0]
        
        with open(file_path, 'r') as posting_file:
            posting_text = posting_file.read()
    
        entity_index = 1
        entity_ann_file_text = ''
        posting_details = posting_ner(stfd_ner_dir, DATA_DIRECTORY, posting_fname)
#         with open('/Users/nazih.bissat/Desktop/match.fracking/stanford-ner-tagger/' + posting_fname + '-ner.txt', 'r') as o:
#             ner_output = o.read()
        with open(stfd_ner_dir + posting_fname + '-ner.txt', 'r') as o:
            ner_output = o.read()    
        
        regex = re.compile(r'\<(.*?)>')
        iterator = regex.finditer(ner_output)
        
        ind = list()
        for i in iterator:
            ind.append(i.span())
    
        counter = 0
        for i in np.arange(0, len(ind), 2):
            entity_text = posting_text[(ind[i][0] - counter):(ind[i+1][0] + ind[i][0] - ind[i][1] - counter)]
            entity_type = ner_output[(ind[i][0]+1):(ind[i][1]-1)]
            start_char = ind[i][0] - counter
            end_char = ind[i+1][0] + ind[i][0] - ind[i][1] - counter
            entity_ann_file_text += 'T' + str(entity_index) + '\t' + entity_type + ' '                                         + str(start_char) + ' ' + str(end_char) + '\t'                                         + entity_text + '\n'
            entity_index += 1
            counter += 2 * (ind[i][1] - ind[i][0]) + 1
    
        with open(join(DATA_DIRECTORY, posting_fname + '.ann'), 'w') as ann_file:
            ann_file.write(entity_ann_file_text)
            ann_file.close()
        
        os.remove(stfd_ner_dir + posting_fname + '-ner.txt')


def main(retrain, data_dir, stfd_ner_dir):
    if retrain == False:
        results = load_postings()
        startup_corenlp_server()
        postings_to_brat(results, data_dir)
        compile_training_data(data_dir)
        train_ner_model(stfd_ner_dir)
        reannotate_postings(data_dir, stfd_ner_dir)
        shutdown_corenlp_server()
    else:
        startup_corenlp_server()
        train_ner_model(stfd_ner_dir)
        reannotate_postings(data_dir, stfd_ner_dir)
        shutdown_corenlp_server()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Input the path to the correct folder in the brat data directory and the path to the stanford-ner-tagger directory.')
    parser.add_argument('data_dir', help='Path to the correct folder in the brat data directory')
    parser.add_argument('stfd_ner_dir', help='Path to the stanford-ner-tagger directory.')
    parser.add_argument('--retrain', dest='retrain', action='store_true')
    parser.add_argument('--no-retrain', dest='retrain', action='store_false')
    parser.set_defaults(retrain=True)
    args = parser.parse_args()
    main(args.retrain, args.data_dir, args.stfd_ner_dir)
