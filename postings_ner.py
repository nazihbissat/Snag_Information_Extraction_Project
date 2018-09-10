import json
import atexit
import signal
import os
import subprocess
import time
import logging
import requests
import multiprocessing as mp
#from bson import json_util
from requests.adapters import HTTPAdapter
import gc
# import objgraph
from string import punctuation
from string import digits, whitespace
from collections import Counter, defaultdict
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import pprint
from random import sample
import bs4
from langdetect.detector_factory import init_factory


def filter_token(tok):
    char_freqs = defaultdict(lambda: 0, Counter(list(tok)))
    tok_len = len(tok)
    n_punct = sum([char_freqs[c] for c in punctuation + digits + whitespace ])
    return n_punct/float(tok_len) > 0.50


def load_stopwords(stopwords_path="./resources/stopwords.lex"):
    with open(stopwords_path, "r") as f:
        stopwords = {w.strip("\n") for w in f.readlines()}

    return stopwords

def worker_init_corpus(stops_in):
    global sess
    global stops
    sess = requests.Session()
    requests.mount("http://", HTTPAdapter(max_retries=10))
    stops = stops_in
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    init_factory()


def scrub(text):
    text = text.replace("part time", "part-time")
    text = text.replace("full time", "full-time")
    text = text.replace("parttime", "part-time")
    text = text.replace("fulltime", "full-time")
    text = text.replace("full-time/part-time", "full-time or part-time ")
    text = text.replace("part-time/full-time", "full-time or part-time ")
    return text


def annotate_posting_with_regex(posting_text):
    posting_text = scrub(posting_text)
    pipeline = '{"annotators": "tokenize,ssplit,pos,ner,regexner,entitymentions", "ner.useSUTime": "false", "outputFormat": "json"}'
    server_url = 'http://localhost:9000/?properties='+pipeline
    req = requests.post(data=posting_text.encode("utf-8"), url=server_url)
    return req.json()

def annotate_posting(posting_text):
    posting_text = scrub(posting_text)
    pipeline = '{"annotators": "tokenize,ssplit,pos,ner,entitymentions", "ner.useSUTime": "false", "outputFormat": "json", "ner.applyFineGrained": "false"}'
    server_url = 'http://localhost:9000/?properties='+pipeline
    req = requests.post(data=posting_text.encode("utf-8"), url=server_url)
    return req.json()

def depparse_posting(posting_text):
    posting_text = scrub(posting_text)
    pipeline = '{"annotators": "tokenize,ssplit,pos,depparse", "outputFormat": "json"}'
    server_url = 'http://localhost:9000/?properties='+pipeline
    req = requests.post(data=posting_text.encode("utf-8"), url=server_url)
    return req.json()

def parse_posting(posting_text):
    posting_text = scrub(posting_text)
    pipeline = '{"annotators": "tokenize,ssplit,pos,parse"}'
    server_url = 'http://localhost:9000/?properties='+pipeline
    req = requests.post(data=posting_text.encode("utf-8"), url=server_url)
    return req.json()

## TO USE ON LOCAL MACHINE, CHANGE DIRECTORY TO '/Users/nazih.bissat/Desktop/stanford-corenlp-python/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar:/Users/nazih.bissat/Desktop/stanford-corenlp-python/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar'
def startup_corenlp_server():
    args = ['nohup',
    'java',
     '-mx10g',
     '-cp',
     '/Users/nazih.bissat/Desktop/stanford-corenlp-python/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar:/Users/nazih.bissat/Desktop/stanford-corenlp-python/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar',
     'edu.stanford.nlp.pipeline.StanfordCoreNLPServer',
     '-port',
     '9000',
     '-timeout',
     '600000',
     '&']
    p = subprocess.Popen(args)
    time.sleep(10)
    logging.info("CoreNLP Server running, process id: {}".format(p.pid))


@atexit.register
def shutdown_corenlp_server():
    try:
        with open("/tmp/corenlp.shutdown", "r") as f:
            shutdown_key = f.read()
            requests.get("http://localhost:9000/shutdown?key="+shutdown_key)
        logging.info("CoreNLP Server shutdown")
    except:
        pass