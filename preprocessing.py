import argparse
import json
import os
import multiprocessing as mp
import logging
import html2text
import re
import signal
import csv
from collections import Counter
import pprint
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def worker_init():
    global header_pattern
    global newline_pattern
    header_pattern = re.compile("(^|\n)#{1,}\s+")
    newline_pattern = re.compile("\n\s+\n")

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def scrub(text):
    text = text.replace("part time", "part-time")
    text = text.replace("full time", "full-time")
    text = text.replace("parttime", "part-time")
    text = text.replace("fulltime", "full-time")
    text = text.replace("full-time/part-time", "full-time or part-time ")
    text = text.replace("part-time/full-time", "full-time or part-time ")
    text = re.sub(header_pattern, "\n\n", text)
    text = re.sub(newline_pattern, "\n\n", text)
    text = text.strip()
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return text


def scrub_posting(posting):
    h2t = html2text.HTML2Text()
    h2t.ignore_links = True
    h2t.ignore_images = True
    h2t.ignore_emphasis = True
    h2t.ignore_anchors = True
    h2t.ul_item_mark = "-"
    h2t.body_width = 0
    posting["JD_SCRUBBED"] = scrub(h2t.handle(posting.get("JOBDESCRIPTION","")))
    #try:
    #   posting["JR_SCRUBBED"] = scrub(h2t.handle(posting.get("JOBREQUIREMENT","")))
    #except NameError:
    #    posting["JR_SCRUBBED"] = []
    #try:
    #    posting["JT_SCRUBBED"] = scrub(h2t.handle(posting.get("JOBTITLE","")))
    #except NameError:
    #    posting["JT_SCRUBBED"] = []
    
    
    #try:
    #    posting["CATEGORIES"] = eval(posting["CATEGORIES"])
    #except NameError:
    #    posting["CATEGORIES"] = []
    #try:
    #    posting["INDUSTRIES"] = eval(posting["INDUSTRIES"])
    #except NameError:
    #   posting["INDUSTRIES"] = []
    #try:
    #    posting["DRIVING"] = eval(posting["DRIVING"])
    #except NameError:
    #    posting["DRIVING"] = []
    
    #posting["features"] = {}
    #posting["features"]["parttime"] = float('Part-time' in posting["CATEGORIES"])
    return posting
    
