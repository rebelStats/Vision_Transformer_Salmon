#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 22:27:16 2022

@author: user
"""

import requests


resp = requests.post('http://0.0.0.0:5000/predict', files={'file': open('salmon.jpg', 'rb')})

print(resp.text)