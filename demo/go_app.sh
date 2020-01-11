#!/usr/bin/env bash
#@Author  : fanzfeng
cid=$(lsof -i:8000 | sed -n "2, 1p" | awk '{print $2}')
echo $cid
kill $cid
python3 manage.py runserver localhost:8000