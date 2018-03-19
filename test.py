#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pyodbc
import base64
from sqlserver_connector import Connector
import json

import sys
reload(sys)
sys.setdefaultencoding('utf-8')



if __name__ == '__main__':

    cntr = Connector()
    #cntr.saveQuestion("test2", "a1", values)
    #cntr.updateQuestion("test", "a1", "3")
    #cntr.getQuestionCount("test","a1")

    print cntr.showTableDesc('case_main')