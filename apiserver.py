import auxiliary
from flask import Flask, render_template
from flask import jsonify
from flask import request
from flask_cors import CORS, cross_origin
import json
import pandas as pd
from stockstats import StockDataFrame
import model
import flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
