import mysql.connector 
import hashlib
 
import sys	 
import random	

"""
Connect to a MySQL database (connect_db), create user/password records in the following table (create_user), and
authenticate login requests against the table (authenticate_user).
"""

DB_HOST = "localhost"	 
DB_USER = "devel" 
DB_PASS = "devel"	 
DB_NAME = "test"	 
 
def connect_db():	 
    """
    Try to connect DB and return DB instance, if not, return False.
    """
    try:	 
        return mysql.connector.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)	 
    except:	 
        return False	 
